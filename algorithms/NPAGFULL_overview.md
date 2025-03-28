# NPAG Algorithm Overview

## Purpose

The Non-Parametric Adaptive Grid (NPAG) algorithm is a population pharmacokinetic/pharmacodynamic modeling approach that estimates a non-parametric joint probability density of model parameters without assuming a specific parametric distribution (e.g., normal, log-normal).

## Core Concept

NPAG maintains a discrete grid of support points in parameter space, where each point represents a plausible set of parameter values with an associated probability. The algorithm iteratively refines this grid to maximize the likelihood of observing the measured drug concentrations.

---

## NPAGFULL vs NPAGFULL11: Key Differences

### NPAGFULL (Full Optimization)

**Purpose**: Obtains the **single best** posterior parameter estimate for a subject given their data and a prior density.

**Process**:

1. Starts with a prior density (grid of support points with probabilities)
2. Reads subject data (doses and observations)
3. Runs **full NPAG optimization cycles** (up to MAXCYC iterations)
4. Performs:
   - **Evaluation**: Calculates likelihood for each support point
   - **Optimization**: Uses interior point method (IPM) to optimize probabilities
   - **Gamma optimization**: Optimizes error model parameters
   - **Condensation**: Removes support points with negligible probability
   - **Expansion**: Creates new support points around promising regions
5. Iterates until convergence or maximum cycles reached
6. **Returns**: A single optimal support point (the MAP estimate - Maximum A Posteriori)

**Key Characteristics**:

- Full iterative optimization
- Computationally intensive
- Convergence criteria: `|Δ objective function| ≤ 0.01`
- Returns **one** refined parameter vector
- Best for final parameter estimation

---

### NPAGFULL11 (Bayesian Posterior Selection)

**Purpose**: Obtains **all support points** from the prior that are reasonably compatible with the subject's data (Bayesian posterior).

**Process**:

1. Starts with a prior density (grid of support points with probabilities)
2. Reads subject data (doses and observations)
3. **Does NOT run optimization cycles** (MAXCYC hardcoded to 0)
4. Performs only:
   - **Evaluation**: Calculates likelihood P(data|parameters) for each support point
   - **Bayesian update**: Calculates posterior P(parameters|data) = P(data|parameters) × P(parameters) / P(data)
5. **Filters support points**: Keeps all points where posterior probability > 1e-100 × max_probability
6. **Returns**: Multiple support points with their posterior probabilities

**Key Characteristics**:

- Single-pass Bayesian calculation (no iterative optimization)
- Very fast (no cycles)
- Threshold: Keeps points within 1e-100 of maximum posterior probability
- Returns **multiple** support points representing uncertainty
- Best for maintaining uncertainty in dose optimization

---

## Use in BestDose

BestDose uses a **two-step hybrid approach** to balance accuracy and uncertainty quantification:

### Step 1: NPAGFULL11 (Broad Posterior)

```
Prior Density (e.g., 1000 points)
    ↓
NPAGFULL11 (Bayesian filtering)
    ↓
Filtered Posterior (e.g., 5-20 points that match patient data reasonably well)
```

- Identifies which regions of the prior are compatible with patient's past data
- Retains uncertainty by keeping multiple plausible parameter sets

### Step 2: NPAGFULL (Refinement)

```
For each of the 5-20 points from Step 1:
    ↓
NPAGFULL (full optimization starting from this point)
    ↓
One refined "daughter" point per parent
    ↓
Final Posterior: 5-20 refined points
```

- Refines each plausible region to find its local optimum
- Each refined point represents a distinct mode in the posterior

### Why This Approach?

**Problem with NPAGFULL alone**:

- Returns only 1 point → loses uncertainty information
- Dose optimization needs uncertainty to balance efficacy vs. variability

**Problem with NPAGFULL11 alone**:

- Points are not optimized → may not be precisely at likelihood maxima
- Could miss better solutions near original grid points

**Hybrid Solution**:

- NPAGFULL11: Identifies promising regions (fast)
- NPAGFULL: Refines each region (accurate)
- Result: Multiple refined points representing posterior uncertainty

---

## Mathematical Framework

### Bayesian Update (NPAGFULL11)

For each support point θᵢ in the prior:

```
P(θᵢ|data) = P(data|θᵢ) × P(θᵢ) / P(data)

where:
- P(θᵢ) = prior probability from NPAG density file
- P(data|θᵢ) = likelihood calculated from error model
- P(data) = Σ P(data|θⱼ) × P(θⱼ) for all j (normalization)
```

Filter criterion:

```
Keep θᵢ if P(θᵢ|data) > 1e-100 × max{P(θⱼ|data)}
```

### Optimization (NPAGFULL)

Maximizes log-likelihood:

```
L = Σ log P(yⱼ|θ)

where:
- yⱼ = observations for subject j
- P(yⱼ|θ) = Σᵢ P(yⱼ|θᵢ) × w(θᵢ)
- w(θᵢ) = optimized probability weights
```

Uses:

- **Interior Point Method (IPM)** for weight optimization
- **Adaptive grid expansion** to explore parameter space
- **Convergence detection** based on objective function stability

---

## Practical Implications

### When to Use NPAGFULL

- Final parameter estimation for reporting
- When computational time is not limiting
- When you need the single "best" parameter estimate
- For model validation and diagnostics

### When to Use NPAGFULL11

- Quick screening of which prior points are compatible
- As first step in two-stage estimation (as in BestDose)
- When you need to maintain multiple plausible scenarios
- For uncertainty quantification

### When to Use the Hybrid (BestDose)

- **Dose optimization** where you need to balance:
  - Target achievement (needs accuracy)
  - Robustness to uncertainty (needs multiple scenarios)
- When you want refined estimates that still represent uncertainty
- For patient-specific dosing with limited data

---

## Computational Complexity

| Algorithm         | Cycles   | Likelihood Evaluations | Time          |
| ----------------- | -------- | ---------------------- | ------------- |
| NPAGFULL          | 1-100+   | High (many iterations) | Minutes-Hours |
| NPAGFULL11        | 0        | Low (single pass)      | Seconds       |
| Hybrid (BestDose) | Variable | Medium (N × NPAGFULL)  | Minutes       |

where N = number of points retained by NPAGFULL11 (typically 5-20)

---

## Implementation Details

### Common Elements (Both Algorithms)

1. **File I/O**: Read patient data from working copy files
2. **Steady-state handling**: Convert steady-state dose indicators to explicit doses
3. **Error models**: Support multiple output equations with various error structures
4. **Integration**: Numerical integration over parameter space
5. **ODE solving**: LSODA/DVODE for PK/PD differential equations

### NPAGFULL-Specific

- **emint**: Interior point method for probability optimization
- **checkd**: Minimum distance check for new support points
- **Expansion routine**: Adaptive grid refinement
- **Convergence testing**: Multiple criteria (objective function, resolve parameter)

### NPAGFULL11-Specific

- **Early exit**: Jumps to label 900 immediately after likelihood calculation
- **Threshold filtering**: 1e-100 cutoff (vs 1e-10 in older versions)
- **Minimal output**: No cycle logs or iteration statistics

---

## Version History Notes

### NPAGFULLA (used in BestDose 119+)

- Updated from NPAGFULL
- Supports steady-state dosing in "past" data
- Enhanced error handling (writes to ERRORxxxx files)
- Max output equations: 7 (up from 6)

### NPAGFULLA11 (used in BestDose 119+)

- Updated from NPAGFULL11
- Changed threshold from 1e-10 to 1e-100 (keeps more points)
- Synchronized with NPAGFULLA for steady-state and error handling
- MAXCYC hardcoded to 0 (no argument needed)

---

## Summary Table

| Feature         | NPAGFULL                | NPAGFULL11                  |
| --------------- | ----------------------- | --------------------------- |
| **Cycles**      | Multiple (up to MAXCYC) | Zero (MAXCYC = 0)           |
| **Output**      | 1 refined point         | Multiple filtered points    |
| **Method**      | Full optimization       | Bayesian filtering only     |
| **Speed**       | Slow                    | Fast                        |
| **Accuracy**    | High (optimized)        | Moderate (prior grid)       |
| **Uncertainty** | Lost (1 point)          | Preserved (multiple points) |
| **Use Case**    | Final estimation        | Preliminary screening       |
| **In BestDose** | Step 2 (refinement)     | Step 1 (filtering)          |
