# Algorithm Documentation Summary

This directory contains comprehensive documentation of the NPAG and BestDose algorithms in both Fortran and Rust implementations.

## Files

### 1. [NPAGFULL_overview.md](./NPAGFULL_overview.md)

**Complete explanation of NPAGFULL vs NPAGFULL11**

**Key findings**:

- **NPAGFULL**: Full iterative optimization, returns single best parameter estimate (MAP)
- **NPAGFULL11**: Bayesian filtering only (0 cycles), returns all support points with probability > 1e-100 of maximum
- **BestDose usage**: Two-step hybrid approach
  1. NPAGFULL11 identifies compatible parameter regions (fast)
  2. NPAGFULL refines each region (accurate)
  3. Result: Multiple refined points representing uncertainty

**Contents**:

- Mathematical framework (Bayesian updates, optimization)
- Convergence criteria and computational complexity
- Practical implications and use cases
- Version history and implementation details

---

### 2. [Rust_vs_Fortran_NPAG.md](./Rust_vs_Fortran_NPAG.md)

**Line-by-line comparison of implementations**

**Key findings**:

- Fortran: Subject-level Bayesian calculator (single subject, configurable cycles)
- Rust: Population-level parameter estimator (multiple subjects, always iterative)
- Both achieve same mathematical goals with different programming paradigms

**14 detailed comparisons**:

1. Architecture (procedural vs trait-based)
2. Convergence strategies (resolve vs epsilon)
3. Grid management (explicit vs functional)
4. Optimization methods (3-point vs 2-point gamma search)
5. Likelihood calculation (sequential vs parallel)
6. Interior point method (stateful vs functional)
7. Cycle structure (GOTO vs method-based)
8. Error handling (STOP vs Result)
9. Data structures (arrays vs types)
10. Type safety (implicit vs strong)
11. Memory management (manual vs RAII)
12. Parallelization (external vs built-in)
13. Testing (manual vs integrated)
14. Design philosophy (1960s-1990s vs 2010s-2020s)

---

### 3. [BestDose_algorithm.md](./BestDose_algorithm.md)

**Complete BestDose dose optimization algorithm**

**Three-stage process**:

#### Stage 1: Posterior Density Calculation

```
Prior (N points)
    → NPAGFULL11 (Bayesian filter)
    → M compatible points
    → NPAGFULL (refine each)
    → M refined points with probabilities
```

#### Stage 2: Dose Optimization

```
Cost = (1-λ) × Variance + λ × Bias²

Variance = Σᵢ P(θᵢ|past) × Σⱼ (Targetⱼ - Pred(θᵢ,dose))²
Bias² = Σⱼ (Targetⱼ - Ȳⱼ)²  where Ȳⱼ = Σᵢ P(θᵢ) × Pred(θᵢ,dose)

Minimize via Nelder-Mead simplex
```

#### Stage 3: Prediction and Output

```
Optimal dose(s) → Predictions → Metrics (variance, bias, combined cost)
```

**Key features**:

- Handles missing past data
- Multiple doses and routes
- Concentration or AUC targets
- Bias weight (λ) for safety/personalization balance

---

## Quick Reference

### When to use which NPAG variant?

| Need                      | Use           | Why                                   |
| ------------------------- | ------------- | ------------------------------------- |
| Final parameter estimates | NPAGFULL      | Full optimization, single best point  |
| Quick screening           | NPAGFULL11    | Fast Bayesian filter, multiple points |
| Dose optimization         | Hybrid (both) | Balance accuracy and uncertainty      |
| Population analysis       | Rust NPAG     | Multiple subjects, parallel           |

### Cost function weights (λ)

| λ value | Behavior             | Use when                                   |
| ------- | -------------------- | ------------------------------------------ |
| 0.0     | Full personalization | Good patient data, individual optimization |
| 0.5     | Balanced             | Moderate data, want safety margin          |
| 1.0     | Population-based     | Poor/no data, high risk, conservative      |

### Algorithm complexity

| Component            | Time        | Notes                      |
| -------------------- | ----------- | -------------------------- |
| NPAGFULL11           | Seconds     | Single evaluation pass     |
| NPAGFULL (per point) | 10-60s      | Full optimization cycles   |
| Dose optimization    | 30-300s     | ~100-1000 cost evaluations |
| **Total BestDose**   | **1-5 min** | Patient-specific dosing    |

---

## Mathematical Foundations

### Bayesian Posterior (NPAGFULL11)

```
P(θᵢ|data) = P(data|θᵢ) × P(θᵢ) / P(data)

Keep if: P(θᵢ|data) > 1e-100 × max{P(θⱼ|data)}
```

### Maximum Likelihood (NPAGFULL)

```
Maximize: L = Σⱼ log P(yⱼ|θ)

Converges when: |Δobjective| < 0.01
```

### Dose Optimization (BestDose)

```
Minimize: (1-λ) × E[error²] + λ × (target - E[pred])²
```

---

## Key Insights

### 1. Two-Step Posterior in BestDose

**Why not just NPAGFULL?**

- Returns only 1 point → loses uncertainty
- Need multiple plausible scenarios for robust dosing

**Why not just NPAGFULL11?**

- Points not optimized → suboptimal solutions
- Grid resolution limits accuracy

**Hybrid solution:**

- NPAGFULL11: Identifies promising regions (5-50 points)
- NPAGFULL: Refines each region to local optimum
- Result: Multiple refined points spanning parameter uncertainty

### 2. Variance vs Bias Trade-off

**Variance term**:

- "How much do predictions vary across plausible parameters?"
- Patient-specific uncertainty
- Minimizes expected prediction error

**Bias term**:

- "How far is patient from population average?"
- Population-level constraint
- Prevents extreme doses for outliers

**Balance via λ**: Clinician controls personalization vs conservatism

### 3. Parallel Fortran vs Rust

**Fortran sequential**:

- NPAGFULL processes subjects one at a time
- Each support point evaluated sequentially
- External parallelism possible but not built-in

**Rust parallel**:

- `par_iter()` automatically distributes work
- Subjects and support points both parallelized
- Near-linear speedup on multi-core systems

---

## Code Organization

### Fortran (LAPKB/PMcore/Fortran/)

```
bestdose.for         - Main program (~30,000 lines)
NPAGFULLA.FOR        - NPAGFULL implementation (~9,000 lines)
NPAGFULLA11.FOR      - NPAGFULL11 implementation (~1,200 lines)
CALCBST15.FOR        - Cost function calculations
IDM1X15.FOR          - ODE integration
IDM3X151.FOR         - Multi-drug ODE integration
BLASNPAG.FOR         - Linear algebra routines
```

### Rust (LAPKB/PMcore/src/)

```
algorithms/
    npag.rs          - NPAG struct and Algorithms trait impl
bestdose/
    mod.rs           - BestDose optimization
routines/
    evaluation/      - IPM (burke), QR decomposition
    expansion/       - Adaptive grid
    initialization/  - Prior sampling
    optimization/    - (Future: additional methods)
structs/
    psi.rs           - Likelihood matrix
    theta.rs         - Support points
    weights.rs       - Probability weights
```

---

## Implementation Status

### Fortran (Complete)

✅ NPAGFULL and NPAGFULL11
✅ BestDose with all features
✅ File I/O and reporting
✅ Steady-state handling
✅ Multiple error models
✅ Concentration and AUC targets

### Rust (Core Complete)

✅ NPAG algorithm (population-level)
✅ BestDose optimization core
✅ Parallel computation
✅ Type-safe abstractions
⚠️ No NPAGFULL11 mode (could add easily)
⚠️ Limited I/O (by design - library focus)

### Missing in Rust (Future Work)

- [ ] Zero-cycle Bayesian mode (NPAGFULL11 equivalent)
- [ ] Complete file I/O scaffolding
- [ ] Report generation
- [ ] Three-point gamma search (currently two-point)

---

## Usage Guidelines

### For Dose Optimization

1. **Prepare prior density**: Run NPAG on population data
2. **Optional: Collect past data**: Patient's historical doses and concentrations
3. **Define targets**: Future dosing times and target concentrations/AUCs
4. **Set constraints**: Dose range, bias weight (λ)
5. **Run BestDose**: Obtains optimal dose(s) and predictions

### For Parameter Estimation

1. **Population**: Use Rust NPAG for speed and parallelism
2. **Individual**: Use Fortran NPAGFULL for single-subject MAP estimate
3. **Bayesian posterior**: Use Fortran NPAGFULL11 for uncertainty quantification

---

## References

### Original Fortran Code

- BESTDOS121.FOR (June 2016)
- NPAGFULLA.FOR (June 2014)
- NPAGFULLA11.FOR (June 2014)
- Based on NPAG algorithm by Roger Jelliffe and Alan Schumitzky (USC)

### Rust Implementation

- PMcore library (2024-2025)
- Modern reimplementation with parallel computation
- Maintains mathematical fidelity to original algorithms

### Key Papers (Referenced in Code)

- Interior Point Method for NPAG
- Non-parametric population modeling
- Bayesian dose optimization

---

## Contact and Contributions

This documentation was created through detailed analysis of:

- Fortran source code (~40,000 lines)
- Rust implementation (~5,000 lines)
- Algorithm comments and version history
- Mathematical derivations in code

For questions or contributions, refer to the PMcore repository.

---

## Glossary

**NPAG**: Non-Parametric Adaptive Grid - population PK/PD modeling algorithm  
**IPM**: Interior Point Method - optimization algorithm for probability weights  
**MAP**: Maximum A Posteriori - most likely parameter estimate  
**Support point**: Discrete point in parameter space with associated probability  
**Prior density**: Population parameter distribution before seeing patient data  
**Posterior density**: Updated distribution after incorporating patient data  
**Psi matrix**: Likelihood of observations given each support point  
**Lambda**: Probability weights from IPM optimization  
**Theta**: Matrix of support point parameter values  
**Gamma**: Error model parameter (variance or SD)  
**Resolve**: Adaptive grid expansion parameter (Fortran)  
**Epsilon**: Convergence tolerance parameter (Rust)

---

## New Documentation

### 7. [Rust_BestDose_Fixes.md](./Rust_BestDose_Fixes.md)

**Complete documentation of bug fixes in Rust BestDose implementation**

**Fixed bugs**:

1. Missing two-step posterior calculation
2. Incorrect probability usage in cost function
3. Wrong posterior weights in final predictions
4. Performance issues from repeated burke() calls

**Result**: ~100-1000x speedup and correct algorithm implementation

---

### 8. [Cost_Function_Fix.md](./Cost_Function_Fix.md)

**Side-by-side before/after comparison of cost function**

Shows exactly what was wrong and how it was fixed, with Fortran equivalence verification.

---

### 9. [NPAGFULL_Implementation_Complete.md](./NPAGFULL_Implementation_Complete.md) ✨

**🎉 Complete implementation guide - START HERE for usage!**

**Full two-step posterior now implemented**:

- ✅ NPAGFULL11 (Bayesian filtering)
- ✅ NPAGFULL (individual point refinement)
- ✅ Optional refinement flag for speed vs accuracy tradeoff

**Includes**:

- Quick start examples
- Performance characteristics
- Testing recommendations
- Comparison with Fortran
- Usage patterns for fast vs accurate modes

**Status**: Feature-complete! (except AUC targets)

---

_Last updated: October 2025_
