# BestDose Algorithm: Detailed Analysis

## Purpose

BestDose optimizes future dosing regimens to achieve target drug concentrations (or AUCs) while accounting for:

1. **Patient history** ("past" data): Previous doses and observed concentrations
2. **Population variability**: Non-parametric density of parameter values
3. **Prediction uncertainty**: Multiple plausible parameter sets for the patient

---

## Algorithm Overview

### Input

1. **Prior density**: Non-parametric distribution from population NPAG analysis
   - Grid of parameter support points (e.g., 1000 points)
   - Each point has a probability weight
2. **Past data** (optional): Patient's historical doses and concentrations
3. **Future template**: Proposed dosing regimen with target concentrations
4. **Model**: Pharmacokinetic/pharmacodynamic differential equations
5. **Settings**:
   - Target type: Concentration or AUC
   - Dose range constraints
   - Bias weight (λ)

### Output

1. **Optimal dose(s)**: Best dose amount(s) to achieve targets
2. **Predictions**: Concentration-time profiles for optimal doses
3. **Performance metrics**:
   - Variance (expected squared prediction error)
   - Bias (squared difference from population mean)
   - Combined cost function

---

## Three-Stage Process

### Stage 1: Posterior Density Calculation

#### Step 1.1: Bayesian Filtering (NPAGFULL11)

```
Prior Density (N points)
    ↓
Calculate P(past_data | θᵢ) for each point θᵢ
    ↓
Apply Bayes' Rule: P(θᵢ | past_data) ∝ P(past_data | θᵢ) × P(θᵢ)
    ↓
Filter: Keep points where P(θᵢ | past_data) > 1e-100 × max
    ↓
Filtered Posterior (M points, typically 5-50)
```

**Purpose**: Identify which regions of parameter space are consistent with patient's past

**Code** (Fortran):

```fortran
CALL NPAGFULL11(MAXSUB,MAXGRD,MAXDIM,NVAR,NUMEQT,WORK,WORKK,
     CORDEN,NDIM,MF,RTOL,ATOL,NOFIX,IRAN,VALFIX,AB,ierrmod,GAMLAM,
     NGRID,NACTVE,PYJGX,DENSTOR,CORDLAST)
```

**Result**:

- `CORDEN(1:NACTVE, 1:NVAR)` = parameter values for M points
- `CORDEN(1:NACTVE, NVAR+1)` = posterior probabilities

#### Step 1.2: Local Optimization (NPAGFULL)

```
For each of M filtered points:
    ↓
    Use as starting point for NPAGFULL
    ↓
    Run full NPAG optimization (multiple cycles)
    ↓
    Find refined "daughter" point (local optimum)
    ↓
End loop
    ↓
Final Posterior: M refined points with probabilities from NPAGFULL11
```

**Purpose**: Refine each plausible parameter region to its local maximum

**Code** (Fortran):

```fortran
DO IACTIVE = 1,NACTVE
    ! Set CORD1 to single point from NPAGFULL11
    DO J = 1,NVAR+1
        CORD1(1,J) = CORDEN(IACTIVE,J)
    END DO

    REWIND(27)  ! Reset patient data file
    NACTVE1 = 1

    ! Optimize starting from this point
    CALL NPAGFULL(MAXSUB,MAXGRD,MAXDIM,NVAR,NUMEQT,WORK,WORKK,CORD1,
         NDIM,MF,RTOL,ATOL,NOFIX,IRAN,VALFIX,AB,ierrmod,GAMLAM,NGRID,
         NACTVE1,PYJGX,DENSTOR,CORDLAST,MAXCYC)

    ! Update CORDEN with refined point (but keep original probability!)
    DO J = 1,NVAR
        CORDEN(IACTIVE,J) = CORD1(1,J)
    END DO
END DO
```

**Critical Detail**:

- Only parameter values updated (`J = 1:NVAR`)
- Probabilities from NPAGFULL11 **preserved** (`NVAR+1` not updated)
- This maintains uncertainty weighting from Bayesian step

#### Normalization

```fortran
SUMD = 0.D0
DO I = 1,NACTVE
    SUMD = SUMD + CORDEN(I,NVAR+1)
    DO J = 1,NVAR
        DENSITY(I,J) = CORDEN(I,J)
    END DO
END DO

DO I = 1,NACTVE
    DENSITY(I,NVAR+1) = CORDEN(I,NVAR+1)/SUMD
END DO
```

**Result**: `DENSITY` array with M support points, normalized probabilities

---

### Stage 2: Dose Optimization

#### Cost Function

The optimization minimizes a weighted combination of variance and bias:

```
Cost = (1 - λ) × Variance + λ × Bias²
```

where λ ∈ [0, 1] is the bias weight parameter.

##### Variance Term (Expected Prediction Error)

```
Variance = Σᵢ P(θᵢ | past) × Σⱼ (Obsⱼ - Pred(θᵢ, dose))²
```

**Interpretation**:

- Expected squared difference between targets and predictions
- Averaged over all plausible parameter sets
- Weighted by posterior probability
- **Represents uncertainty in patient's parameters**

##### Bias Term (Population Average Deviation)

```
Bias² = Σⱼ (Obsⱼ - Ȳⱼ)²

where:
Ȳⱼ = Σᵢ P_prior(θᵢ) × Pred(θᵢ, dose)
```

**Interpretation**:

- Squared difference between targets and **population mean** prediction
- Uses **prior** (not posterior) probabilities
- **Represents deviation from typical patient**

#### Why Two Terms?

**Variance alone** (λ = 0):

- Minimizes expected error for **this specific patient**
- Can lead to extreme doses if patient is unusual
- Focuses on individual optimization

**Bias alone** (λ = 1):

- Targets population-average behavior
- Conservative, avoids extreme doses
- Ignores patient-specific information

**Weighted combination**:

- Balance patient-specific vs population-typical
- Default λ = 0 (full personalization)
- Increase λ for safety/conservativeness

#### Implementation (Fortran)

```fortran
FUNCTION COST(DOSE_VECTOR)
    ! Modify future subject with new dose(s)
    ! ... code to set doses ...

    ! Build observation vector
    ! obs_vec = target concentrations from future file

    variance = 0.0
    y_bar(:) = 0.0

    ! Loop over support points
    DO I = 1, NACTVE
        theta_i = DENSITY(I, 1:NVAR)
        posterior_prob = DENSITY(I, NVAR+1)
        prior_prob = PRIOR_DENSITY(I, NVAR+1)

        ! Simulate with this parameter set
        predictions = SIMULATE(theta_i, future_doses)

        ! Accumulate variance
        sumsq = 0.0
        DO J = 1, N_OBS
            sumsq = sumsq + (obs_vec(J) - predictions(J))**2
            y_bar(J) = y_bar(J) + prior_prob * predictions(J)
        END DO
        variance = variance + posterior_prob * sumsq
    END DO

    ! Calculate bias
    bias = 0.0
    DO J = 1, N_OBS
        bias = bias + (obs_vec(J) - y_bar(J))**2
    END DO

    ! Combined cost
    COST = (1.0 - lambda) * variance + lambda * bias
END FUNCTION
```

#### Implementation (Rust)

```rust
impl CostFunction for BestDoseProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        // Modify target subject with new dose(s)
        let mut target_subject = self.target.clone();
        let mut dose_number = 0;

        for occasion in target_subject.iter_mut() {
            for event in occasion.iter_mut() {
                match event {
                    Event::Bolus(bolus) => {
                        bolus.set_amount(param[dose_number]);
                        dose_number += 1;
                    }
                    Event::Infusion(infusion) => {
                        infusion.set_amount(param[dose_number]);
                        dose_number += 1;
                    }
                    Event::Observation(_) => {}
                }
            }
        }

        // Build observation vector
        let obs_vec: Vec<f64> = target_subject
            .occasions()
            .iter()
            .flat_map(|occ| occ.events())
            .filter_map(|event| match event {
                Event::Observation(obs) => obs.value(),
                _ => None,
            })
            .collect();

        let n_obs = obs_vec.len();
        let mut variance = 0.0_f64;
        let mut y_bar = vec![0.0_f64; n_obs];

        // Iterate over support points
        for ((row, post_prob), pop_prob) in self
            .theta
            .matrix()
            .row_iter()
            .zip(self.prior.iter())
            .zip(self.prior.iter())  // Second prior for population mean
        {
            let spp = row.iter().copied().collect::<Vec<f64>>();

            // Simulate
            let pred = self.eq.simulate_subject(&target_subject, &spp, None)?;
            let preds_i: Vec<f64> = pred.0.flat_predictions();

            // Accumulate variance (using posterior)
            let mut sumsq_i = 0.0_f64;
            for (j, &obs_val) in obs_vec.iter().enumerate() {
                let pj = preds_i[j];
                let se = (obs_val - pj).powi(2);
                sumsq_i += se;
                y_bar[j] += pop_prob * pj;  // Using prior for population mean
            }
            variance += post_prob * sumsq_i;
        }

        // Calculate bias
        let mut bias = 0.0_f64;
        for (j, &obs_val) in obs_vec.iter().enumerate() {
            bias += (obs_val - y_bar[j]).powi(2);
        }

        // Combined cost
        let cost = (1.0 - self.bias_weight) * variance + self.bias_weight * bias;

        Ok(cost)
    }
}
```

#### Optimization Method: Nelder-Mead Simplex

**Algorithm**: Direct search (derivative-free)

- Suitable for non-smooth cost functions
- Handles multiple doses simultaneously
- No gradient calculation needed

**Initialization**:

```rust
fn create_initial_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let num_dimensions = initial_point.len();
    let perturbation_percentage = 0.008;  // 0.8%

    let mut vertices = Vec::new();
    vertices.push(initial_point.to_vec());  // First vertex

    // Create n additional vertices
    for i in 0..num_dimensions {
        let perturbation = if initial_point[i] == 0.0 {
            0.00025  // Special case for zero
        } else {
            perturbation_percentage * initial_point[i]
        };

        let mut perturbed_point = initial_point.to_owned();
        perturbed_point[i] += perturbation;
        vertices.push(perturbed_point);
    }

    vertices  // n+1 vertices for n-dimensional optimization
}
```

**Starting point**: Midpoint of dose range

```rust
let initial_guess = (self.doserange.min + self.doserange.max) / 2.0;
```

**Constraints**:

- Min/max dose bounds enforced by simplex boundaries
- Invalid doses (< min or > max) penalized by high cost

**Termination**:

- Maximum iterations (1000)
- Or simplex size < tolerance

---

### Stage 3: Prediction and Output

After optimization finds optimal dose(s):

#### Prediction for Optimal Doses

```rust
// Set optimal doses in target subject
for (event, &dose) in target_subject.iter_mut()
    .zip(optimal_doses.iter())
{
    match event {
        Event::Bolus(b) => b.set_amount(dose),
        Event::Infusion(i) => i.set_amount(dose),
        _ => {}
    }
}

// Calculate psi for these doses
let psi = calculate_psi(
    &self.eq,
    &Data::new(vec![target_subject.clone()]),
    &self.theta,
    &self.error_models,
    false,
    true,
)?;

// Get optimal weights (IPM)
let (w, likelihood) = burke(&psi)?;

// Calculate posterior
let posterior = Posterior::calculate(&psi, &w)?;

// Generate predictions
let predictions = NPPredictions::calculate(
    &self.eq,
    &Data::new(vec![target_subject]),
    self.theta.clone(),
    &w,
    &posterior,
    0.0,
    0.0,
)?;
```

#### Output Structure

```rust
pub struct BestDoseResult {
    pub dose: Vec<f64>,           // Optimal dose amount(s)
    pub objf: f64,                // Final objective function value
    pub status: String,           // Optimization termination status
    pub preds: NPPredictions,     // Predicted concentration-time profiles
}
```

**NPPredictions includes**:

- Population prediction (using all support points)
- Individual predictions (for each support point)
- Confidence/credible intervals
- Observed vs predicted comparisons

---

## Key Algorithm Features

### 1. Handling Missing Past Data

```fortran
IF(INCLUDPAST .EQ. 0 .OR. IPRIOROBS .EQ. 0) THEN
    ! No past data or no observations
    ! Use prior density directly
    DENSITY = PRIOR_DENSITY
    NGRD = NGRID_PRIOR
ELSE
    ! Past data exists with observations
    ! Calculate posterior via NPAGFULL11 + NPAGFULL
    ! ... (Stage 1 above) ...
ENDIF
```

**Two scenarios**:

- **No past**: Use population prior directly
- **Past available**: Calculate patient-specific posterior

### 2. Multiple Doses

BestDose can optimize:

- Single dose (scalar optimization)
- Multiple sequential doses (vector optimization)
- Loading + maintenance doses
- Different routes (IV bolus, infusion, oral)

**Example**: 3-dose regimen

```
param = [dose1, dose2, dose3]
Initial simplex: 4 vertices in 3D space
```

### 3. Target Types

#### Concentration Targets (ITARGET = 1)

```fortran
! Targets are concentration values at observation times
obs_vec = [C_target_1, C_target_2, ..., C_target_n]
predictions = [C_pred_1, C_pred_2, ..., C_pred_n]
```

#### AUC Targets (ITARGET = 2)

```fortran
! Targets are cumulative AUC values from time 0
! AUC reset at TNEXT (start of "future")
obs_vec = [AUC_target_1, AUC_target_2, ..., AUC_target_n]
predictions = [AUC_pred_1, AUC_pred_2, ..., AUC_pred_n]
```

**AUC calculation**:

- Integrated during ODE solving
- Cumulative from start of "future" period
- Evaluated at target times

### 4. Time Handling

```
<----- Past -----><----- Future ----->
                  ^
                TNEXT = 0 for "future"
```

**Past** (optional):

- Historical doses and observations
- Times relative to start of patient history
- Used to calculate posterior

**Future**:

- Doses to optimize
- Target observations
- Times reset to start at 0 (TNEXT)
- All past times + TNEXT when combined

### 5. Steady State Dosing

**Past steady state**:

- Expanded to 100 explicit doses before optimization
- Ensures model reaches steady state
- Negative time indicator converted to regular doses

**Future**: No steady state allowed (must be explicit doses)

---

## Algorithm Complexity

### Computational Cost

**Stage 1: Posterior (one-time)**

- NPAGFULL11: O(N × S) where N = prior points, S = subject complexity
  - Fast: Single evaluation pass
- NPAGFULL: O(M × C × N_expanded) where:
  - M = filtered points (5-50)
  - C = cycles to convergence (5-20)
  - N_expanded = expanded grid per cycle
  - Expensive: Full optimization per point

**Stage 2: Optimization (iterative)**

- Per cost evaluation: O(M × S)
  - M = posterior points
  - S = ODE integration cost
- Nelder-Mead iterations: ~100-1000
- Total: O(iterations × M × S)

**Stage 3: Predictions (one-time)**

- O(M × S) for final predictions

**Typical timing** (single-core):

- Stage 1: 10-60 seconds
- Stage 2: 30-300 seconds
- Stage 3: <1 second
- **Total**: 1-5 minutes per patient

### Scalability

**Rust improvements**:

- Parallel ODE solving: M support points computed simultaneously
- Rayon parallelism: Automatic work distribution
- Speedup: ~Nx on N cores (near-linear)

**Bottlenecks**:

- ODE integration (cannot parallelize single solve)
- Nelder-Mead (sequential by nature)
- File I/O (Fortran only)

---

## Practical Considerations

### 1. Dose Range Selection

**Too narrow**:

- May not include optimal dose
- Optimization hits bounds frequently
- Results unreliable

**Too wide**:

- Longer optimization time
- May find unsafe/impractical doses
- Need explicit safety constraints

**Best practice**:

```
min_dose = 0.5 × expected_dose
max_dose = 2.0 × expected_dose
```

### 2. Bias Weight Selection

**λ = 0** (default): Full personalization

- Use when: Patient data is reliable and substantial
- Risk: May recommend extreme doses for outlier patients

**λ = 0.5**: Balanced

- Use when: Moderate patient data, want safety
- Compromises between individual and population

**λ = 1**: Population-based only

- Use when: No/poor patient data, or very high risk
- Ignores patient-specific information entirely

### 3. Target Selection

**Concentration targets**:

- Direct interpretation (Cmax, Ctrough, etc.)
- Easier to specify clinically
- May not capture full exposure

**AUC targets**:

- Better for efficacy/toxicity relationships
- Total drug exposure
- Less intuitive to specify

### 4. Multiple Target Times

**Advantages**:

- Better constraint on concentration profile
- Can target both Cmax and Ctrough
- Reduces risk of undershoot/overshoot

**Example**:

```
Time    Target Type    Value
12h     Cmax          20 mg/L
24h     Ctrough       5 mg/L
```

### 5. Prior Quality

**Good prior** (from large population):

- Broad coverage of parameter space
- Appropriate support point density
- Stable optimization

**Poor prior** (small/biased population):

- May miss patient's true parameters
- Extrapolation required
- Unreliable dose recommendations

---

## Limitations and Extensions

### Current Limitations

1. **Single subject**: Optimizes one patient at a time
2. **Fixed schedule**: Dose times specified, not optimized
3. **No adaptive dosing**: Doesn't plan future measurements
4. **Deterministic targets**: Fixed target values, not ranges
5. **No constraints**: Beyond min/max dose (e.g., no divisibility)

### Potential Extensions

#### 1. Dose Timing Optimization

```
Optimize: [dose1, time1, dose2, time2, ...]
Current:  [dose1, dose2, ...] with fixed times
```

#### 2. Target Ranges

```
Cost = penalty if prediction outside [target_min, target_max]
Current: Cost = (prediction - target)²
```

#### 3. Multiple Objectives

```
Cost = w1 × efficacy_loss + w2 × toxicity_risk + w3 × dose_cost
Current: Cost = distance from single target
```

#### 4. Adaptive Dosing Protocol

```
Optimize: [dose1, measure_time1, dose2, measure_time2, ...]
Goal: Minimize total doses while ensuring target achievement
```

#### 5. Robustness Optimization

```
Cost = max over scenarios (worst-case)
Current: Cost = expected value (average-case)
```

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    BestDose Algorithm                        │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
            ┌───▼────┐                  ┌────▼────┐
            │ Prior  │                  │  Past   │
            │Density │                  │  Data   │
            │(NPAG)  │                  │(optional)│
            └───┬────┘                  └────┬────┘
                │                            │
                └──────────┬─────────────────┘
                           │
                  ┌────────▼─────────┐
                  │    Stage 1:      │
                  │   Posterior      │
                  │   Calculation    │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────▼──────┐          ┌──────▼─────┐
        │NPAGFULL11  │          │ NPAGFULL   │
        │(Bayesian   │──────────│ (Refine    │
        │ Filter)    │          │  each pt)  │
        └─────┬──────┘          └──────┬─────┘
              │                        │
              └───────────┬────────────┘
                          │
                  ┌───────▼────────┐
                  │   Posterior    │
                  │   Density      │
                  │(M refined pts) │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │    Stage 2:    │
                  │     Dose       │
                  │  Optimization  │
                  └───────┬────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        ┌─────▼──────┐        ┌──────▼─────┐
        │   Nelder   │        │    Cost    │
        │   Mead     │◄───────│  Function  │
        │  Simplex   │        │  Minimize  │
        └─────┬──────┘        └────────────┘
              │                      │
              │              ┌───────┴──────┐
              │              │              │
              │         ┌────▼───┐    ┌─────▼────┐
              │         │Variance│    │  Bias²   │
              │         │ (1-λ)  │    │   (λ)    │
              │         └────┬───┘    └─────┬────┘
              │              └────────┬──────┘
              │                       │
              └───────────┬───────────┘
                          │
                  ┌───────▼────────┐
                  │   Optimal      │
                  │   Dose(s)      │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │    Stage 3:    │
                  │   Generate     │
                  │  Predictions   │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │    Output:     │
                  │  Optimal Doses │
                  │  + Predictions │
                  │  + Metrics     │
                  └────────────────┘
```

---

## Comparison: Fortran vs Rust Implementation

### Fortran BestDose (BESTDOS121.FOR)

- **Complete**: Includes file I/O, user interface, report generation
- **Monolithic**: Single large program with many subroutines
- **Mature**: Decades of refinement and validation
- **Limitations**: Sequential execution, manual memory management

### Rust BestDose (mod.rs)

- **Core algorithm only**: Focuses on mathematical optimization
- **Modular**: Separate concerns (I/O, optimization, simulation)
- **Modern**: Type safety, parallel execution, error handling
- **Extensible**: Easy to add features (new cost functions, constraints)

### Shared Core

Both implement the same mathematical algorithm:

1. Posterior calculation (hybrid NPAGFULL11 + NPAGFULL approach)
2. Cost function (variance + bias² with λ weighting)
3. Nelder-Mead optimization
4. Prediction generation

The Rust version could fully replace Fortran given complete I/O scaffolding.
