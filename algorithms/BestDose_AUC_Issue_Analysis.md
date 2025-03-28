# BestDose AUC Target Issue Analysis

## Problem Statement

When running the `bestdose_auc` example with AUC targets of 50 and 80 mg·h/L at 6h and 12h respectively, the optimizer selects a dose of 1145 mg that produces weighted mean predictions of:

- AUC@6h: 44.01 (target: 50.0) - **12% under**
- AUC@12h: 66.14 (target: 80.0) - **17% under**

This seems counterintuitive: why doesn't the optimizer choose a higher dose when both predictions are below the target?

## Root Cause Analysis

### The Bimodal Distribution

The prior distribution has a **bimodal clearance distribution** with two distinct clusters:

1. **Low clearance cluster** (ke ~ 0.02-0.13): ~50% of population

   - These patients need moderate doses (~1000-1200 mg)
   - The posterior weights **favor** this cluster

2. **High clearance cluster** (ke ~ 0.28-0.35): ~40% of population
   - These patients need much higher doses (>2000 mg)
   - Lower posterior weights

### What the Cost Function Actually Minimizes

The cost function is:

```
Cost = Variance = Σᵢ posterior_weight[i] × Σⱼ (target[j] - pred[i,j])²
```

This is the **Bayesian expected squared error** across all support points, NOT the squared error of the weighted mean prediction!

### Why 1145 mg is "Optimal"

At 1145 mg dose:

- **Support point 0** (ke=0.0874, posterior_weight=0.064023):
  - AUC@6h: 51.38, AUC@12h: 81.78
  - Squared errors: 1.90 + 3.18 = **5.08** ← Very small!
- **Support point 1** (ke=0.1311, posterior_weight=0.039216):

  - AUC@6h: 49.09, AUC@12h: 71.43
  - Squared errors: 0.84 + 73.38 = **74.22** ← Small

- **Total Variance**: 1408.07

At 1500 mg dose:

- **Support point 0** (ke=0.0874, posterior_weight=0.064023):

  - AUC@6h: 67.31, AUC@12h: 107.14
  - Squared errors: 299.62 + 736.59 = **1036.21** ← Much larger!

- **Support point 1** (ke=0.1311, posterior_weight=0.039216):

  - AUC@6h: 64.30, AUC@12h: 93.58
  - Squared errors: 204.59 + 184.45 = **389.04** ← Larger

- **Total Variance**: 2127.99

The high-posterior-weight support points (low clearance) have **much smaller errors at 1145 mg** than at 1500 mg, even though the weighted mean at 1500 mg is closer to the target!

### The Conceptual Gap

There are two different objectives:

1. **Current implementation**: Minimize E[(target - prediction)²]

   - This is the **expected squared error** given uncertainty
   - Favors doses that work well for high-probability patient types
   - Result: **1145 mg** is optimal

2. **User expectation**: Minimize (target - E[prediction])²
   - This is the **squared error of the mean**
   - Tries to make the average prediction match the target
   - Result: Would choose **~1300 mg** where mean predictions ≈ targets

## Is This a Bug or Feature?

**This is technically correct Bayesian decision theory**, but it may not align with clinical intuition:

### Arguments for current behavior (minimize expected error):

- ✅ Theoretically sound from a Bayesian perspective
- ✅ Protects high-probability patient types from overdosing
- ✅ Minimizes expected harm across the posterior distribution

### Arguments for alternative (match mean prediction to target):

- ✅ More intuitive interpretation
- ✅ Aligns with how clinicians think about "population average"
- ✅ When displaying weighted mean predictions, they should match targets
- ✅ The bias term uses weighted means anyway

## Comparison with Concentration Targets

This issue is **more pronounced with AUC targets** than concentration targets because:

1. **AUC accumulates over time**: Small differences in clearance create large AUC differences
2. **Bimodal distributions**: Two clearance clusters need very different doses
3. **No prior data**: When using pure prior (no patient history), both clusters have similar weights

With concentration targets and patient data, NPAGFULL11 usually narrows the posterior to one cluster, making this less of an issue.

## Proposed Solutions

### Option 1: Add a flag to control optimization objective

Add a parameter to `BestDoseProblem` to choose between:

- `minimize_expected_error` (current behavior)
- `match_mean_prediction` (alternative)

```rust
pub enum OptimizationObjective {
    MinimizeExpectedError,  // Current: Σᵢ wᵢ × (target - predᵢ)²
    MatchMeanPrediction,    // Alternative: (target - Σᵢ wᵢ × predᵢ)²
}
```

### Option 2: Weight by prior²×posterior instead of just posterior

This would give more weight to support points that are both:

1. Common in the population (high prior)
2. Compatible with patient data (high posterior)

```rust
let effective_weight = prior_weight * posterior_weight.sqrt();
```

### Option 3: Use the bias term properly

The bias term already calculates `(target - population_mean)²`. We could:

- Set a higher default `bias_weight` (e.g., 0.5 instead of 0.0)
- Or automatically increase `bias_weight` when posterior is diffuse

### Option 4: Hybrid approach for AUC targets

For AUC targets specifically, use a different cost function:

```rust
match problem.target_type {
    Target::Concentration => {
        // Use expected error (current behavior)
        cost = variance
    }
    Target::AUC => {
        // Use mean-matching with penalty for variance
        let mean_error = (target - weighted_mean)²;
        let spread_penalty = variance;
        cost = mean_error + 0.1 * spread_penalty;
    }
}
```

## Recommended Action

I recommend **Option 1** (add a flag) because:

1. Preserves current behavior for users who want Bayesian expected error
2. Allows users to choose the more intuitive mean-matching objective
3. Documents the difference clearly
4. Could make `MatchMeanPrediction` the default for AUC targets

Implementation would involve:

1. Add `OptimizationObjective` enum to types
2. Add field to `BestDoseProblem`
3. Modify `calculate_cost` to compute cost differently based on objective
4. Update documentation and examples

## Immediate Workaround

For the current code, users can:

1. **Increase `bias_weight`**: Set to 0.5-0.8 to favor matching population mean
2. **Provide patient data**: Even minimal data will narrow the posterior and reduce this issue
3. **Accept the behavior**: Understand that 1145 mg minimizes expected error even if mean undershoots

## Test Case

The `bestdose_auc` example should be updated to:

1. Show both the weighted mean AUC AND individual support point contributions
2. Explain why the optimizer chose that dose
3. Optionally demonstrate both optimization objectives side-by-side
