# Analysis: Mixed Zero/Non-Zero Doses in Target

**Question**: What happens if some doses in the target are zero and others are not?  
**Expected Behavior**: Only zero doses should be optimized, non-zero doses should be fixed.

---

## Test Case: Mixed Doses with Concatenation

### Scenario Setup

```rust
// Past subject (6 hours of history)
let past = Subject::builder("patient")
    .bolus(0.0, 500.0, 0)       // Initial dose: 500 mg at t=0
    .observation(6.0, 15.0, 0)  // Observation: 15 mg/L at t=6
    .build();

// Future subject (relative times)
let future = Subject::builder("patient")
    .bolus(0.0, 0.0, 0)         // Dose 1 at t=0 (relative): ZERO → optimize
    .bolus(6.0, 100.0, 0)       // Dose 2 at t=6 (relative): 100 mg → fixed
    .bolus(12.0, 0.0, 0)        // Dose 3 at t=12 (relative): ZERO → optimize
    .observation(24.0, 10.0, 0) // Target at t=24 (relative)
    .build();

// Concatenate with current_time = 6.0
let combined = concatenate_past_and_future(&past, &future, 6.0);
```

### After Concatenation

```rust
// Combined subject (absolute times)
combined:
    .bolus(0.0, 500.0, 0)       // [0] Past: 500 mg at t=0 (fixed)
    .bolus(6.0, 0.0, 0)         // [1] Future: 0 mg at t=6 (optimize)
    .bolus(12.0, 100.0, 0)      // [2] Future: 100 mg at t=12 (fixed)
    .bolus(18.0, 0.0, 0)        // [3] Future: 0 mg at t=18 (optimize)
    .observation(30.0, 10.0, 0) // Target at t=30
```

---

## Code Execution Trace

### Step 1: Optimization Extracts Doses

From `run_single_optimization` (optimization.rs lines 111-121):

```rust
let all_doses: Vec<f64> = target_subject
    .iter()
    .flat_map(|occ| {
        occ.iter().filter_map(|event| match event {
            Event::Bolus(bolus) => Some(bolus.amount()),
            // ...
        })
    })
    .collect();

// Result:
all_doses = [500.0, 0.0, 100.0, 0.0]
```

### Step 2: Count Optimizable Doses

From `run_single_optimization` (optimization.rs line 124):

```rust
let num_optimizable = all_doses.iter().filter(|&&d| d == 0.0).count();
let num_fixed = all_doses.len() - num_optimizable;

// Result:
num_optimizable = 2  // Doses at indices 1 and 3
num_fixed = 2        // Doses at indices 0 and 2
```

✅ **CORRECT**: Identifies exactly the zero doses as optimizable!

### Step 3: Create Simplex for Optimizable Doses Only

From `run_single_optimization` (optimization.rs lines 136-139):

```rust
let initial_guess = (min_dose + max_dose) / 2.0;
let initial_point = vec![initial_guess; num_optimizable];

// If dose range is [0, 300]:
initial_point = [150.0, 150.0]  // 2 values for 2 optimizable doses
```

✅ **CORRECT**: Creates simplex with correct dimensionality!

### Step 4: Optimization Runs

Nelder-Mead optimizes over the 2-dimensional space of optimizable doses:

- `candidate_doses[0]` corresponds to dose at index 1 (t=6)
- `candidate_doses[1]` corresponds to dose at index 3 (t=18)

### Step 5: Cost Function Updates Subject

From `calculate_cost` (cost.rs lines 130-150):

```rust
let mut target_subject = problem.target.clone();
let mut optimizable_dose_number = 0;

for occasion in target_subject.iter_mut() {
    for event in occasion.iter_mut() {
        match event {
            Event::Bolus(bolus) => {
                if bolus.amount() == 0.0 {
                    bolus.set_amount(candidate_doses[optimizable_dose_number]);
                    optimizable_dose_number += 1;
                }
                // If amount > 0, keep original
            }
            // ...
        }
    }
}
```

**Iteration trace:**

```
Dose [0]: amount = 500.0 → NOT zero → keep 500.0
Dose [1]: amount = 0.0 → IS zero → set to candidate_doses[0]
Dose [2]: amount = 100.0 → NOT zero → keep 100.0
Dose [3]: amount = 0.0 → IS zero → set to candidate_doses[1]
```

✅ **CORRECT**: Only updates zero doses, keeps non-zero doses fixed!

### Step 6: Map Optimized Doses Back

From `run_single_optimization` (optimization.rs lines 171-183):

```rust
let mut full_doses = Vec::with_capacity(all_doses.len());
let mut opt_idx = 0;

for &original_dose in all_doses.iter() {
    if original_dose == 0.0 {
        full_doses.push(optimized_doses[opt_idx]);
        opt_idx += 1;
    } else {
        full_doses.push(original_dose);
    }
}
```

**Iteration trace:**

```
original_dose = 500.0 → NOT zero → push 500.0 (fixed)
original_dose = 0.0 → IS zero → push optimized_doses[0] (optimized)
original_dose = 100.0 → NOT zero → push 100.0 (fixed)
original_dose = 0.0 → IS zero → push optimized_doses[1] (optimized)
```

**Result:**

```rust
full_doses = [500.0, optimized_value_1, 100.0, optimized_value_2]
```

✅ **CORRECT**: Correctly interleaves fixed and optimized doses!

---

## Verification: Complete Example

### Input

```rust
// Past: 1 dose
past.bolus(0.0, 500.0, 0)

// Future: 3 doses (mix of zero and non-zero)
future.bolus(0.0, 0.0, 0)      // Will be optimized
future.bolus(6.0, 100.0, 0)    // Fixed at 100 mg
future.bolus(12.0, 0.0, 0)     // Will be optimized
```

### After Concatenation (current_time = 6.0)

```rust
combined.bolus(0.0, 500.0, 0)   // [0] Fixed (past)
combined.bolus(6.0, 0.0, 0)     // [1] Optimize
combined.bolus(12.0, 100.0, 0)  // [2] Fixed (future, non-zero)
combined.bolus(18.0, 0.0, 0)    // [3] Optimize
```

### During Optimization

```
all_doses = [500.0, 0.0, 100.0, 0.0]
num_optimizable = 2
num_fixed = 2

Optimizer searches 2D space for doses [1] and [3]
Dose [0] stays 500.0 ✓
Dose [2] stays 100.0 ✓
```

### Final Result

```rust
optimal_doses = [500.0, <optimized>, 100.0, <optimized>]
```

✅ **WORKS PERFECTLY!**

---

## Answer to Your Question

**YES, your expectation is exactly correct!** ✅

The implementation handles mixed zero/non-zero doses correctly:

### What Happens:

1. **Zero doses (amount = 0.0)**: Are treated as placeholders and **optimized**
2. **Non-zero doses (amount > 0)**: Are treated as fixed and **kept at their specified value**

### This Works For:

- ✅ Past doses (always non-zero after concatenation) → **Fixed**
- ✅ Future doses with zero amounts → **Optimized**
- ✅ Future doses with non-zero amounts → **Fixed at specified value**

### Real-World Use Case

This is actually very useful! You can specify:

```rust
let future = Subject::builder("patient")
    .bolus(0.0, 0.0, 0)         // Optimize first dose
    .bolus(6.0, 200.0, 0)       // Force second dose to 200 mg (e.g., max safe dose)
    .bolus(12.0, 0.0, 0)        // Optimize third dose
    .observation(24.0, 10.0, 0) // Target concentration
    .build();
```

The optimizer will find the best values for the first and third doses while keeping the second dose at exactly 200 mg.

---

## Implementation Quality

### Strengths ✅

1. **Flexible masking**: Supports any mix of fixed and optimizable doses
2. **Correct indexing**: Properly maps between optimizable-only and full dose vectors
3. **Efficient**: Only optimizes the free parameters (reduces dimensionality)
4. **Intuitive**: Zero = optimize, non-zero = fixed

### Edge Cases Handled ✅

1. **All doses zero**: Optimizes all ✓
2. **All doses non-zero**: Optimizes none (returns immediately) ✓
3. **Mixed doses**: Optimizes only zeros ✓
4. **Past + Future with concatenation**: Correctly distinguishes based on amount ✓

---

## Conclusion

**Your expectation is 100% correct, and the implementation delivers exactly that behavior!** ✅

The zero/non-zero masking approach is actually quite elegant and flexible:

- Simple rule: `amount == 0.0` means optimize
- Works correctly in all modes (standard and concatenation)
- Allows fine-grained control over which doses to optimize
- Properly handles the interaction between fixed and optimizable doses

**No bugs found in this aspect of the implementation!** 🎉

The only caveat (from my previous analysis) is that when using concatenation mode, you **must** ensure future doses you want to optimize have `amount = 0.0` in the template. But as long as you follow that convention, everything works perfectly!
