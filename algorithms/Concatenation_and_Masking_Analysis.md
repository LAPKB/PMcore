# Deep Analysis: Concatenation and Dose Masking Correctness

**Date**: October 17, 2025  
**Issue**: Verify concatenation logic and dose masking are correct

## Executive Summary

**STATUS: ⚠️ CRITICAL BUG FOUND**

The implementation has a **major logical flaw** in how doses are masked for optimization. The current implementation uses **dose amount == 0** as the criterion for "optimizable", but this conflicts with how concatenation works.

---

## The Problem: Logical Conflict

### Current Implementation Logic

1. **Concatenation** (`concatenate_past_and_future`):

   - Takes past doses with their **actual amounts** (e.g., 500 mg)
   - Takes future doses with their **initial amounts** (e.g., 100 mg or 0 mg)
   - Combines them into one subject

2. **Masking** (`calculate_dose_optimization_mask`):

   - Returns `true` if dose amount == 0.0 (treat as optimizable)
   - Returns `false` if dose amount > 0.0 (treat as fixed)

3. **Cost Function** (`calculate_cost`):
   - Only updates doses where amount == 0.0
   - Keeps doses with amount > 0.0 unchanged

### The Bug

**After concatenation, ALL doses have non-zero amounts!**

Example:

```rust
// Past subject
past.bolus(0.0, 500.0, 0)      // 500 mg at t=0

// Future subject
future.bolus(0.0, 100.0, 0)    // 100 mg at t=0 (relative)

// After concatenation with current_time=6.0
combined.bolus(0.0, 500.0, 0)  // Past: 500 mg at t=0
combined.bolus(6.0, 100.0, 0)  // Future: 100 mg at t=6

// Masking result
mask[0] = false  // 500.0 != 0, so NOT optimizable ✓ CORRECT
mask[1] = false  // 100.0 != 0, so NOT optimizable ✗ WRONG!
```

**The future dose (100 mg) should be optimizable, but it has a non-zero amount!**

---

## Analysis of Each Component

### 1. Concatenation Function ✅ CORRECT

```rust
fn concatenate_past_and_future(
    past: &pharmsol::prelude::Subject,
    future: &pharmsol::prelude::Subject,
    current_time: f64,
) -> pharmsol::prelude::Subject {
    // Add past doses at original times
    for occasion in past.occasions() {
        for event in occasion.events() {
            match event {
                Event::Bolus(bolus) => {
                    builder = builder.bolus(
                        bolus.time(),           // Original time
                        bolus.amount(),         // Original amount (e.g., 500 mg)
                        bolus.input()
                    );
                }
                // ...
            }
        }
    }

    // Add future doses at offset times
    for occasion in future.occasions() {
        for event in occasion.events() {
            match event {
                Event::Bolus(bolus) => {
                    builder = builder.bolus(
                        bolus.time() + current_time,  // Offset time
                        bolus.amount(),               // Keep amount (e.g., 100 mg)
                        bolus.input()
                    );
                }
                // ...
            }
        }
    }

    builder.build()
}
```

**Analysis:**

- ✓ Correctly takes doses from past
- ✓ Correctly skips observations from past
- ✓ Correctly offsets future times by current_time
- ✓ Correctly preserves future doses and observations

**Problem:** The function preserves dose amounts (as it should), but this breaks the masking logic downstream!

### 2. Masking Function ✗ INCORRECT FOR CONCATENATION MODE

```rust
fn calculate_dose_optimization_mask(subject: &pharmsol::prelude::Subject) -> Vec<bool> {
    let mut mask = Vec::new();

    for occasion in subject.occasions() {
        for event in occasion.events() {
            match event {
                Event::Bolus(bolus) => {
                    // Dose is optimizable if amount is 0 (placeholder)
                    mask.push(bolus.amount() == 0.0);  // ✗ WRONG!
                }
                // ...
            }
        }
    }

    mask
}
```

**Analysis:**

- ✓ Works correctly in standard mode (no concatenation)
- ✗ **FAILS in concatenation mode** because future doses have non-zero amounts

**Why this is used:** This mask is only calculated for logging (lines 605-609 in mod.rs), not for actual optimization. But it shows a conceptual misunderstanding.

### 3. Optimization ✅ CORRECT (by accident?)

```rust
fn run_single_optimization(...) -> Result<(Vec<f64>, f64)> {
    // Get all doses from target subject
    let all_doses: Vec<f64> = target_subject
        .iter()
        .flat_map(|occ| {
            occ.iter().filter_map(|event| match event {
                Event::Bolus(bolus) => Some(bolus.amount()),
                // ...
            })
        })
        .collect();

    // Count optimizable doses (amount == 0)
    let num_optimizable = all_doses.iter().filter(|&&d| d == 0.0).count();  // ✗ WRONG CRITERION

    // ...

    // Map optimized doses back to full vector
    for &original_dose in all_doses.iter() {
        if original_dose == 0.0 {  // ✗ WRONG CRITERION
            full_doses.push(optimized_doses[opt_idx]);
            opt_idx += 1;
        } else {
            full_doses.push(original_dose);
        }
    }
}
```

**Analysis:**

- Uses same wrong criterion: `amount == 0.0` means optimizable
- But this function gets its subject from `problem.target`
- What is in `problem.target` after concatenation?

### 4. Cost Function ✅ CORRECT (same logic as optimization)

```rust
pub fn calculate_cost(problem: &BestDoseProblem, candidate_doses: &[f64]) -> Result<f64> {
    let mut target_subject = problem.target.clone();
    let mut optimizable_dose_number = 0;

    for occasion in target_subject.iter_mut() {
        for event in occasion.iter_mut() {
            match event {
                Event::Bolus(bolus) => {
                    if bolus.amount() == 0.0 {  // ✗ WRONG CRITERION
                        bolus.set_amount(candidate_doses[optimizable_dose_number]);
                        optimizable_dose_number += 1;
                    }
                }
                // ...
            }
        }
    }
}
```

**Analysis:**

- Uses same logic as optimization
- Consistency is good, but is it correct?

---

## The Key Question: What Does `problem.target` Contain?

Let me trace through the initialization in `BestDoseProblem::new`:

```rust
// Line 600-612 in mod.rs
let (final_target, final_past_data_for_storage) = if let Some(t) = current_time {
    tracing::info!("  Concatenating past and future (MAKETMP approach)...");
    let combined = concatenate_past_and_future(&final_past_data, &target, t);

    // Calculate mask for logging
    let mask = calculate_dose_optimization_mask(&combined);
    let num_fixed = mask.iter().filter(|&&x| !x).count();
    let num_optimizable = mask.iter().filter(|&&x| x).count();
    tracing::info!("  Past doses (fixed): {}", num_fixed);
    tracing::info!("  Future doses (optimizable): {}", num_optimizable);

    (combined, final_past_data)  // ← combined goes into final_target
} else {
    (target, final_past_data)
};

// Line 621-633 in mod.rs
Ok(BestDoseProblem {
    past_data: final_past_data_for_storage,
    target: final_target,  // ← This is the combined subject!
    // ...
})
```

**CRITICAL INSIGHT:**

- In concatenation mode, `problem.target` contains the **COMBINED** subject
- The combined subject has past doses (500 mg) + future doses (100 mg)
- **ALL doses have non-zero amounts!**

---

## The Bug Manifestation

### Scenario 1: Future doses start at 0 mg (placeholders)

```rust
// Past
past.bolus(0.0, 500.0, 0)       // 500 mg at t=0

// Future (0 mg as placeholder)
future.bolus(0.0, 0.0, 0)       // 0 mg at t=0 (to be optimized)

// After concatenation
combined.bolus(0.0, 500.0, 0)   // 500 mg at t=0
combined.bolus(6.0, 0.0, 0)     // 0 mg at t=6

// Masking
mask[0] = false  // 500.0 != 0 → NOT optimizable ✓ CORRECT
mask[1] = true   // 0.0 == 0 → optimizable ✓ CORRECT

// Optimization extracts doses
all_doses = [500.0, 0.0]
num_optimizable = 1  ✓ CORRECT

// Cost function
dose[0] = 500.0  → keep as-is ✓ CORRECT
dose[1] = 0.0    → update with candidate ✓ CORRECT
```

**This works correctly!** ✅

### Scenario 2: Future doses start at non-zero (initial guess)

```rust
// Past
past.bolus(0.0, 500.0, 0)       // 500 mg at t=0

// Future (100 mg as initial guess)
future.bolus(0.0, 100.0, 0)     // 100 mg at t=0 (to be optimized)

// After concatenation
combined.bolus(0.0, 500.0, 0)   // 500 mg at t=0
combined.bolus(6.0, 100.0, 0)   // 100 mg at t=6

// Masking
mask[0] = false  // 500.0 != 0 → NOT optimizable ✓ CORRECT
mask[1] = false  // 100.0 != 0 → NOT optimizable ✗ WRONG!

// Optimization extracts doses
all_doses = [500.0, 100.0]
num_optimizable = 0  ✗ WRONG!

// Result: NO DOSES ARE OPTIMIZED!
```

**This fails completely!** ❌

---

## Documentation vs Implementation

### Documentation says (lines 409-420 in mod.rs):

````rust
/// This allows users to specify a combined subject with:
/// - Non-zero doses for past doses (e.g., 500 mg at t=0) - these are fixed
/// - Zero doses as placeholders for future doses (e.g., 0 mg at t=6) - these are optimized
///
/// # Example
///
/// ```rust,ignore
/// let subject = Subject::builder("patient")
///     .bolus(0.0, 500.0, 0)    // Past dose (fixed) - mask[0] = false
///     .bolus(6.0, 0.0, 0)      // Future dose (optimize) - mask[1] = true
///     .observation(30.0, 10.0, 0)
///     .build();
/// ```
````

**This clearly states that future doses MUST have amount = 0 for optimization to work!**

But the concatenation function documentation (lines 328-337) shows:

```rust
/// // Future: dose at t=0 (relative), target at t=24 (relative)
/// let future = Subject::builder("patient")
///     .bolus(0.0, 100.0, 0)  // Dose to optimize, will be at t=6 absolute
///     .observation(24.0, 10.0, 0)  // Target at t=30 absolute
///     .build();
```

**This shows a non-zero dose (100 mg) that should be optimized!**

**CONTRADICTION!** 📛

---

## Root Cause Analysis

The fundamental issue is **mixing two different conventions**:

### Convention 1: Zero-based masking (current implementation)

- Dose amount == 0 → optimizable
- Dose amount != 0 → fixed
- **Problem:** Requires future doses to be zero, but this loses information about initial guesses

### Convention 2: Time-based masking (what Fortran does)

- Doses before current_time → fixed (from past)
- Doses at/after current_time → optimizable (from future)
- **Advantage:** Allows non-zero initial guesses for future doses

### What the Code Actually Needs

In concatenation mode, we need to distinguish:

1. **Past doses** (time < current_time): Always fixed, keep their amounts
2. **Future doses** (time >= current_time): Always optimizable, regardless of amount

The current implementation assumes we can tell them apart by amount (0 vs non-zero), but this is fragile and incorrect!

---

## The Correct Solution

### Option 1: Force Zero Amounts for Future Doses ⚠️ WORKAROUND

**Change concatenation to zero out future doses:**

```rust
fn concatenate_past_and_future(
    past: &pharmsol::prelude::Subject,
    future: &pharmsol::prelude::Subject,
    current_time: f64,
) -> pharmsol::prelude::Subject {
    // ... (past doses unchanged) ...

    // Add future events with time offset AND ZERO AMOUNTS
    for occasion in future.occasions() {
        for event in occasion.events() {
            match event {
                Event::Bolus(bolus) => {
                    builder = builder.bolus(
                        bolus.time() + current_time,
                        0.0,  // ← Force zero amount for masking
                        bolus.input()
                    );
                }
                // ...
            }
        }
    }

    builder.build()
}
```

**Pros:**

- Minimal change to existing code
- Works with current masking logic

**Cons:**

- Loses information about initial dose guesses from future template
- Inconsistent with documentation example showing 100 mg future dose

### Option 2: Time-Based Masking ✅ CORRECT SOLUTION

**Store the boundary time and use it for masking:**

```rust
// In BestDoseProblem
pub struct BestDoseProblem {
    // ... existing fields ...
    pub current_time: Option<f64>,  // Already exists!
}

// In optimization
fn run_single_optimization(...) -> Result<(Vec<f64>, f64)> {
    let all_doses_with_times: Vec<(f64, f64)> = target_subject
        .iter()
        .flat_map(|occ| {
            occ.iter().filter_map(|event| match event {
                Event::Bolus(bolus) => Some((bolus.time(), bolus.amount())),
                // ...
            })
        })
        .collect();

    // Determine which doses are optimizable based on time
    let num_optimizable = if let Some(boundary) = problem.current_time {
        // Concatenation mode: optimize doses at/after boundary
        all_doses_with_times.iter()
            .filter(|(time, _)| *time >= boundary)
            .count()
    } else {
        // Standard mode: optimize doses with amount == 0
        all_doses_with_times.iter()
            .filter(|(_, amount)| *amount == 0.0)
            .count()
    };

    // ... rest of optimization ...

    // Map optimized doses back
    for (time, original_dose) in all_doses_with_times.iter() {
        let is_optimizable = if let Some(boundary) = problem.current_time {
            *time >= boundary  // Time-based
        } else {
            *original_dose == 0.0  // Amount-based
        };

        if is_optimizable {
            full_doses.push(optimized_doses[opt_idx]);
            opt_idx += 1;
        } else {
            full_doses.push(*original_dose);
        }
    }
}
```

**Pros:**

- ✅ Correct semantics
- ✅ Allows non-zero initial guesses for future doses
- ✅ Matches Fortran's intent
- ✅ Flexible for standard mode (no concatenation)

**Cons:**

- More code changes needed
- Need to pass times through the optimization

---

## Recommendation

### Immediate Action Required 🚨

**Implement Option 2 (Time-Based Masking)** because:

1. It's the correct semantic model
2. It matches what Fortran does (ND41+1 to ND are optimizable)
3. It allows flexibility in specifying initial dose guesses
4. The infrastructure is already there (`current_time` is stored)

### Files to Change

1. **`src/bestdose/optimization.rs`**:
   - `run_single_optimization`: Use time-based masking when `current_time` is set
2. **`src/bestdose/cost.rs`**:

   - `calculate_cost`: Use time-based masking when `current_time` is set

3. **`src/bestdose/mod.rs`**:
   - `calculate_dose_optimization_mask`: Add `current_time` parameter
   - Update documentation to clarify the two modes

### Testing

Add tests for:

1. ✅ Standard mode with zero-amount placeholders
2. ✅ Concatenation mode with zero-amount future doses
3. ✅ **Concatenation mode with non-zero future doses** (currently broken!)
4. ✅ Verify past doses are never modified
5. ✅ Verify future doses are always optimized

---

## Verification: Current Usage

Looking at `examples/bestdose.rs` (lines 70-75):

```rust
let target_data = Subject::builder("Thomas Edison")
    .bolus(0.0, 0.0, 0)      // ← Zero amount!
    .observation(2.0, conc(2.0, 150.0), 0)
    .bolus(12.0, 0.0, 0)     // ← Zero amount!
    .observation(14.0, conc(2.0, 75.0) + conc(14.0, 150.0), 0)
    .build();

let problem = BestDoseProblem::new(
    // ...
    Some(past_data.clone()),
    target_data.clone(),
    Some(20.0),  // ← current_time set!
    // ...
)?;
```

**KEY INSIGHT:** The example code uses **zero amounts** for all future doses! This explains why the code works in practice despite the conceptual issue.

---

## Conclusion

### Is the concatenation correct?

**YES** ✅ - The `concatenate_past_and_future` function correctly combines past and future subjects per Fortran's MAKETMP.

### Is the masking correct?

**TECHNICALLY YES, BUT FRAGILE** ⚠️

The implementation is correct **IF** users follow the undocumented requirement:

- Future doses MUST have amount = 0.0 in the template
- This is enforced by convention, not by code
- The example follows this convention
- But the documentation is contradictory (shows 100.0 in one place, 0.0 in another)

### Are we optimizing the right doses?

**YES, IF USED CORRECTLY** ✅

When future doses have amount = 0.0:

- ✅ Past doses (non-zero) are correctly identified as fixed
- ✅ Future doses (zero) are correctly identified as optimizable
- ✅ Optimization works as intended

**If future doses had non-zero amounts, everything would break!** ❌

### Is the simulated data correct?

**YES** ✅ - As long as the masking is correct (which it is, given the zero-amount convention), the simulated data is correct.

### Critical Finding

**The implementation is correct BUT relies on an implicit requirement:**

**Future doses in the template MUST be set to 0.0 for optimization to work!**

This requirement is:

- ✅ Followed by the example code
- ✅ Mentioned in some documentation (line 409-420)
- ❌ Contradicted by other documentation (line 328-337 shows 100.0)
- ❌ Not validated or enforced by the code

### Risk Assessment

**MEDIUM RISK** ⚠️

- Current usage is correct ✅
- But the design is fragile and error-prone
- A user who copies the concatenation example (line 328-337) verbatim would get silently wrong results
- No error or warning would be raised

### Recommended Actions

1. **DOCUMENTATION** (High Priority):

   - Fix the contradiction in documentation
   - Clearly state: "Future doses MUST have amount = 0.0"
   - Update the concatenation example to show zero amounts

2. **VALIDATION** (High Priority):

   - Add validation in `new()` when `current_time` is set
   - Check that all future doses (after current_time) have amount = 0.0
   - Raise error if not

3. **REFACTORING** (Medium Priority):

   - Consider implementing time-based masking (Option 2)
   - This would allow non-zero initial guesses
   - More robust and flexible

4. **TESTING** (High Priority):
   - Add test that verifies future doses must be zero
   - Add test that raises error if non-zero future doses are used
