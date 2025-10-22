# BestDose Infusion Support - Visual Architecture

## Current Problem: Manual Index Tracking

```
┌─────────────────────────────────────────────────────────────────┐
│ Current Approach (FRAGILE)                                      │
└─────────────────────────────────────────────────────────────────┘

Subject Events:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Bolus       │ Infusion    │ Bolus       │ Infusion    │
│ amt=0       │ amt=0       │ amt=500     │ amt=0       │
│             │ dur=2h      │ (FIXED)     │ dur=0       │
└─────────────┴─────────────┴─────────────┴─────────────┘

Manual iteration with counters:
let mut dose_idx = 0;        ❌ Easy to get out of sync
let mut param_idx = 0;       ❌ Manual increment logic
for event in events {
    if optimizable {
        param_idx += 1;      ❌ Forgot to check if infusion?
    }
    dose_idx += 1;           ❌ What if we skip observations?
}

Result: 🐛 Index errors, crashes, wrong doses!
```

## Proposed Solution: DoseParameterMap

```
┌─────────────────────────────────────────────────────────────────┐
│ New Approach (ROBUST)                                           │
└─────────────────────────────────────────────────────────────────┘

Step 1: Build the map during initialization
═══════════════════════════════════════════

Subject Events:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Bolus       │ Infusion    │ Bolus       │ Infusion    │
│ amt=0       │ amt=0       │ amt=500     │ amt=0       │
│             │ dur=2h      │ (FIXED)     │ dur=0       │
└─────────────┴─────────────┴─────────────┴─────────────┘
      │             │             │             │
      │             │             │             │
      ▼             ▼             ▼             ▼
DoseParameterMap:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Entry 0     │ Entry 1     │ Entry 2     │ Entry 3     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Type: Bolus │ Type: Inf   │ Type: Bolus │ Type: Inf   │
│ Params: [0] │ Params: [1] │ Params: []  │ Params: [2,3]│
└─────────────┴─────────────┴─────────────┴─────────────┘
      │             │                             │
      │             │                             │
      ▼             ▼                             ▼
Optimization Vector:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Param 0     │ Param 1     │ Param 2     │ Param 3     │
│ Bolus amt   │ Inf amt     │ Inf amt     │ Inf dur     │
└─────────────┴─────────────┴─────────────┴─────────────┘

Step 2: Use during optimization
════════════════════════════════

Optimizer returns: [150.0, 75.0, 50.0, 1.5]
                     │      │     │     │
                     │      │     │     └── Param 3 (Entry 3, duration)
                     │      │     └──────── Param 2 (Entry 3, amount)
                     │      └────────────── Param 1 (Entry 1, amount)
                     └───────────────────── Param 0 (Entry 0, amount)

map.apply_to_subject() automatically:
  ✅ Entry 0 (Bolus): Set amount = 150.0
  ✅ Entry 1 (Infusion): Set amount = 75.0, keep duration = 2h
  ✅ Entry 2 (Bolus): Keep amount = 500 (fixed)
  ✅ Entry 3 (Infusion): Set amount = 50.0, set duration = 1.5

Result: ✨ Correct doses applied automatically!
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    BestDose Architecture                         │
│                  (With Infusion Support)                         │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ BestDoseProblem::new()                                 │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1. Calculate posterior (STAGE 1)                     │
│     ↓                                                  │
│  2. Prepare target subject                            │
│     ↓                                                  │
│  3. Build DoseParameterMap  ⭐ NEW                    │
│     │                                                  │
│     ├─ Scan all events                                │
│     ├─ Identify optimizable parameters                │
│     ├─ Build event → param mapping                    │
│     └─ Validate & log structure                       │
│                                                        │
└────────────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────┐
│ BestDoseProblem (Ready for optimization)               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  • theta (posterior support points)                   │
│  • weights (posterior probabilities)                  │
│  • target (subject with placeholder doses)            │
│  • dose_param_map  ⭐ NEW                             │
│  • doserange (constraints)                            │
│  • ... other fields ...                               │
│                                                        │
└────────────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────┐
│ optimization::dual_optimization() (STAGE 2)            │
├────────────────────────────────────────────────────────┤
│                                                        │
│  For each optimization (posterior & uniform):          │
│    │                                                   │
│    ├─ Create simplex with N parameters                │
│    │  (N = dose_param_map.num_parameters())           │
│    │                                                   │
│    └─ Nelder-Mead optimization                        │
│        ↓                                               │
│       Each iteration calls cost function:             │
│        ↓                                               │
│       cost::calculate_cost()                          │
│        │                                               │
│        ├─ map.apply_to_subject(params)  ⭐ NEW       │
│        ├─ Simulate with updated doses                 │
│        └─ Return cost                                 │
│                                                        │
│  Select best result (min cost)                        │
│                                                        │
└────────────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────┐
│ predictions::calculate_final_predictions() (STAGE 3)   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1. map.apply_to_subject(optimal_params)  ⭐ NEW     │
│     ↓                                                  │
│  2. Calculate psi with optimal doses                  │
│     ↓                                                  │
│  3. Calculate posterior                               │
│     ↓                                                  │
│  4. Generate predictions                              │
│     ↓                                                  │
│  5. Calculate AUC (if needed)                         │
│     • Now with full infusion support! ⭐ NEW         │
│                                                        │
└────────────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────┐
│ BestDoseResult                                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│  • dose: Vec<f64>                                     │
│    (All optimization parameters)                      │
│    [bolus_amt, inf_amt, inf_dur, ...]                │
│                                                        │
│  • preds: NPPredictions                               │
│  • auc_predictions: Option<Vec<(f64,f64)>>           │
│  • objf: f64                                          │
│  • optimization_method: String                        │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## DoseParameterMap Internals

```
┌─────────────────────────────────────────────────────────────────┐
│ DoseParameterMap Structure                                      │
└─────────────────────────────────────────────────────────────────┘

pub struct DoseParameterMap {
    entries: Vec<DoseParameterEntry>,
    total_params: usize,
}

Each DoseParameterEntry:
┌────────────────────────────────────┐
│ event_index: usize                 │  Which event in subject
├────────────────────────────────────┤
│ dose_type: DoseType                │  Bolus or Infusion
├────────────────────────────────────┤
│ param_indices: Vec<usize>          │  Which parameters
│                                    │
│   Examples:                        │
│   • []        → Fixed dose         │
│   • [5]       → One parameter      │
│   • [5, 6]    → Two parameters     │
└────────────────────────────────────┘

Key Methods:
┌─────────────────────────────────────────────────────────────────┐
│ from_subject(subject)                                           │
│   • Scans all events                                            │
│   • Builds mapping structure                                    │
│   • Returns DoseParameterMap                                    │
├─────────────────────────────────────────────────────────────────┤
│ apply_to_subject(subject, params)                               │
│   • Validates param count                                       │
│   • Clones subject                                              │
│   • Applies params to correct events                            │
│   • Returns modified subject                                    │
├─────────────────────────────────────────────────────────────────┤
│ num_parameters()                                                │
│   • Returns total parameter count                               │
├─────────────────────────────────────────────────────────────────┤
│ num_doses()                                                     │
│   • Returns total dose event count                              │
└─────────────────────────────────────────────────────────────────┘
```

### Parameter Mapping Cases

```
┌─────────────────────────────────────────────────────────────────┐
│ All Possible Dose Configurations                               │
└─────────────────────────────────────────────────────────────────┘

Case 1: Fixed Bolus
  bolus(time, 500.0, ...)
  → param_info = []
  → No optimization

Case 2: Optimizable Bolus
  bolus(time, 0.0, ...)
  → param_info = [ParameterInfo { index: N, type: Amount }]
  → Optimize amount

Case 3: Fixed Infusion
  infusion(time, 500.0, ..., 2.0)
  → param_info = []
  → No optimization

Case 4: Infusion - Amount Only
  infusion(time, 0.0, ..., 2.0)
  → param_info = [ParameterInfo { index: N, type: Amount }]
  → Optimize amount, duration fixed at 2.0h

Case 5: Infusion - Duration Only
  infusion(time, 500.0, ..., 0.0)
  → param_info = [ParameterInfo { index: N, type: Duration }]
  → Amount fixed at 500mg, optimize duration

Case 6: Infusion - Both
  infusion(time, 0.0, ..., 0.0)
  → param_info = [
      ParameterInfo { index: N, type: Amount },
      ParameterInfo { index: N+1, type: Duration }
    ]
  → Optimize both amount and duration

Note: ParameterType enum distinguishes Amount vs Duration
      This allows correct identification even when both have 1 parameter
```

## Example Flow: Mixed Dosing

```
┌─────────────────────────────────────────────────────────────────┐
│ User Creates Subject                                            │
└─────────────────────────────────────────────────────────────────┘

let target = Subject::builder("patient")
    .bolus(0.0, 0.0, 0)              // Event 0: Optimize amount
    .infusion(2.0, 0.0, 0, 0.0)      // Event 1: Optimize amt + dur
    .bolus(12.0, 250.0, 0)           // Event 2: Fixed
    .observation(24.0, 5.0, 0)       // Event 3: Not a dose
    .build();

          ↓ BestDoseProblem::new()

┌─────────────────────────────────────────────────────────────────┐
│ DoseParameterMap Construction                                   │
└─────────────────────────────────────────────────────────────────┘

Scanning events:
─────────────────────────────────────────────────────────────────
Event 0: Bolus(amt=0)
  → Optimizable: amount
  → Assign param index 0
  → Entry: { event=0, type=Bolus, params=[0] }

Event 1: Infusion(amt=0, dur=0)
  → Optimizable: amount AND duration
  → Assign param indices 1, 2
  → Entry: { event=1, type=Infusion, params=[1, 2] }

Event 2: Bolus(amt=250)
  → Fixed (non-zero)
  → No param indices
  → Entry: { event=2, type=Bolus, params=[] }

Event 3: Observation
  → Not a dose, skip

Result:
  entries = [Entry0, Entry1, Entry2]
  total_params = 3

          ↓ optimize()

┌─────────────────────────────────────────────────────────────────┐
│ Optimization Process                                            │
└─────────────────────────────────────────────────────────────────┘

Optimizer creates initial simplex in 3D space:
  Dimension 0: Bolus amount (Event 0)
  Dimension 1: Infusion amount (Event 1)
  Dimension 2: Infusion duration (Event 1)

Each cost function call:
  1. Receives params: [p0, p1, p2]
  2. map.apply_to_subject(subject, params):
     - Event 0: Bolus.set_amount(p0)
     - Event 1: Infusion.set_amount(p1), .set_duration(p2)
     - Event 2: Keep amount=250 (fixed)
  3. Simulate modified subject
  4. Calculate cost

After optimization:
  Optimal params = [120.0, 60.0, 1.8]

          ↓ calculate_final_predictions()

┌─────────────────────────────────────────────────────────────────┐
│ Final Result                                                    │
└─────────────────────────────────────────────────────────────────┘

BestDoseResult {
    dose: [120.0, 60.0, 1.8],
    //     │      │     └─ Infusion duration (hours)
    //     │      └─────── Infusion amount (mg)
    //     └────────────── Bolus amount (mg)
    
    preds: <predictions with optimal doses>,
    auc_predictions: <if AUC mode>,
    objf: 0.0234,
    optimization_method: "posterior",
}

User interprets:
  "Give bolus of 120 mg at t=0"
  "Start infusion of 60 mg over 1.8 hours at t=2h"
  "Give bolus of 250 mg at t=12h" (as specified, fixed)
```

## Comparison: Before vs After

```
┌─────────────────────────────────────────────────────────────────┐
│ BEFORE: Manual Index Tracking                                  │
└─────────────────────────────────────────────────────────────────┘

❌ Problems:
  • Manual counter management (error-prone)
  • Separate loops for different purposes
  • Hard to handle mixed dose types
  • Code duplication across modules
  • Difficult to test edge cases
  • Infusion duration not supported

Code characteristics:
  • ~50 lines per module for index tracking
  • Nested conditions and counters
  • "TODO: support infusions" comments
  • Warnings and unsupported features

┌─────────────────────────────────────────────────────────────────┐
│ AFTER: DoseParameterMap                                        │
└─────────────────────────────────────────────────────────────────┘

✅ Advantages:
  • No manual index tracking
  • Single source of truth (the map)
  • Handles all dose combinations automatically
  • Code reuse across modules
  • Easy to test and validate
  • Full infusion support (amt + dur)

Code characteristics:
  • ~5 lines per module for map usage
  • Simple, declarative code
  • No special cases or conditions
  • Comprehensive feature support

┌─────────────────────────────────────────────────────────────────┐
│ Example: Applying Parameters                                   │
└─────────────────────────────────────────────────────────────────┘

BEFORE (cost.rs):
───────────────────────────────────────────────────────────────
let mut target_subject = problem.target.clone();
let mut optimizable_dose_number = 0;

for occasion in target_subject.iter_mut() {
    for event in occasion.iter_mut() {
        match event {
            Event::Bolus(bolus) => {
                if bolus.amount() == 0.0 {
                    bolus.set_amount(params[optimizable_dose_number]);
                    optimizable_dose_number += 1;
                }
            }
            Event::Infusion(infusion) => {
                // Only amount, duration not supported!
                if infusion.amount() == 0.0 {
                    infusion.set_amount(params[optimizable_dose_number]);
                    optimizable_dose_number += 1;
                }
                // Duration: ❌ NOT SUPPORTED
            }
            Event::Observation(_) => {}
        }
    }
}

AFTER (cost.rs):
───────────────────────────────────────────────────────────────
let target_subject = problem.dose_param_map
    .apply_to_subject(&problem.target, params)?;

// That's it! Map handles everything: ✨
// - Boluses (optimizable or fixed)
// - Infusions (amt only, dur only, or both)
// - Mixed dose types
// - Validation
```

## Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     Key Innovation                              │
└─────────────────────────────────────────────────────────────────┘

             DoseParameterMap
                    │
         ┌──────────┴──────────┐
         │                     │
    Bidirectional          Self-Contained
      Mapping               Knowledge
         │                     │
    ┌────┴────┐           ┌────┴────┐
    │         │           │         │
  Events → Params    Handles All
  Params → Events     Dose Types

Result: Simple, Robust, Maintainable Code ✨
```
