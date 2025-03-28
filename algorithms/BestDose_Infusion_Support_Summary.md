# BestDose Infusion Support - Executive Summary

## The Problem

Currently, BestDose can only optimize **bolus dose amounts**. We want to support:
- ✅ Bolus amounts (already works)
- ❌ Infusion amounts (partially works)
- ❌ Infusion durations (**not implemented**)
- ❌ Both amount + duration together

## The Core Challenge

Different event types contribute different numbers of optimization parameters:

```
Event Type                    → Optimization Parameters
════════════════════════════════════════════════════════
Bolus (amount=0)             → 1 parameter (amount)
Bolus (amount=500)           → 0 parameters (fixed)
Infusion (amt=0, dur=2h)     → 1 parameter (amount)
Infusion (amt=500, dur=0)    → 1 parameter (duration)
Infusion (amt=0, dur=0)      → 2 parameters (both!)
Infusion (amt=500, dur=2h)   → 0 parameters (fixed)
```

**Problem**: How do we track which parameter in the optimization vector corresponds to which dose event?

## The Solution: DoseParameterMap

Create a **bidirectional mapping structure** that tracks:
1. Which events are optimizable
2. Which parameter indices belong to each event
3. How to apply parameters back to events

```rust
struct DoseParameterMap {
    entries: Vec<DoseParameterEntry>,
    total_params: usize,
}

struct DoseParameterEntry {
    event_index: usize,
    dose_type: DoseType,
    param_indices: Vec<usize>,  // Empty, 1, or 2 elements
}
```

### Example Usage

```rust
// Create from subject
let subject = Subject::builder("patient")
    .bolus(0.0, 0.0, 0)           // Param 0: amount
    .infusion(12.0, 0.0, 0, 0.0)  // Param 1: amount, Param 2: duration
    .bolus(24.0, 500.0, 0)        // Fixed (no params)
    .build();

let map = DoseParameterMap::from_subject(&subject);

// Use in optimization
let params = vec![100.0, 50.0, 1.5];  // [bolus_amt, inf_amt, inf_dur]
let result = map.apply_to_subject(&subject, &params)?;
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- Add `DoseParameterMap` struct
- Integrate into `BestDoseProblem`
- Basic tests

### Phase 2: Refactor Optimization (Week 1-2)
- Update `cost.rs` to use map
- Update `optimization.rs` to use map
- Remove manual index tracking

### Phase 3: Update Predictions (Week 2)
- Support infusions in `predictions.rs`
- Handle infusions in AUC mode
- Remove warning messages

### Phase 4: Enhanced Features (Week 3)
- Add duration ranges to `DoseRange`
- Smart simplex initialization
- Comprehensive documentation

### Phase 5: Testing & Examples (Week 3-4)
- Unit tests for all modules
- Integration tests
- Complete infusion example

## Key Benefits

✅ **Eliminates index tracking bugs** - No more manual counter increments
✅ **Handles all dose combinations** - Bolus/infusion mixing just works
✅ **Type-safe** - Compile-time guarantees
✅ **Backward compatible** - Existing code unchanged
✅ **Self-documenting** - Map structure is explicit
✅ **Easy to extend** - Future dose types can be added easily

## Example Use Cases

### 1. Optimize Infusion Amount Only
```rust
let target = Subject::builder("patient")
    .infusion(0.0, 0.0, 0, 2.0)  // amt=0 (optimize), dur=2h (fixed)
    .observation(24.0, 5.0, 0)
    .build();

// Result: [optimal_amount]
```

### 2. Optimize Both Amount and Duration
```rust
let target = Subject::builder("patient")
    .infusion(0.0, 0.0, 0, 0.0)  // Both optimizable!
    .observation(24.0, 5.0, 0)
    .build();

// Result: [optimal_amount, optimal_duration]
```

### 3. Loading Bolus + Maintenance Infusion
```rust
let target = Subject::builder("patient")
    .bolus(0.0, 0.0, 0)          // Loading dose
    .infusion(0.0, 0.0, 0, 0.0)  // Maintenance
    .observation(2.0, 10.0, 0)   // Peak
    .observation(24.0, 5.0, 0)   // Trough
    .build();

// Result: [bolus_amount, infusion_amount, infusion_duration]
```

## Migration Path

**No breaking changes!**

- Old code using only boluses: **Works unchanged**
- Old code with fixed infusions: **Works unchanged**
- New code with optimizable infusions: **Works automatically**

## Timeline

**Total: 3-4 weeks**

- Week 1: Core infrastructure + refactoring
- Week 2: Predictions + helpers
- Week 3: Features + documentation
- Week 4: Testing + validation

## Success Criteria

✅ All existing tests pass
✅ Can optimize: amount, duration, or both
✅ Can mix boluses and infusions
✅ AUC mode works with infusions
✅ Complete documentation and examples
✅ No performance regression

## Next Steps

1. Review and approve this plan
2. Create GitHub issue/milestone
3. Start Phase 1 implementation
4. Iterative development with testing

---

**Full detailed plan**: See `BestDose_Infusion_Support_Plan.md`
