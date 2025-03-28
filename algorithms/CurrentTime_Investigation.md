# Investigation: current_time Events Concatenation

**Date**: October 17, 2025  
**Issue**: Investigating whether the implementation of `current_time` (events concatenation) is correct

## Summary of Findings

**STATUS: ⚠️ POTENTIAL ISSUE FOUND**

The Rust implementation appears to be correct in most aspects, but there is a **critical ambiguity** about what `current_time` should represent.

---

## Fortran MAKETMP Subroutine Analysis

### What MAKETMP Does (lines 12171-12550 in bestdose.for)

The Fortran `MAKETMP` subroutine:

1. **Reads File 41 ("past")**: Takes ONLY doses (no observations)
2. **Reads File 42 ("future")**: Takes both doses AND observations
3. **Creates File 43 (combined)**:
   - Past doses at **original times**
   - Future doses at **times + TNEXT**
   - Future observations at **times + TNEXT**

```fortran
C Line 12411 - Adding future doses:
READ(42,*) SIG,(RS(J),J=1,NI)
WRITE(43,*) SIG + TNEXT,(RS(J),J=1,NI)

C Line 12534 - Adding future observations:
READ(42,*) TIM,(YO(J),J=1,NUMEQT)
WRITE(43,*) TIM + TNEXT,(YO(J),J=1,NUMEQT)
```

### What is TNEXT?

From Fortran comments (lines 2909-2911):

```
"THE 'FUTURE' BEGINS AT TIME = TNEXT. IN OTHER WORDS, THE 'FUTURE'
FILE SHOULD START AS USUAL WITH TIME = 0, BUT ALL THE TIMES IN THIS
FILE WILL BE INCREASED BY TNEXT HOURS"
```

**TNEXT is the boundary time** between past and future. It's **user-provided** (line 2918).

---

## Rust Implementation Analysis

### Function: `concatenate_past_and_future` (mod.rs lines 343-393)

```rust
fn concatenate_past_and_future(
    past: &pharmsol::prelude::Subject,
    future: &pharmsol::prelude::Subject,
    current_time: f64,
) -> pharmsol::prelude::Subject {
    // Add past doses only (skip observations from past)
    for occasion in past.occasions() {
        for event in occasion.events() {
            match event {
                Event::Bolus(bolus) => {
                    builder = builder.bolus(bolus.time(), bolus.amount(), bolus.input());
                }
                Event::Observation(_) => {
                    // Skip observations from past
                }
            }
        }
    }

    // Add future events with time offset
    for occasion in future.occasions() {
        for event in occasion.events() {
            match event {
                Event::Bolus(bolus) => {
                    builder = builder.bolus(
                        bolus.time() + current_time,  // ✓ Offset by current_time
                        bolus.amount(),
                        bolus.input()
                    );
                }
                Event::Observation(obs) => {
                    builder = builder.observation(
                        obs.time() + current_time,     // ✓ Offset by current_time
                        obs.value().unwrap_or(0.0),
                        obs.outeq()
                    );
                }
            }
        }
    }

    builder.build()
}
```

### Rust vs Fortran Comparison

| Aspect                                                 | Fortran MAKETMP | Rust `concatenate_past_and_future` | Match? |
| ------------------------------------------------------ | --------------- | ---------------------------------- | ------ |
| Takes doses from past                                  | ✓               | ✓                                  | ✅ YES |
| Skips observations from past                           | ✓               | ✓                                  | ✅ YES |
| Offsets future dose times by TNEXT/current_time        | ✓               | ✓                                  | ✅ YES |
| Offsets future observation times by TNEXT/current_time | ✓               | ✓                                  | ✅ YES |
| Takes future doses                                     | ✓               | ✓                                  | ✅ YES |
| Takes future observations                              | ✓               | ✓                                  | ✅ YES |

**Conclusion: The Rust implementation correctly mirrors the Fortran logic.**

---

## THE CRITICAL QUESTION: What Should `current_time` Be?

### Current Rust Validation (mod.rs lines 518-536)

```rust
// Validate: current_time must be >= max time in past_data
if let Some(past_subject) = &past_data {
    let max_past_time = past_subject
        .occasions()
        .iter()
        .flat_map(|occ| occ.events())
        .map(|event| match event {
            Event::Bolus(b) => b.time(),
            Event::Infusion(i) => i.time(),
            Event::Observation(o) => o.time(),
        })
        .fold(0.0_f64, |max, time| max.max(time));

    if t < max_past_time {
        return Err(antml::anyhow!(
            "Invalid current_time: {} is before the last past_data event at time {}. \
            current_time must be >= the maximum time in past_data to avoid time travel!",
            t, max_past_time
        ));
    }
}
```

**This validation is correct** - it prevents `current_time` from being before the last past event (which would cause time overlap/collision).

### Two Valid Interpretations

#### Interpretation 1: current_time = max(past event times) ✅

**Example:**

- Past: dose at t=0, observation at t=6
- current_time = 6.0 (the time of the last observation)
- Future template: dose at t=0 (relative), target at t=24 (relative)
- Result: past dose at t=0, future dose at t=6, target at t=30

**This makes sense because:**

- The future starts immediately after the last past observation
- No gap in the timeline
- Matches the Fortran comment: "THE 'FUTURE' BEGINS AT TIME = TNEXT"

#### Interpretation 2: current_time > max(past event times) ✅

**Example:**

- Past: dose at t=0, observation at t=6
- current_time = 8.0 (user wants 2-hour gap)
- Future template: dose at t=0 (relative), target at t=24 (relative)
- Result: past dose at t=0, future dose at t=8, target at t=32

**This also makes sense because:**

- Allows for a "washout" or gap period
- User might want to schedule future doses at a specific clock time
- The validation allows this (current_time >= max_past_time)

---

## POTENTIAL ISSUE: Semantic Ambiguity

### The Problem

The Fortran documentation says:

> "ENTER TNEXT, A POSITIVE NO. OF HOURS, NOW"

This suggests TNEXT is **user-chosen** and could be either:

1. Equal to the last past time (Interpretation 1)
2. Greater than the last past time (Interpretation 2)

### The Risk

If users misunderstand what `current_time` should be, they might:

- Set it too early (caught by validation ✓)
- Set it incorrectly thinking it means something else
- Not understand the gap between past and future

### Documentation Issue

The Rust documentation (mod.rs lines 303-341) is **excellent** and provides a clear example:

```rust
/// // Past: dose at t=0, observation at t=6 (patient has been on therapy 6 hours)
/// let past = Subject::builder("patient")
///     .bolus(0.0, 500.0, 0)
///     .observation(6.0, 15.0, 0)  // 15 mg/L at 6 hours
///     .build();
///
/// // Future: dose at t=0 (relative), target at t=24 (relative)
/// let future = Subject::builder("patient")
///     .bolus(0.0, 100.0, 0)  // Dose to optimize, will be at t=6 absolute
///     .observation(24.0, 10.0, 0)  // Target at t=30 absolute
///     .build();
///
/// // Concatenate with current_time = 6.0
/// let combined = concatenate_past_and_future(&past, &future, 6.0);
/// // Result: dose at t=0 (fixed, 500mg), dose at t=6 (optimizable, 100mg initial),
/// //         observation target at t=30 (10 mg/L)
```

This example shows **Interpretation 1** (current_time = max_past_time = 6.0).

---

## Recommendations

### 1. ✅ Implementation is Correct

The Rust `concatenate_past_and_future` function correctly implements the Fortran MAKETMP logic.

### 2. ⚠️ Consider Enhanced Validation

Add a warning or check for common misunderstandings:

```rust
// After the existing validation...
if t == max_past_time {
    tracing::info!("  current_time = last past event time (no gap)");
} else if t > max_past_time {
    let gap = t - max_past_time;
    tracing::info!("  Gap between past and future: {} hours", gap);
    if gap > 24.0 {
        tracing::warn!(
            "  Large gap ({} hours) between past and future. Is this intentional?",
            gap
        );
    }
}
```

### 3. ✅ Documentation is Good

The existing documentation with the example is clear. Consider adding:

- A note about the choice of `current_time` value
- Whether to use `max_past_time` vs. a later time
- Clinical interpretation (e.g., "current_time represents the clock time when future dosing begins")

### 4. 📝 Test Coverage

Add tests for:

- `current_time == max_past_time` (no gap)
- `current_time > max_past_time` (with gap)
- Verify that events are correctly offset
- Verify that past observations are excluded

---

## Conclusion

### Is the implementation correct?

**YES** ✅

The Rust implementation correctly mirrors the Fortran MAKETMP logic:

- Takes only doses from past (no observations) ✓
- Offsets all future event times by `current_time` ✓
- Validation prevents time travel ✓
- Documentation is clear with good examples ✓

### Any concerns?

**MINOR** ⚠️

1. The semantic meaning of `current_time` could be clearer (is it always `max_past_time` or can it be later?)
2. Consider adding a warning for unusually large gaps between past and future
3. Consider documenting the clinical/practical meaning of `current_time`

### Action Items

1. Add the enhanced logging/validation suggested above (optional)
2. Add test cases for different `current_time` scenarios (recommended)
3. Update documentation to clarify the semantic meaning (optional)

**Overall: The implementation is correct and faithful to the Fortran algorithm.**
