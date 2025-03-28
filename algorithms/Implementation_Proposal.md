# Implementation Proposal: CHECKBIG and Fortran-style Gamma Timing

## Overview

Analysis of implementation difficulty and recommendations for adding Fortran-compatible features to Rust NPAG.

---

## 1. CHECKBIG Convergence Metric

### Difficulty: **EASY** ⭐

### What it is

Fortran calculates CHECKBIG as the median of relative parameter changes:

```fortran
CHECKBIG = median(abs((theta_new - theta_old) / theta_old))
```

### Implementation Steps

#### Step 1: Add helper function (10 lines)

```rust
// In src/algorithms/npag.rs or new src/algorithms/convergence.rs

/// Calculate CHECKBIG metric as in Fortran NPAGFULLA
/// Returns median of relative parameter changes across all support points
fn calculate_checkbig(theta_old: &Theta, theta_new: &Theta) -> f64 {
    let mut changes = Vec::new();
    let old_mat = theta_old.matrix();
    let new_mat = theta_new.matrix();

    for row in 0..old_mat.nrows() {
        for col in 0..old_mat.ncols() {
            let old_val = old_mat.get(row, col);
            let new_val = new_mat.get(row, col);

            if old_val.abs() > 1e-10 {
                let rel_change = ((new_val - old_val) / old_val).abs();
                changes.push(rel_change);
            }
        }
    }

    if changes.is_empty() {
        return 0.0;
    }

    changes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = changes.len() / 2;

    if changes.len() % 2 == 0 {
        (changes[mid - 1] + changes[mid]) / 2.0
    } else {
        changes[mid]
    }
}
```

#### Step 2: Store old theta (2 lines)

```rust
// In NPAG struct
pub struct NPAG<E: Equation> {
    // ... existing fields ...
    theta_old: Option<Theta>,  // Add this field
}
```

#### Step 3: Calculate in convergence_evaluation (5 lines)

```rust
fn convergence_evaluation(&mut self) {
    let psi = self.psi.matrix();
    let w = &self.w;

    if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
        self.eps /= 2.;
        if self.eps <= THETA_E {
            let pyl = psi * w.weights();
            self.f1 = pyl.iter().map(|x| x.ln()).sum();

            // Calculate CHECKBIG if we have old theta
            let checkbig = if let Some(ref old_theta) = self.theta_old {
                calculate_checkbig(old_theta, &self.theta)
            } else {
                f64::MAX  // First cycle, no old theta
            };

            // Log both metrics
            tracing::info!(
                "Cycle {}: f1-f0={:.6e}, CHECKBIG={:.6e}",
                self.cycle, (self.f1 - self.f0).abs(), checkbig
            );

            // Use f1-f0 by default, but could switch based on flag
            if (self.f1 - self.f0).abs() <= THETA_F {
                tracing::info!("The model converged after {} cycles", self.cycle);
                self.converged = true;
                self.status = Status::Converged;
            } else {
                self.f0 = self.f1;
                self.eps = 0.2;
            }
        }
    }
    self.last_objf = self.objf;

    // Save current theta for next cycle's CHECKBIG calculation
    self.theta_old = Some(self.theta.clone());
}
```

#### Step 4: Add feature flag (optional, 10 lines)

```rust
// In Settings struct
pub struct Settings {
    // ... existing fields ...
    convergence_metric: ConvergenceMetric,
}

pub enum ConvergenceMetric {
    F1MinusF0,      // Default: Rust approach
    CheckBig,       // Fortran approach
    Both,           // Require both to converge
}

// Then in convergence check:
let converged = match self.settings.convergence_metric {
    ConvergenceMetric::F1MinusF0 => (self.f1 - self.f0).abs() <= THETA_F,
    ConvergenceMetric::CheckBig => checkbig <= THETA_E,
    ConvergenceMetric::Both => {
        (self.f1 - self.f0).abs() <= THETA_F && checkbig <= THETA_E
    }
};
```

### Total Implementation Time

- **Core functionality**: 30 minutes
- **Feature flag + tests**: 1 hour
- **Documentation**: 30 minutes
- **Total**: ~2 hours

### Benefits

✅ Provides diagnostic information  
✅ Enables exact Fortran reproduction studies  
✅ Allows comparison of convergence criteria  
✅ Minimal performance impact (only when eps <= THETA_E)

### Recommendation

**IMPLEMENT** - Easy win for compatibility and diagnostics

---

## 2. Fortran-style Gamma Timing (Cycles 1,2,3 only)

### Difficulty: **VERY EASY** ⭐

### What it is

Fortran only optimizes gamma in cycles 1, 2, and 3, then stops.  
Rust optimizes gamma every cycle.

### Implementation Steps

#### Step 1: Add cycle check in optimizations() (3 lines)

```rust
fn optimizations(&mut self) -> Result<()> {
    // Add feature flag check
    let should_optimize_gamma = match self.settings.gamma_optimization_mode {
        GammaMode::Continuous => true,  // Current Rust behavior
        GammaMode::Fortran => self.cycle <= 3,  // Fortran behavior
    };

    if !should_optimize_gamma {
        return Ok(());  // Skip gamma optimization
    }

    // Rest of existing optimization code...
    self.error_models
        .clone()
        .iter_mut()
        .filter_map(|(outeq, em)| {
            // ... existing gamma optimization ...
        })
        // ...
}
```

#### Step 2: Add settings enum (5 lines)

```rust
// In Settings
pub enum GammaOptimizationMode {
    Continuous,  // Default: optimize every cycle
    Fortran,     // Optimize only cycles 1, 2, 3
}

pub struct Settings {
    // ... existing fields ...
    gamma_optimization_mode: GammaOptimizationMode,
}
```

#### Step 3: Add CLI flag (if using clap)

```rust
#[arg(long, default_value = "continuous")]
gamma_mode: String,  // "continuous" or "fortran"
```

### Total Implementation Time

- **Core functionality**: 15 minutes
- **Settings integration**: 15 minutes
- **Tests**: 30 minutes
- **Documentation**: 15 minutes
- **Total**: ~1 hour 15 minutes

### Pros and Cons

**Pros:**

- ✅ Extremely easy to implement
- ✅ Enables exact Fortran reproduction
- ✅ Could reduce computation time (though gamma opt is cheap)
- ✅ Useful for compatibility testing

**Cons:**

- ❌ Fortran approach is theoretically inferior
- ❌ Could confuse users (why have a worse mode?)
- ❌ Adds maintenance burden
- ❌ May give worse results in some cases

### Recommendation

**OPTIONAL** - Only implement if you need exact Fortran reproduction for validation studies.

Default should always be `Continuous` mode.

---

## 3. Combined Implementation

If implementing both, they work together nicely:

```rust
// Settings builder pattern
let settings = Settings::builder()
    .set_algorithm(Algorithm::NPAG)
    .set_convergence_metric(ConvergenceMetric::Both)  // Require both metrics
    .set_gamma_mode(GammaMode::Fortran)  // Use Fortran gamma timing
    .build();
```

This would create "maximum Fortran compatibility mode" for validation.

---

## Implementation Priority

### High Priority: CHECKBIG

**Reason**: Provides valuable diagnostic information regardless of which metric you use for convergence.

**Suggested Implementation:**

1. Calculate CHECKBIG every cycle (cheap operation)
2. Log both f1-f0 AND CHECKBIG
3. Keep f1-f0 as default convergence criterion
4. Add optional flag `--convergence-metric checkbig` for Fortran mode

Example output:

```
INFO  Cycle 42: f1-f0=8.3e-5, CHECKBIG=1.2e-4 (both improving)
INFO  Cycle 43: f1-f0=3.1e-5, CHECKBIG=9.5e-5 (f1-f0 converged ✓)
INFO  The model converged after 43 cycles
```

### Low Priority: Fortran Gamma Mode

**Reason**: Current continuous approach is better. Only needed for exact reproduction studies.

**Suggested Implementation:**

- Add as hidden/advanced flag: `--gamma-cycles-123-only`
- Don't advertise it in main documentation
- Use only when comparing against Fortran results

---

## Code Structure Suggestions

Create new file: `src/algorithms/convergence.rs`

```rust
pub mod convergence {
    use crate::structs::theta::Theta;

    pub enum ConvergenceMetric {
        F1MinusF0,
        CheckBig,
        Both,
    }

    pub fn calculate_checkbig(theta_old: &Theta, theta_new: &Theta) -> f64 {
        // Implementation here
    }

    pub fn evaluate_convergence(
        f0: f64,
        f1: f64,
        theta_old: &Theta,
        theta_new: &Theta,
        metric: &ConvergenceMetric,
        theta_e: f64,
        theta_f: f64,
    ) -> bool {
        let f1_f0_converged = (f1 - f0).abs() <= theta_f;
        let checkbig = calculate_checkbig(theta_old, theta_new);
        let checkbig_converged = checkbig <= theta_e;

        match metric {
            ConvergenceMetric::F1MinusF0 => f1_f0_converged,
            ConvergenceMetric::CheckBig => checkbig_converged,
            ConvergenceMetric::Both => f1_f0_converged && checkbig_converged,
        }
    }
}
```

This keeps convergence logic separate and testable.

---

## Testing Strategy

### For CHECKBIG

```rust
#[test]
fn test_checkbig_convergence() {
    // Create scenarios where CHECKBIG would converge but f1-f0 wouldn't
    // and vice versa - verify both work correctly
}

#[test]
fn test_checkbig_calculation_accuracy() {
    // Verify median calculation matches Fortran exactly
}
```

### For Gamma Timing

```rust
#[test]
fn test_gamma_fortran_mode() {
    // Verify gamma only optimized in cycles 1, 2, 3
    // Verify gamma_delta unchanged after cycle 3
}

#[test]
fn test_gamma_continuous_mode() {
    // Verify gamma optimized every cycle (current behavior)
}
```

---

## Final Recommendation

### ✅ IMPLEMENT: CHECKBIG (High Value, Low Effort)

**Timeline**: 2 hours  
**Benefits**:

- Diagnostic insight into convergence behavior
- Compatibility with Fortran validation studies
- Minimal code complexity
- Can be logged alongside f1-f0 without changing default behavior

### ⚠️ OPTIONAL: Fortran Gamma Mode (Low Value, Very Low Effort)

**Timeline**: 1 hour  
**Benefits**:

- Exact Fortran reproduction for validation
- Could slightly reduce computation (negligible)

**Use Cases**:

- Publishing paper comparing Fortran and Rust implementations
- Validating against historical Fortran results
- Regulatory submission requiring demonstrated equivalence

### 🎯 Suggested Implementation Order

1. **Now**: Implement CHECKBIG calculation and logging
2. **Later**: Add convergence metric selection flag if users request it
3. **Much Later**: Add Fortran gamma mode only if needed for specific validation study

---

## Estimated Total Effort

| Feature               | Core | Tests | Docs | Total      |
| --------------------- | ---- | ----- | ---- | ---------- |
| CHECKBIG calculation  | 30m  | 30m   | 20m  | 1h 20m     |
| CHECKBIG feature flag | 20m  | 20m   | 10m  | 50m        |
| Gamma timing mode     | 15m  | 30m   | 15m  | 1h         |
| **Total**             |      |       |      | **3h 10m** |

Both features combined: **Half a day of work** for full implementation, testing, and documentation.

---

## Questions to Consider

1. **Do you need exact Fortran reproduction?**

   - If YES → Implement both features
   - If NO → Just implement CHECKBIG for diagnostics

2. **Are you publishing a comparison paper?**

   - If YES → Definitely implement both
   - If NO → CHECKBIG is sufficient

3. **Do users want to choose convergence criteria?**

   - If YES → Full CHECKBIG feature flag
   - If NO → Just log CHECKBIG, keep f1-f0 as criterion

4. **Is there regulatory requirement for Fortran equivalence?**
   - If YES → Implement Fortran gamma mode
   - If NO → Skip it

Let me know what you think and I can implement whichever features make sense for your use case!
