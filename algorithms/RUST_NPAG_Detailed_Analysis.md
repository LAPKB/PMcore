# RUST NPAG Algorithm - Detailed Analysis

## File: src/algorithms/npag.rs

### Overview

Modern Rust implementation of the NPAG algorithm with separation of concerns, using Burke's Interior Point Method for optimization and separate QR decomposition for condensation.

---

## Main Algorithm Structure (Trait Implementation)

The Rust implementation follows the `Algorithms<E>` trait pattern with modular phases.

### 1. INITIALIZATION (new function, Lines 68-91)

```rust
fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>> {
    Ok(Box::new(Self {
        equation,
        ranges: settings.parameters().ranges(),
        psi: Psi::new(),
        theta: Theta::new(),
        lambda: Weights::default(),
        w: Weights::default(),
        eps: 0.2,                    // Initial resolution (20%)
        last_objf: -1e30,
        objf: f64::NEG_INFINITY,
        f0: -1e30,
        f1: f64::default(),
        cycle: 0,
        gamma_delta: vec![0.1; settings.errormodels().len()],
        error_models: settings.errormodels().clone(),
        converged: false,
        status: Status::Starting,
        cycle_log: CycleLog::new(),
        settings,
        data,
    }))
}
```

**Key Fields:**

- `theta: Theta`: Support points matrix
- `psi: Psi`: Likelihood matrix (nsub × npoint)
- `lambda: Weights`: Probability weights from IPM
- `w: Weights`: Final normalized weights
- `eps: f64`: Grid resolution (starts at 0.2)
- `gamma_delta: Vec<f64>`: Per-output-equation gamma adjustment factor

---

## 2. MAIN PHASES (Called by algorithm runner)

### Phase 2.1: EVALUATION (Lines 186-209)

```rust
fn evaluation(&mut self) -> Result<()> {
    // Calculate likelihood matrix
    self.psi = calculate_psi(
        &self.equation,
        &self.data,
        &self.theta,
        &self.error_models,
        self.cycle == 1 && self.settings.config().progress,
        self.cycle != 1,
    )?;

    self.validate_psi()?;

    // Optimize weights using Burke's IPM
    (self.lambda, _) = match burke(&self.psi) {
        Ok((lambda, objf)) => (lambda.into(), objf),
        Err(err) => {
            bail!("Error in IPM during evaluation: {:?}", err);
        }
    };
    Ok(())
}
```

**Key Difference from Fortran:**

- Fortran: Calculates psi in loop 800, stores in PYJGX array
- Rust: Delegates to `calculate_psi()` function (from pharmsol crate)
- Fortran: Integration via NOTINT
- Rust: Integration handled inside `burke()`

---

### Phase 2.2: CONDENSATION (Lines 211-268)

```rust
fn condensation(&mut self) -> Result<()> {
    // 1. Lambda filtering (max/1000 criterion)
    let max_lambda = self.lambda.iter().fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

    let mut keep = Vec::<usize>::new();
    for (index, lam) in self.lambda.iter().enumerate() {
        if lam > max_lambda / 1000_f64 {
            keep.push(index);
        }
    }

    if self.psi.matrix().ncols() != keep.len() {
        tracing::debug!(
            "Lambda (max/1000) dropped {} support point(s)",
            self.psi.matrix().ncols() - keep.len(),
        );
    }

    self.theta.filter_indices(keep.as_slice());
    self.psi.filter_column_indices(keep.as_slice());

    // 2. QR decomposition
    let (r, perm) = qr::qrd(&self.psi)?;

    let mut keep = Vec::<usize>::new();
    let keep_n = self.psi.matrix().ncols().min(self.psi.matrix().nrows());

    for i in 0..keep_n {
        let test = r.col(i).norm_l2();
        let r_diag_val = r.get(i, i);
        let ratio = r_diag_val / test;
        if ratio.abs() >= 1e-8 {
            keep.push(*perm.get(i).unwrap());
        }
    }

    if self.psi.matrix().ncols() != keep.len() {
        tracing::debug!(
            "QR decomposition dropped {} support point(s)",
            self.psi.matrix().ncols() - keep.len(),
        );
    }

    self.theta.filter_indices(keep.as_slice());
    self.psi.filter_column_indices(keep.as_slice());

    // 3. Re-run IPM after condensation
    self.validate_psi()?;
    (self.lambda, self.objf) = match burke(&self.psi) {
        Ok((lambda, objf)) => (lambda.into(), objf),
        Err(err) => {
            return Err(anyhow::anyhow!(
                "Error in IPM during condensation: {:?}",
                err
            ));
        }
    };
    self.w = self.lambda.clone().into();
    Ok(())
}
```

**Key Differences from Fortran:**

- Fortran: QR done INSIDE emint when ijob≠0
- Rust: QR done as separate phase, calls separate `qr::qrd()` function
- Fortran: Uses DQRDC from LINPACK
- Rust: Uses faer::linalg::solvers::ColPivQr
- Fortran: Sorts pivot indices to avoid collisions
- Rust: Uses perm vector directly (no sorting needed with modern API)

---

### Phase 2.3: OPTIMIZATIONS (Gamma) (Lines 270-358)

```rust
fn optimizations(&mut self) -> Result<()> {
    self.error_models
        .clone()
        .iter_mut()
        .filter_map(|(outeq, em)| {
            if em.optimize() {
                Some((outeq, em))
            } else {
                None
            }
        })
        .try_for_each(|(outeq, em)| -> Result<()> {
            // Calculate gamma_up and gamma_down
            let gamma_up = em.factor()? * (1.0 + self.gamma_delta[outeq]);
            let gamma_down = em.factor()? / (1.0 + self.gamma_delta[outeq]);

            // Create error models with adjusted gamma
            let mut error_model_up = self.error_models.clone();
            error_model_up.set_factor(outeq, gamma_up)?;

            let mut error_model_down = self.error_models.clone();
            error_model_down.set_factor(outeq, gamma_down)?;

            // Calculate psi with adjusted gamma
            let psi_up = calculate_psi(
                &self.equation,
                &self.data,
                &self.theta,
                &error_model_up,
                false,
                true,
            )?;
            let psi_down = calculate_psi(
                &self.equation,
                &self.data,
                &self.theta,
                &error_model_down,
                false,
                true,
            )?;

            // Run IPM for both
            let (lambda_up, objf_up) = match burke(&psi_up) {
                Ok((lambda, objf)) => (lambda, objf),
                Err(err) => {
                    return Err(anyhow::anyhow!("Error in IPM during optim: {:?}", err));
                }
            };
            let (lambda_down, objf_down) = match burke(&psi_down) {
                Ok((lambda, objf)) => (lambda, objf),
                Err(err) => {
                    return Err(anyhow::anyhow!("Error in IPM during optim: {:?}", err));
                }
            };

            // Accept better solution
            if objf_up > self.objf {
                self.error_models.set_factor(outeq, gamma_up)?;
                self.objf = objf_up;
                self.gamma_delta[outeq] *= 4.;
                self.lambda = lambda_up;
                self.psi = psi_up;
            }
            if objf_down > self.objf {
                self.error_models.set_factor(outeq, gamma_down)?;
                self.objf = objf_down;
                self.gamma_delta[outeq] *= 4.;
                self.lambda = lambda_down;
                self.psi = psi_down;
            }

            // Update gamma_delta
            self.gamma_delta[outeq] *= 0.5;
            if self.gamma_delta[outeq] <= 0.01 {
                self.gamma_delta[outeq] = 0.1;
            }
            Ok(())
        })?;

    Ok(())
}
```

**Key Differences from Fortran:**

- Fortran: Uses IGAMMA counter with MOD(IGAMMA, 3) logic
- Rust: Tests both gamma_up and gamma_down in same cycle
- Fortran: Three separate calls to emint (base, plus, minus)
- Rust: Two calls to burke (up, down) after base evaluation
- Both: Accept if improvement, multiply gamma_delta by 4; always multiply by 0.5 and reset to 0.1 if < 0.01

---

### Phase 2.4: CONVERGENCE EVALUATION (Lines 132-180)

```rust
fn convergence_evaluation(&mut self) {
    let psi = self.psi.matrix();
    let w = &self.w;

    // Check if objective improvement is small and eps can be reduced
    if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
        self.eps /= 2.;
        if self.eps <= THETA_E {
            // Calculate f1 and compare to f0
            let pyl = psi * w.weights();
            self.f1 = pyl.iter().map(|x| x.ln()).sum();
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

    // Stop if maximum cycles reached
    if self.cycle >= self.settings.config().cycles {
        tracing::warn!("Maximum number of cycles reached");
        self.converged = true;
        self.status = Status::MaxCycles;
    }

    // Stop if stopfile exists
    if std::path::Path::new("stop").exists() {
        tracing::warn!("Stopfile detected - breaking");
        self.status = Status::ManualStop;
    }

    // Create cycle log entry
    let state = NPCycle::new(
        self.cycle,
        -2. * self.objf,
        self.error_models.clone(),
        self.theta.clone(),
        self.theta.nspp(),
        (self.last_objf - self.objf).abs(),
        self.status.clone(),
    );

    self.cycle_log.push(state);
    self.last_objf = self.objf;
}
```

**Constants:**

- `THETA_E = 1e-4` (minimum resolution)
- `THETA_G = 1e-4` (objective convergence threshold)
- `THETA_F = 1e-2` (major cycle convergence)

**Key Differences from Fortran:**

- Fortran: Complex CHECKBIG calculation involving medians
- Rust: Simpler convergence based on `|f1 - f0| ≤ THETA_F`
- Fortran: RESOLVE variable (same as eps)
- Rust: Direct eps variable

---

### Phase 2.5: EXPANSION (Lines 380-384)

```rust
fn expansion(&mut self) -> Result<()> {
    adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D)?;
    Ok(())
}
```

Where `THETA_D = 1e-4` (minimum distance between points).

**Delegates to:** `src/routines/expansion/adaptative_grid.rs`

```rust
pub fn adaptative_grid(
    theta: &mut Theta,
    eps: f64,
    ranges: &[(f64, f64)],
    min_dist: f64,
) -> Result<()> {
    let mut candidates = Vec::new();

    // Collect all potential new points
    for spp in theta.matrix().row_iter() {
        for (j, val) in spp.iter().enumerate() {
            let l = eps * (ranges[j].1 - ranges[j].0);

            // Try adding point at +eps
            if val + l < ranges[j].1 {
                let mut plus = Row::zeros(spp.ncols());
                plus[j] = l;
                plus += spp;
                candidates.push(plus.iter().copied().collect::<Vec<f64>>());
            }

            // Try adding point at -eps
            if val - l > ranges[j].0 {
                let mut minus = Row::zeros(spp.ncols());
                minus[j] = -l;
                minus += spp;
                candidates.push(minus.iter().copied().collect::<Vec<f64>>());
            }
        }
    }

    // Filter candidates based on minimum distance
    let keep = candidates
        .iter()
        .filter(|point| theta.check_point(point, min_dist))
        .cloned()
        .collect::<Vec<_>>();

    // Add all valid candidates
    for point in keep {
        theta.add_point(point.as_slice())?;
    }

    Ok(())
}
```

**Key Differences from Fortran:**

- Fortran: Divides existing point probability by (2\*NVAR + 1)
- Rust: Doesn't adjust probabilities (will be recalculated in next cycle)
- Fortran: Adds points immediately during loop
- Rust: Collects candidates, filters, then adds
- Both: Check bounds and minimum distance before adding

---

## 3. BURKE'S INTERIOR POINT METHOD (src/routines/evaluation/ipm.rs)

### Algorithm (Lines 33-282)

```rust
pub fn burke(psi: &Psi) -> anyhow::Result<(Weights, f64)> {
    let mut psi = psi.matrix().to_owned();

    // Ensure finite and non-negative
    psi.row_iter_mut().try_for_each(|row| {
        row.iter_mut().try_for_each(|x| {
            if !x.is_finite() {
                bail!("Input matrix must have finite entries")
            } else {
                *x = x.abs();
                Ok(())
            }
        })
    })?;

    let (n_sub, n_point) = psi.shape();

    // Create unit vectors
    let ecol: Col<f64> = Col::from_fn(n_point, |_| 1.0);
    let erow: Row<f64> = Row::from_fn(n_sub, |_| 1.0);

    // Initialize: plam = psi · ecol (row sums)
    let mut plam: Col<f64> = &psi * &ecol;
    let eps: f64 = 1e-8;
    let mut sig: f64 = 0.0;

    // Initialize lam = ones
    let mut lam = ecol.clone();

    // w = 1 ./ plam
    let mut w: Col<f64> = Col::from_fn(plam.nrows(), |i| 1.0 / plam.get(i));

    // ptw = ψᵀ · w
    let mut ptw: Col<f64> = psi.transpose() * &w;

    // Scaling (SHRINK = 2 * max(ptw))
    let ptw_max = ptw.iter().fold(f64::NEG_INFINITY, |acc, &x| x.max(acc));
    let shrink = 2.0 * ptw_max;
    lam *= shrink;
    plam *= shrink;
    w /= shrink;
    ptw /= shrink;

    // y = ecol - ptw
    let mut y: Col<f64> = &ecol - &ptw;

    // r = erow - (w .* plam)
    let mut r: Col<f64> = Col::from_fn(n_sub, |i| erow.get(i) - w.get(i) * plam.get(i));
    let mut norm_r: f64 = r.iter().fold(0.0, |max, &val| max.max(val.abs()));

    // Duality gap
    let sum_log_plam: f64 = plam.iter().map(|x| x.ln()).sum();
    let sum_log_w: f64 = w.iter().map(|x| x.ln()).sum();
    let mut gap: f64 = (sum_log_w + sum_log_plam).abs() / (1.0 + sum_log_plam);

    // Duality measure
    let mut mu = lam.transpose() * &y / n_point as f64;

    // Allocate Hessian and workspace
    let mut psi_inner: Mat<f64> = Mat::zeros(psi.nrows(), psi.ncols());
    let n_threads = faer::get_global_parallelism().degree();
    let rows = psi.nrows();
    let mut output: Vec<Mat<f64>> = (0..n_threads).map(|_| Mat::zeros(rows, rows)).collect();
    let mut h: Mat<f64> = Mat::zeros(rows, rows);

    // Main IPM loop
    while mu > eps || norm_r > eps || gap > eps {
        let smu = sig * mu;
        let inner = Col::from_fn(lam.nrows(), |i| lam.get(i) / y.get(i));
        let w_plam = Col::from_fn(plam.nrows(), |i| plam.get(i) / w.get(i));

        // Build Hessian: H = (psi * diag(inner)) * psi' + diag(w_plam)
        // Uses parallel computation if psi.ncols() > n_threads * 128

        if psi.ncols() > n_threads * 128 {
            // Parallel path (omitted for brevity)
            // ...
        } else {
            // Sequential path
            psi_inner
                .as_mut()
                .col_iter_mut()
                .zip(psi.col_iter())
                .zip(inner.iter())
                .for_each(|((col, psi_col), inner_val)| {
                    col.iter_mut().zip(psi_col.iter()).for_each(|(x, psi_val)| {
                        *x = psi_val * inner_val;
                    });
                });
            faer::linalg::matmul::triangular::matmul(
                h.as_mut(),
                faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                faer::Accum::Replace,
                &psi_inner,
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                psi.transpose(),
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                1.0,
                faer::Par::Seq,
            );
        }

        // Add diagonal
        for i in 0..h.nrows() {
            h[(i, i)] += w_plam[i];
        }

        // Cholesky decomposition
        let uph = match h.llt(faer::Side::Lower) {
            Ok(llt) => llt,
            Err(_) => {
                bail!("Error during Cholesky decomposition. The matrix might not be positive definite. This is usually due to model misspecification or numerical issues.")
            }
        };
        let uph = uph.L().transpose().to_owned();

        // Construct RHS: rhsdw = (erow ./ w) - (psi · smuyinv)
        let smuyinv: Col<f64> = Col::from_fn(ecol.nrows(), |i| smu * (ecol[i] / y[i]));
        let psi_dot_muyinv: Col<f64> = &psi * &smuyinv;
        let rhsdw: Row<f64> = Row::from_fn(erow.ncols(), |i| erow[i] / w[i] - psi_dot_muyinv[i]);

        let mut dw = Mat::from_fn(rhsdw.ncols(), 1, |i, _j| *rhsdw.get(i));

        // Solve triangular systems: uph' * uph * dw = rhsdw
        solve_lower_triangular_in_place(uph.transpose().as_ref(), dw.as_mut(), faer::Par::rayon(0));
        solve_upper_triangular_in_place(uph.as_ref(), dw.as_mut(), faer::Par::rayon(0));

        let dw = dw.col(0);

        // Compute dy = -(ψᵀ · dw)
        let dy = -(psi.transpose() * dw);

        // Compute dlam = smuyinv - lam - (inner .* dy)
        let inner_times_dy = Col::from_fn(ecol.nrows(), |i| inner[i] * dy[i]);
        let dlam: Row<f64> =
            Row::from_fn(ecol.nrows(), |i| smuyinv[i] - lam[i] - inner_times_dy[i]);

        // Calculate primal step length
        let ratio_dlam_lam = Row::from_fn(lam.nrows(), |i| dlam[i] / lam[i]);
        let min_ratio_dlam = ratio_dlam_lam.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfpri: f64 = -1.0 / min_ratio_dlam.min(-0.5);
        alfpri = (0.99995 * alfpri).min(1.0);

        // Calculate dual step length
        let ratio_dy_y = Row::from_fn(y.nrows(), |i| dy[i] / y[i]);
        let min_ratio_dy = ratio_dy_y.iter().cloned().fold(f64::INFINITY, f64::min);
        let ratio_dw_w = Row::from_fn(dw.nrows(), |i| dw[i] / w[i]);
        let min_ratio_dw = ratio_dw_w.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfdual = -1.0 / min_ratio_dy.min(-0.5);
        alfdual = alfdual.min(-1.0 / min_ratio_dw.min(-0.5));
        alfdual = (0.99995 * alfdual).min(1.0);

        // Update iterates
        lam += alfpri * dlam.transpose();
        w += alfdual * dw;
        y += alfdual * &dy;

        mu = lam.transpose() * &y / n_point as f64;
        plam = &psi * &lam;
        r = Col::from_fn(n_sub, |i| erow.get(i) - w.get(i) * plam.get(i));
        ptw -= alfdual * dy;

        norm_r = r.norm_max();
        let sum_log_plam: f64 = plam.iter().map(|x| x.ln()).sum();
        let sum_log_w: f64 = w.iter().map(|x| x.ln()).sum();
        gap = (sum_log_w + sum_log_plam).abs() / (1.0 + sum_log_plam);

        // Adjust sigma
        if mu < eps && norm_r > eps {
            sig = 1.0;
        } else {
            let candidate1 = (1.0 - alfpri).powi(2);
            let candidate2 = (1.0 - alfdual).powi(2);
            let candidate3 = (norm_r - mu) / (norm_r + 100.0 * mu);
            sig = candidate1.max(candidate2).max(candidate3).min(0.3);
        }
    }

    // Finalize
    lam /= n_sub as f64;
    let obj = (psi * &lam).iter().map(|x| x.ln()).sum();
    let lam_sum: f64 = lam.iter().sum();
    lam = &lam / lam_sum;

    Ok((lam.into(), obj))
}
```

**Key Differences from Fortran emint:**

- Fortran: Condensation integrated (when ijob≠0)
- Rust: Only does IPM optimization
- Fortran: Uses DPOTRF/DPOTRS from LAPACK
- Rust: Uses faer::llt() for Cholesky, triangular solvers
- Fortran: All loops explicit
- Rust: Uses iterators, functional style
- Fortran: Can use parallel matmul (BLAS)
- Rust: Has explicit parallel path for large matrices (ncols > n_threads\*128)
- Both: Same convergence criteria (eps=1e-8 vs eps=1e-10, very similar)
- Both: Same step length calculation (0.99995 safety factor)
- Both: Same sigma adjustment strategy

---

## 4. QR DECOMPOSITION (src/routines/evaluation/qr.rs)

```rust
pub fn qrd(psi: &Psi) -> Result<(Mat<f64>, Vec<usize>)> {
    let mut mat = psi.matrix().to_owned();

    // Normalize rows to sum to 1
    for (index, row) in mat.row_iter_mut().enumerate() {
        let row_sum: f64 = row.as_ref().iter().sum();

        if row_sum.abs() == 0.0 {
            bail!("In psi, the row with index {} sums to zero", index);
        }
        row.iter_mut().for_each(|x| *x /= row_sum);
    }

    // Perform column pivoted QR decomposition
    let qr: ColPivQr<f64> = mat.col_piv_qr();

    // Extract R matrix
    let r_mat: faer::Mat<f64> = qr.R().to_owned();

    // Get permutation
    let perm = qr.P().arrays().0.to_vec();
    Ok((r_mat, perm))
}
```

**Key Differences from Fortran:**

- Fortran: Uses DQRDC from LINPACK (older library)
- Rust: Uses faer::ColPivQr (modern library)
- Fortran: Row normalization done before DQRDC call
- Rust: Row normalization done inside qrd()
- Fortran: Returns permutation as IPIVOT array
- Rust: Returns permutation as Vec<usize>
- Both: Normalize psi rows by subject likelihood sum
- Both: Column-pivoted QR for numerical stability

---

## KEY HYPERPARAMETERS

| Parameter            | Rust Value    | Fortran Value | Match?              |
| -------------------- | ------------- | ------------- | ------------------- |
| `THETA_E`            | 1e-4          | 1e-4          | ✓                   |
| `THETA_G`            | 1e-4          | 1e-4          | ✓                   |
| `THETA_F`            | 1e-2          | 1e-2          | ✓                   |
| `THETA_D`            | 1e-4          | 1e-4          | ✓                   |
| Initial `eps`        | 0.2           | 0.2           | ✓                   |
| IPM `eps`            | 1e-8          | 1e-10         | ≈ (both very small) |
| Lambda cutoff        | max/1000      | max/1000      | ✓                   |
| QR threshold         | 1e-8          | 1e-8          | ✓                   |
| `SHRINK`             | 2 \* max(PTW) | 2 \* max(PTW) | ✓                   |
| Step safety          | 0.99995       | 0.99995       | ✓                   |
| `SIG` max            | 0.3           | 0.3           | ✓                   |
| Gamma delta init     | 0.1           | 0.1           | ✓                   |
| Gamma delta multiply | 4.0 / 0.5     | 4.0 / 0.5     | ✓                   |
| Gamma delta reset    | 0.01 → 0.1    | 0.01 → 0.1    | ✓                   |

---

## CONVERGENCE CRITERIA

### IPM Convergence (burke):

- Rust: `mu ≤ 1e-8 && norm_r ≤ 1e-8 && gap ≤ 1e-8`
- Fortran: `mu ≤ 1e-10 && rmax ≤ 1e-10 && gap ≤ 1e-10`
- **Difference:** Rust uses slightly looser tolerance (1e-8 vs 1e-10), negligible in practice

### NPAG Convergence:

- Rust: `|Δ objf| ≤ 1e-4` AND `eps ≤ 1e-4` AND `|f1 - f0| ≤ 0.01`
- Fortran: `|Δ fobj| ≤ 1e-4` AND `resolve ≤ 0.0001` AND `|CHECKBIG| ≤ 0.01`
- **Difference:**
  - Fortran's CHECKBIG is median of parameter changes
  - Rust's f1-f0 is change in log-likelihood sum
  - These are different metrics but serve similar purpose

---

## ARCHITECTURAL DIFFERENCES

### Fortran:

- **Monolithic:** Main loop contains all logic
- **Integrated:** emint does IPM + condensation
- **File-based:** Reads from scratch files 27/37
- **Single-threaded:** Some BLAS parallelism possible
- **Fixed arrays:** Dimension limits in PARAMETER statements

### Rust:

- **Modular:** Phases separated into trait methods
- **Separated:** burke() for IPM, qr::qrd() for QR, separate condensation phase
- **In-memory:** Data structures passed as references
- **Parallel-aware:** Explicit parallel path for large matrices
- **Dynamic:** Vec and Mat types grow as needed

---

## NUMERICAL DIFFERENCES TO INVESTIGATE

1. **Convergence Metric:**

   - Fortran: CHECKBIG = median parameter changes
   - Rust: f1 - f0 = log-likelihood change
   - **Impact:** Could converge at different points

2. **Probability Weight Handling:**

   - Fortran: Divides probabilities during expansion
   - Rust: Doesn't adjust (recalculated next cycle)
   - **Impact:** Intermediate probabilities differ, but final should match

3. **IPM Tolerance:**

   - Fortran: 1e-10
   - Rust: 1e-8
   - **Impact:** Minimal, both very tight

4. **Gamma Optimization Order:**

   - Fortran: Tests base, then plus, then minus over 3 cycles
   - Rust: Tests plus and minus in same cycle
   - **Impact:** Rust potentially faster, but different exploration pattern

5. **QR Pivot Sorting:**
   - Fortran: Explicitly sorts IPIVOT to avoid collisions
   - Rust: Uses permutation vector directly
   - **Impact:** Should be equivalent if modern API handles correctly

---

## SUMMARY

The Rust implementation is a **faithful modern translation** of the Fortran algorithm with:

### Identical Core Math:

- Same IPM formulation (Burke's method)
- Same lambda filtering threshold (max/1000)
- Same QR threshold (1e-8)
- Same expansion strategy
- Same gamma optimization strategy
- Same hyperparameters

### Architectural Improvements:

- Modular phase separation
- Type safety
- Memory safety
- Better error handling
- Parallel computation support
- Modern linear algebra libraries

### Potential Numerical Differences:

- Convergence metric (CHECKBIG vs f1-f0)
- Gamma optimization timing (interleaved vs sequential)
- IPM tolerance (1e-10 vs 1e-8, negligible)
- QR pivot handling (explicit sort vs modern API)

### Recommendation:

These implementations should produce **nearly identical results** for well-conditioned problems. The convergence metric difference could cause convergence at slightly different cycles, but final posteriors should be very close.
