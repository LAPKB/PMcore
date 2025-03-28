# Burke vs Emint Analysis

## Key Finding

**Burke (Rust)** ≠ **Emint (Fortran)** when it comes to QR decomposition!

### Fortran `emint` behavior:

```fortran
subroutine emint(psi,ldpsi,theta,ldtheta,npoint,nsub,ijob,...)
```

- **`ijob=0`**: IPM optimization only, returns normalized weights
- **`ijob=1`**: IPM optimization + lambda filtering (1e-3 threshold) + QR decomposition + second IPM, returns filtered theta and weights

### Rust `burke` behavior:

```rust
pub fn burke(psi: &Psi) -> Result<(Weights, f64)>
```

- **Always**: IPM optimization only, returns normalized weights
- **Never**: Does QR decomposition (that's handled separately in NPAG's `condensation()`)

## Fortran NPAG Pattern

In NPAGFULLA11.FOR (and bestdose.for):

```fortran
! First call - WITH QR decomposition
call emint(pyjgx,maxsub,corden,maxgrd,nactve,nsub,1,...)
!             ↑ psi    ↑ theta                          ↑ ijob=1

nactve = keep  ! Update count after filtering

! Second call - NO QR decomposition
call emint(pyjgx,maxsub,corden,maxgrd,nactve,nsub,0,...)
!                                                    ↑ ijob=0
```

**Result**: Two `emint` calls, first does filtering+QR, second recalculates weights

## Rust NPAG Pattern

In `src/algorithms/npag.rs`:

### evaluation() method:

```rust
fn evaluation(&mut self) -> Result<()> {
    self.psi = calculate_psi(...)?;
    (self.lambda, _) = burke(&self.psi)?;  // First burke - like emint ijob=0
    Ok(())
}
```

### condensation() method:

```rust
fn condensation(&mut self) -> Result<()> {
    // Lambda filtering (max/1000)
    let keep: Vec<usize> = self.lambda.iter()
        .filter(|lam| lam > max_lambda / 1000_f64)
        .collect();

    self.theta.filter_indices(&keep);
    self.psi.filter_column_indices(&keep);

    // QR decomposition
    let (r, perm) = qr::qrd(&self.psi)?;

    let keep_qr: Vec<usize> = /* QR filtering logic */;

    self.theta.filter_indices(&keep_qr);
    self.psi.filter_column_indices(&keep_qr);

    // Second burke call
    (self.lambda, self.objf) = burke(&self.psi)?;  // Second burke - recalculate
    self.w = self.lambda.clone().into();
    Ok(())
}
```

**Result**: Two `burke` calls separated by filtering+QR, exactly like Fortran!

## BestDose NPAGFULL11 Implementation

### Fortran (bestdose.for line 3673):

```fortran
CALL NPAGFULL11(MAXSUB,MAXGRD,MAXDIM,NVAR,NUMEQT,WORK,WORKK,
     1 CORDEN,NDIM,MF,RTOL,ATOL,NOFIX,IRAN,VALFIX,AB,ierrmod,GAMLAM,
     2 NGRID,NACTVE,PYJGX,DENSTOR,CORDLAST)
```

This internally does:

1. `emint(..., ijob=1)` - IPM + filter (1e-3) + QR + IPM
2. `emint(..., ijob=0)` - IPM only
3. Custom 1e-100 filtering for BestDose

### Current Rust (src/bestdose/mod.rs):

```rust
pub fn calculate_posterior(...) -> Result<(Theta, Weights)> {
    let psi = calculate_psi(...)?;
    let (posterior_weights, _) = burke(&psi)?;  // ❌ Only ONE burke call!

    // Filter by 1e-100 threshold
    let keep: Vec<usize> = posterior_weights.iter()
        .filter(|w| *w > threshold)
        .collect();

    // NO QR decomposition!
    // NO second burke call!

    Ok((filtered_theta, normalized_weights))
}
```

**Problem**: Missing QR decomposition AND second burke call!

## Correct Rust BestDose Implementation

Should follow NPAG's condensation pattern:

```rust
pub fn calculate_posterior(...) -> Result<(Theta, Weights)> {
    // Step 1: Calculate psi
    let mut psi = calculate_psi(...)?;

    // Step 2: First burke call (like emint ijob=1 start)
    let (posterior_weights, _) = burke(&psi)?;

    // Step 3: Filter by 1e-100 threshold (NPAGFULL11 specific)
    let max_weight = posterior_weights.iter().max();
    let threshold = max_weight * 1e-100;
    let keep: Vec<usize> = posterior_weights.iter()
        .filter(|w| *w > threshold)
        .collect();

    // Apply filtering
    let mut filtered_theta = prior_theta.clone();
    filtered_theta.filter_indices(&keep);
    psi.filter_column_indices(&keep);

    // Step 4: QR decomposition (like emint ijob=1 middle)
    let (r, perm) = qr::qrd(&psi)?;
    let keep_qr: Vec<usize> = /* QR filtering with 1e-8 threshold */;

    filtered_theta.filter_indices(&keep_qr);
    psi.filter_column_indices(&keep_qr);

    // Step 5: Second burke call (like emint ijob=0 or emint ijob=1 end)
    let (final_weights, _) = burke(&psi)?;

    Ok((filtered_theta, final_weights))
}
```

## Summary

| Stage            | Fortran emint(ijob=1) | Fortran emint(ijob=0) | Rust burke | Rust NPAG         |
| ---------------- | --------------------- | --------------------- | ---------- | ----------------- |
| IPM optimization | ✅                    | ✅                    | ✅         | ✅ (evaluation)   |
| Lambda filtering | ✅ (1e-3)             | ❌                    | ❌         | ✅ (condensation) |
| QR decomposition | ✅                    | ❌                    | ❌         | ✅ (condensation) |
| Second IPM       | ✅                    | ❌                    | ❌         | ✅ (condensation) |

**Conclusion**:

- `burke` = `emint` with `ijob=0` (IPM only)
- Full NPAG cycle (evaluation + condensation) = Two `emint` calls (ijob=1 then ijob=0)
- BestDose `calculate_posterior` should follow NPAG's condensation pattern to properly implement NPAGFULL11
