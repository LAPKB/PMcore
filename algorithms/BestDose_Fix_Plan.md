# BestDose calculate_posterior() Fix - Implementation Plan

## Current Status

The current `calculate_posterior()` in Rust BestDose:

- ❌ Calls `burke()` only ONCE
- ❌ Filters by 1e-100 threshold
- ❌ Does NOT do QR decomposition
- ❌ Does NOT recalculate weights after filtering

This is **incorrect** compared to Fortran NPAGFULL11.

## Fortran NPAGFULL11 Pattern

```fortran
! First emint call with ijob=1
call emint(pyjgx,maxsub,corden,maxgrd,nactve,nsub,1,...)
! Returns: keep (number of points after lambda+QR filtering)

nactve = keep  ! Update active count

! Second emint call with ijob=0
call emint(pyjgx,maxsub,corden,maxgrd,nactve,nsub,0,...)
! Recalculates weights on filtered points
```

### What `emint(ijob=1)` does:

1. IPM optimization (burke)
2. Filter by lambda > max \* 1e-3
3. QR decomposition
4. Filter by QR criterion (|R[i,i]| / ||R[:,i]|| >= 1e-8)
5. IPM optimization again on filtered points
6. Returns filtered theta and weights

### What `emint(ijob=0)` does:

1. IPM optimization (burke) only
2. Returns weights

## Rust NPAG Pattern (Already Correct!)

The Rust NPAG implementation already does this correctly:

```rust
// evaluation() - First burke call
fn evaluation(&mut self) -> Result<()> {
    self.psi = calculate_psi(...)?;
    (self.lambda, _) = burke(&self.psi)?;  // Like emint ijob=0
}

// condensation() - Filtering + QR + Second burke call
fn condensation(&mut self) -> Result<()> {
    // Lambda filtering (max/1000)
    let keep = filter_by_lambda(&self.lambda);
    self.theta.filter_indices(&keep);
    self.psi.filter_column_indices(&keep);

    // QR decomposition
    let (r, perm) = qr::qrd(&self.psi)?;
    let keep_qr = filter_by_qr(&r, &perm);
    self.theta.filter_indices(&keep_qr);
    self.psi.filter_column_indices(&keep_qr);

    // Second burke call (like emint ijob=0)
    (self.lambda, self.objf) = burke(&self.psi)?;
}
```

**This is exactly the two emint calls!**

## What BestDose Should Do

`calculate_posterior()` should follow the SAME pattern as NPAG's evaluation+condensation:

```rust
pub fn calculate_posterior(...) -> Result<(Theta, Weights)> {
    // Calculate psi
    let mut psi = calculate_psi(...)?;
    let mut filtered_theta = prior_theta.clone();

    // =================================================
    // FIRST BURKE CALL (like emint ijob=1 start)
    // =================================================
    let (posterior_weights, _) = burke(&psi)?;

    // =================================================
    // LAMBDA FILTERING (NPAGFULL11 specific: 1e-100 threshold)
    // =================================================
    let max_weight = posterior_weights.iter().fold(f64::NEG_INFINITY, |acc, x| x.max(acc));
    let threshold = max_weight * 1e-100;

    let keep_lambda: Vec<usize> = posterior_weights.iter()
        .enumerate()
        .filter(|(_, w)| *w > threshold)
        .map(|(i, _)| i)
        .collect();

    tracing::debug!("Lambda filtering: {} -> {} points",
        prior_theta.nrows(), keep_lambda.len());

    filtered_theta.filter_indices(&keep_lambda);
    psi.filter_column_indices(&keep_lambda);

    // =================================================
    // QR DECOMPOSITION FILTERING (like emint ijob=1 middle)
    // =================================================
    let (r, perm) = qr::qrd(&psi)?;

    let mut keep_qr = Vec::<usize>::new();
    let keep_n = psi.matrix().ncols().min(psi.matrix().nrows());

    for i in 0..keep_n {
        let test = r.col(i).norm_l2();
        let r_diag_val = r.get(i, i);
        let ratio = r_diag_val / test;
        if ratio.abs() >= 1e-8 {
            keep_qr.push(*perm.get(i).unwrap());
        }
    }

    tracing::debug!("QR filtering: {} -> {} points",
        filtered_theta.nrows(), keep_qr.len());

    filtered_theta.filter_indices(&keep_qr);
    psi.filter_column_indices(&keep_qr);

    // =================================================
    // SECOND BURKE CALL (like emint ijob=0 or emint ijob=1 end)
    // =================================================
    let (final_weights, _) = burke(&psi)?;

    tracing::info!("NPAGFULL11 complete: {} -> {} support points",
        prior_theta.nrows(), filtered_theta.nrows());

    Ok((filtered_theta, final_weights))
}
```

## Key Differences from Current Implementation

| Aspect               | Current (Wrong) | Should Be (Correct)   |
| -------------------- | --------------- | --------------------- |
| Burke calls          | 1               | 2                     |
| Lambda filtering     | Yes (1e-100)    | Yes (1e-100)          |
| QR decomposition     | ❌ No           | ✅ Yes                |
| Second burke         | ❌ No           | ✅ Yes                |
| Manual normalization | ✅ Yes          | ❌ No (burke does it) |

## Why This Matters

Without QR decomposition:

- Linearly dependent support points remain
- Can cause numerical instability in optimization
- Different results from Fortran
- Not following the NPAGFULL11 algorithm correctly

## Implementation Status

- [ ] Remove manual weight normalization (burke already normalizes)
- [ ] Add QR decomposition after lambda filtering
- [ ] Add second burke call after QR filtering
- [ ] Update logging to show both filtering stages
- [ ] Test against Fortran to verify same number of points
- [ ] Verify NPAGFULL refinement still works correctly

## Testing Checklist

1. Compare filtered point counts with Fortran NPAGFULL11
2. Verify QR drops linearly dependent points
3. Ensure final weights sum to 1.0
4. Check that NPAGFULL refinement gets correct input
5. Run full BestDose optimization and compare doses
