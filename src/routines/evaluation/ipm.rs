use crate::structs::psi::Psi;
use anyhow::{bail, Context};
use faer::dyn_stack::MemStack;
use faer::linalg::cholesky::llt::factor::cholesky_in_place;
use faer::{Auto, Col, Mat, Row, Spec};
use linfa_linalg::{cholesky::Cholesky, triangular::SolveTriangular};
use ndarray::{array, Array2, Axis};
use ndarray_stats::{DeviationExt, QuantileExt};

/// Applies Burke's Interior Point Method (IPM) to solve a convex optimization problem.
///
/// The objective function to maximize is:
///     f(x) = Σ(log(Σ(ψ_ij * x_j)))   for i = 1 to n_sub
///
/// subject to:
///     1. x_j ≥ 0 for all j = 1 to n_point,
///     2. Σ(x_j) = 1,
///
/// where ψ is an n_sub×n_point matrix with non-negative entries and x is a probability vector.
///
/// # Arguments
///
/// * `psi` - A reference to a Psi structure containing the input matrix.
///
/// # Returns
///
/// On success, returns a tuple `(lam, obj)` where:
///   - `lam` is a faer::Col<f64> containing the computed probability vector,
///   - `obj` is the value of the objective function at the solution.
///
/// # Errors
///
/// This function returns an error if any step in the optimization (e.g. Cholesky factorization)
/// fails.
pub fn burke(psi: &Psi) -> anyhow::Result<(Col<f64>, f64)> {
    // Get the underlying matrix. (Assume psi.matrix() returns an ndarray-compatible matrix.)
    let mut psi = psi.matrix();

    // Ensure all entries are finite and make them non-negative.
    psi.row_iter_mut().try_for_each(|mut row| {
        row.iter_mut().try_for_each(|x| {
            if !x.is_finite() {
                bail!("Input matrix must have finite entries")
            } else {
                // Coerce negatives to non-negative (could alternatively return an error)
                *x = x.abs();
                Ok(())
            }
        })
    })?;

    // Let psi be of shape (n_sub, n_point)
    let (n_sub, n_point) = psi.shape();

    // Create unit vectors:
    // ecol: ones vector of length n_point (used for sums over points)
    // erow: ones row of length n_sub (used for sums over subproblems)
    let ecol: Col<f64> = Col::from_fn(n_point, |_| 1.0);
    let erow: Row<f64> = Row::from_fn(n_sub, |_| 1.0);

    // Compute plam = psi · ecol. This gives a column vector of length n_sub.
    let mut plam: Col<f64> = psi * &ecol;
    let eps: f64 = 1e-8;
    let mut sig: f64 = 0.0;

    // Initialize lam (the variable we optimize) as a column vector of ones (length n_point).
    let mut lam = ecol.clone();

    // w = 1 ./ plam, elementwise.
    let mut w: Col<f64> = Col::from_fn(plam.nrows(), |i| 1.0 / plam.get(i));

    // ptw = ψᵀ · w, which will be a vector of length n_point.
    let mut ptw: Col<f64> = psi.transpose() * &w;

    // Use the maximum entry in ptw for scaling (the "shrink" factor).
    let ptw_max = ptw.iter().fold(f64::NEG_INFINITY, |acc, &x| x.max(acc));
    let shrink = 2.0 * ptw_max;
    lam *= shrink;
    plam *= shrink;
    w /= shrink;
    ptw /= shrink;

    // y = ecol - ptw (a vector of length n_point).
    let mut y: Col<f64> = &ecol - &ptw;
    // r = erow - (w .* plam) (elementwise product; r has length n_sub).
    let mut r: Col<f64> = Col::from_fn(n_sub, |i| erow.get(i) - w.get(i) * plam.get(i));
    let mut norm_r: f64 = r.iter().fold(0.0, |max, &val| max.max(val.abs()));

    // Compute the duality gap.
    let sum_log_plam: f64 = plam.iter().map(|x| x.ln()).sum();
    let sum_log_w: f64 = w.iter().map(|x| x.ln()).sum();
    let mut gap: f64 = (sum_log_w + sum_log_plam).abs() / (1.0 + sum_log_plam);

    // Compute the duality measure mu.
    let mut mu = lam.transpose() * &y / n_point as f64;

    while mu > eps || norm_r > eps || gap > eps {
        let smu = sig * mu;
        // inner = lam ./ y, elementwise.
        let inner = Col::from_fn(lam.nrows(), |i| lam.get(i) / y.get(i));
        // w_plam = plam ./ w, elementwise (length n_sub).
        let w_plam = Col::from_fn(plam.nrows(), |i| plam.get(i) / w.get(i));

        // Scale each column of psi by the corresponding element of 'inner'
        let mut psi_inner: Mat<f64> = psi.clone();
        psi_inner.col_iter_mut().map(|mut col| {
            col.iter_mut()
                .zip(inner.iter())
                .for_each(|(x, &inner_val)| *x *= inner_val);
        });
        // for (mut col_vec, &inner_val) in psi_inner.axis_iter_mut(Axis(1)).zip(inner.iter()) {
        //     col_vec *= inner_val;
        // }
        // Build a diagonal matrix from w_plam.
        let diag_w_plam = Mat::from_fn(w_plam.nrows(), w_plam.ncols(), |i, j| {
            if i == j {
                *w_plam.get(i)
            } else {
                0.0
            }
        });

        // Hessian approximation: h = (psi_inner * psiᵀ) + diag(w_plam)
        let mut h = psi_inner * &psi.transpose() + diag_w_plam.to_owned();

        // Create a buffer for MemStack

        let uph = psi.lblt(faer::Side::Lower);
        let uph = uph.L();
        let uph = uph.transpose();

        // smuyinv = smu * (ecol ./ y)
        let smuyinv = smu * (&ecol / &y);
        // rhsdw = (erow ./ w) - (psi · smuyinv)
        let rhsdw = (&erow / &w) - psi * &smuyinv;
        // Reshape rhsdw into a column vector.
        let a = rhsdw
            .into_shape((n_sub, 1))
            .context("Failed to reshape rhsdw")?;

        // Solve the triangular systems:
        let x = uph
            .transpose()
            .solve_triangular(&a, linfa_linalg::triangular::UPLO::Lower)
            .context("Error solving lower triangular system")?;
        let dw_aux = uph
            .solve_triangular(&x, linfa_linalg::triangular::UPLO::Upper)
            .context("Error solving upper triangular system")?;
        // Extract dw (a column vector) from the solution.
        let dw = dw_aux.column(0);
        // Compute dy = - (ψᵀ · dw)
        let dy = -psi.transpose() * &dw;

        // dlam = smuyinv - lam - (inner .* dy)
        let dlam = &smuyinv - &lam - inner * &dy;

        // Compute the primal step length alfpri.
        let ratio_dlam_lam = &dlam / &lam;
        let min_ratio_dlam = ratio_dlam_lam.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfpri: f64 = -1.0 / min_ratio_dlam.min(-0.5);
        alfpri = (0.99995 * alfpri).min(1.0);

        // Compute the dual step length alfdual.
        let ratio_dy_y = &dy / &y;
        let min_ratio_dy = ratio_dy_y.iter().cloned().fold(f64::INFINITY, f64::min);
        let ratio_dw_w = &dw / &w;
        let min_ratio_dw = ratio_dw_w.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfdual = -1.0 / min_ratio_dy.min(-0.5);
        alfdual = alfdual.min(-1.0 / min_ratio_dw.min(-0.5));
        alfdual = (0.99995 * alfdual as f64).min(1.0);

        // Update the iterates.
        lam = lam + alfpri * dlam;
        w = w + alfdual * &dw;
        y = y + alfdual * &dy;
        mu = lam.dot(&y) / n_point as f64;
        plam = psi.dot(&lam);
        r = Col::from_fn(n_sub, |i| erow.get(i) - w.get(i) * plam.get(i));
        ptw = ptw - alfdual * dy;

        norm_r = norm_inf(&r);
        let sum_log_plam: f64 = plam.iter().map(|x| x.ln()).sum();
        let sum_log_w: f64 = w.iter().map(|x| x.ln()).sum();
        gap = (sum_log_w + sum_log_plam).abs() / (1.0 + sum_log_plam);

        // Adjust sigma.
        if mu < eps && norm_r > eps {
            sig = 1.0;
        } else {
            let candidate1 = (1.0 - alfpri).powi(2);
            let candidate2 = (1.0 - alfdual).powi(2);
            let candidate3 = (norm_r - mu) / (norm_r + 100.0 * mu);
            sig = candidate1.max(candidate2).max(candidate3).min(0.3);
        }
    }
    // Scale lam.
    lam = lam / (n_sub as f64);
    // Compute the objective function value: sum(ln(psi·lam)).
    let obj = (psi * &lam).iter().map(|x| x.ln()).sum();
    // Normalize lam to sum to 1.
    let lam_sum: f64 = lam.iter().sum();
    lam = &lam / lam_sum;

    Ok((lam, obj))
}

/// Computes the infinity norm (maximum absolute value) of a column vector.
fn norm_inf(a: &Col<f64>) -> f64 {
    a.iter().fold(0.0, |max, &val| max.max(val.abs()))
}
