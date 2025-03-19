use std::process::abort;

use crate::structs::psi::Psi;
use anyhow::{bail, Context};
use faer::linalg::solvers::Llt;
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Col, Mat, Row};

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
    let mut psi = psi.matrix().to_owned();

    // Ensure all entries are finite and make them non-negative.
    psi.row_iter_mut().try_for_each(|row| {
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

    let psi_clone = psi.clone();

    // Let psi be of shape (n_sub, n_point)
    let (n_sub, n_point) = psi.shape();

    // Create unit vectors:
    // ecol: ones vector of length n_point (used for sums over points)
    // erow: ones row of length n_sub (used for sums over subproblems)
    let ecol: Col<f64> = Col::from_fn(n_point, |_| 1.0);
    let erow: Row<f64> = Row::from_fn(n_sub, |_| 1.0);

    // Compute plam = psi · ecol. This gives a column vector of length n_sub.
    let mut plam: Col<f64> = psi.clone() * &ecol;
    let eps: f64 = 1e-8;
    let mut sig: f64 = 0.0;

    // Initialize lam (the variable we optimize) as a column vector of ones (length n_point).
    let mut lam = ecol.clone();

    // w = 1 ./ plam, elementwise.
    let mut w: Col<f64> = Col::from_fn(plam.nrows(), |i| 1.0 / plam.get(i));

    // ptw = ψᵀ · w, which will be a vector of length n_point.
    let mut ptw: Col<f64> = psi_clone.clone().transpose() * &w;

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

        let mut psi_inner = psi.clone();
        psi_inner.col_iter_mut().enumerate().for_each(|(i, col)| {
            col.iter_mut().for_each(|x| *x *= inner.get(i));
        });

        // for (mut col_vec, &inner_val) in psi_inner.axis_iter_mut(Axis(1)).zip(inner.iter()) {
        //     col_vec *= inner_val;
        // }
        // Build a diagonal matrix from w_plam.
        let diag_w_plam = Mat::from_fn(w_plam.nrows(), w_plam.nrows(), |i, j| {
            if i == j {
                *w_plam.get(i)
            } else {
                0.0
            }
        });

        let h = psi_inner * &psi.transpose() + diag_w_plam.to_owned();

        let uph = match h.llt(faer::Side::Lower) {
            Ok(llt) => llt,
            Err(_) => {
                bail!("Error during Cholesky decomposition")
            }
        };
        let uph = uph.L().transpose().to_owned();

        // smuyinv = smu * (ecol ./ y)
        let smuyinv: Col<f64> = Col::from_fn(ecol.nrows(), |i| smu * (&ecol[i] / y[i]));

        // let smuyinv = smu * (&ecol / &y);
        // rhsdw = (erow ./ w) - (psi · smuyinv)
        let psi_dot_muyinv: Col<f64> = psi.clone() * &smuyinv;

        let rhsdw: Row<f64> = Row::from_fn(ecol.nrows(), |i| &erow[i] / w[i] - psi_dot_muyinv[i]);

        //let rhsdw = (&erow / &w) - psi * &smuyinv;
        // Reshape rhsdw into a column vector.
        let mut dw = Mat::from_fn(rhsdw.ncols(), 1, |i, _j| *rhsdw.get(i));

        // let a = rhsdw
        //     .into_shape((n_sub, 1))
        //     .context("Failed to reshape rhsdw")?;

        // Solve the triangular systems:
        pprint(&uph, "uph");

        pprint(&dw, "a");

        solve_upper_triangular_in_place(uph.as_ref(), dw.as_mut(), faer::Par::Seq);
        pprint(&dw, "x");
        abort();
        solve_upper_triangular_in_place(uph.as_ref(), dw.as_mut(), faer::Par::Seq);
        //pprint(&dw_aux, "dw_aux");

        // Extract dw (a column vector) from the solution.
        let dw = dw.col(0);
        // dbg!(&dw);
        // let dw = dw_aux.column(0);
        // Compute dy = - (ψᵀ · dw)
        let dy = -psi.clone().transpose() * &dw;

        // dlam = smuyinv - lam - (inner .* dy)
        let inner_dot_dy = inner.transpose() * &dy;
        let dlam: Row<f64> = Row::from_fn(ecol.nrows(), |i| &smuyinv[i] - lam[i] - inner_dot_dy);
        // let dlam = &smuyinv - &lam - inner.transpose() * &dy;

        // Compute the primal step length alfpri.
        let ratio_dlam_lam = Row::from_fn(lam.nrows(), |i| dlam[i] / lam[i]);
        //let ratio_dlam_lam = &dlam / &lam;
        let min_ratio_dlam = ratio_dlam_lam.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfpri: f64 = -1.0 / min_ratio_dlam.min(-0.5);
        alfpri = (0.99995 * alfpri).min(1.0);

        // Compute the dual step length alfdual.
        let ratio_dy_y = Row::from_fn(y.nrows(), |i| dy[i] / y[i]);
        // let ratio_dy_y = &dy / &y;
        let min_ratio_dy = ratio_dy_y.iter().cloned().fold(f64::INFINITY, f64::min);
        let ratio_dw_w = Row::from_fn(dw.nrows(), |i| dw[i] / w[i]);
        //let ratio_dw_w = &dw / &w;
        let min_ratio_dw = ratio_dw_w.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfdual = -1.0 / min_ratio_dy.min(-0.5);
        alfdual = alfdual.min(-1.0 / min_ratio_dw.min(-0.5));
        alfdual = (0.99995 * alfdual as f64).min(1.0);

        // Update the iterates.
        lam = lam + alfpri * dlam.transpose();
        w = w + alfdual * &dw;
        y = y + alfdual * &dy;

        mu = lam.transpose() * &y / n_point as f64;
        plam = &psi * &lam;

        // mu = lam.dot(&y) / n_point as f64;
        // plam = psi.dot(&lam);
        r = Col::from_fn(n_sub, |i| erow.get(i) - w.get(i) * plam.get(i));
        ptw = ptw - alfdual * dy;

        norm_r = r.norm_max();
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

fn pprint(x: &Mat<f64>, name: &str) {
    println!("Matrix: {}", name);
    x.row_iter().for_each(|row| {
        println!("{:?}", row);
    });
}
