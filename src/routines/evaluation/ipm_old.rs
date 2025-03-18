use anyhow::{bail, Context};
use linfa_linalg::{cholesky::Cholesky, triangular::SolveTriangular};
use ndarray::{array, Array, Array2, ArrayBase, Axis, Dim, OwnedRepr};
use ndarray_stats::{DeviationExt, QuantileExt};

// use crate::logger::trace_memory;
type OneDimArray = ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;

/// Applies the Burke's Interior Point Method (IPM) to solve a specific optimization problem.
///
/// The Burke's IPM is an iterative optimization technique used for solving convex optimization
/// problems. It is applied to a matrix `psi`, iteratively updating variables and calculating
/// an objective function until convergence.
///
/// The objective function to maximize is:
/// f(x) = Σ(log(Σ(ψ_ij * x_j))) for i = 1 to n_sub
///
/// Subject to the constraints:
/// 1. x_j >= 0 for all j = 1 to n_point
/// 2. Σ(x_j) = 1 for j = 1 to n_point
///
/// Where:
/// - ψ is an n_sub x n_point matrix with non-negative entries.
/// - x is a probability vector of length n_point.
///
/// # Arguments
///
/// * `psi` - A reference to a 2D Array representing the input matrix for optimization.
///
/// # Returns
///
/// A `Result` containing a tuple with two elements:
///
/// * `lam` - An `Array1<f64>` representing the solution of the optimization problem.
/// * `obj` - A f64 value representing the objective function value at the solution.
///
/// # Errors
///
/// This function returns an error if any of the optimization steps encounter issues. The error
/// type is a boxed dynamic error (`Box<dyn error::Error>`).
///
/// # Example
///
/// Note: This function applies the Interior Point Method (IPM) to iteratively update variables
/// until convergence, solving the convex optimization problem.
///
pub fn burke(
    psi: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> anyhow::Result<(OneDimArray, f64)> {
    let psi = psi.mapv(|x| x.abs());
    let (row, col) = psi.dim();
    if psi.min()? < &0.0 {
        bail!("Input matrix must have non-negative entries")
    }
    let ecol: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::ones(col);
    let mut plam = psi.dot(&ecol);
    let eps = 1e-8;
    let mut sig = 0.;
    let erow: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::ones(row);

    let mut lam = ecol.clone();

    let mut w = 1. / &plam;

    let mut ptw = psi.t().dot(&w);
    let shrink = 2. * *ptw.max().context("Failed to get max value")?;
    lam *= shrink;
    plam *= shrink;
    w /= shrink;
    ptw /= shrink;

    let mut y = &ecol - &ptw;
    let mut r = &erow - &w * &plam;

    let mut norm_r = norm_inf(r);

    let sum_log_plam = plam.mapv(|x: f64| x.ln()).sum();
    let mut gap = (w.mapv(|x: f64| x.ln()).sum() + sum_log_plam).abs() / (1. + sum_log_plam);
    let mut mu = lam.t().dot(&y) / col as f64;
    while mu > eps || norm_r > eps || gap > eps {
        // log::info!("IPM cycle");
        let smu = sig * mu;
        let inner = &lam / &y;

        let w_plam = &plam / &w;

        let mut psi_inner: Array2<f64> = psi.clone();
        for (mut col, inner_val) in psi_inner.axis_iter_mut(Axis(1)).zip(&inner) {
            col *= *inner_val;
        }

        let h = psi_inner.dot(&psi.t()) + Array2::from_diag(&w_plam);

        let uph = h
            .cholesky()
            .context("Error during Cholesky decomposition")?;

        let uph = uph.t();
        let smuyinv = smu * (&ecol / &y);
        let rhsdw = &erow / &w - (psi.dot(&smuyinv));
        let a = rhsdw.clone().into_shape_with_order((rhsdw.len(), 1))?;

        dbg!(&uph);
        dbg!(&a);

        let x = uph
            .t()
            .solve_triangular(&a, linfa_linalg::triangular::UPLO::Lower)?;

        let dw_aux = uph.solve_triangular(&x, linfa_linalg::triangular::UPLO::Upper)?;

        let dw = dw_aux.column(0);
        let dy = -psi.t().dot(&dw);

        let dlam = smuyinv - &lam - inner * &dy;

        let mut alfpri = -1. / ((&dlam / &lam).min()?.min(-0.5));

        alfpri = (0.99995 * alfpri).min(1.0);

        let mut alfdual = -1. / ((&dy / &y).min()?.min(-0.5));
        alfdual = alfdual.min(-1. / (&dw / &w).min()?.min(-0.5));
        alfdual = (0.99995 * alfdual).min(1.0);
        lam = lam + alfpri * dlam;
        w = w + alfdual * &dw;
        y = y + alfdual * &dy;
        mu = lam.t().dot(&y) / col as f64;
        plam = psi.dot(&lam);
        r = &erow - &w * &plam;
        ptw = ptw - alfdual * dy;

        norm_r = norm_inf(r);

        let sum_log_plam = plam.mapv(|x: f64| x.ln()).sum();
        gap = (w.mapv(|x: f64| x.ln()).sum() + sum_log_plam).abs() / (1. + sum_log_plam);
        if mu < eps && norm_r > eps {
            sig = 1.0;
        } else {
            sig = array![[
                (1. - alfpri).powi(2),
                (1. - alfdual).powi(2),
                (norm_r - mu) / (norm_r + 100. * mu)
            ]]
            .max()?
            .min(0.3);
        }
    }
    lam /= row as f64;
    let obj = psi.dot(&lam).mapv(|x| x.ln()).sum();
    lam = &lam / lam.sum();

    Ok((lam, obj))
}

/// Computes the infinity norm (or maximum norm) of a 1-dimensional array
/// The infinity norm is the maximum, absolute value of its elements
fn norm_inf(a: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>) -> f64 {
    let zeros: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::zeros(a.len());
    a.linf_dist(&zeros).unwrap()
}
