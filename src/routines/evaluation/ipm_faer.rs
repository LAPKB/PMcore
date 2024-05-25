use faer::{mat, scale, linalg::solvers::SpSolver, unzipped, zipped, Mat};
use faer_ext::*;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use ndarray_stats::QuantileExt;
use std::error;
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
) -> Result<(OneDimArray, f64), Box<dyn error::Error>> {
    let (row, col) = psi.dim();
    if psi.min().unwrap() < &0.0 {
        return Err("PSI contains negative elements".into());
    }
    let psi: Mat<f64> = psi.view().into_faer().to_owned();
    let ecol = Mat::from_fn(col, 1, |_, _| 1.0);
    let mut plam = &psi * &ecol; //row x 1
    let eps = 1e-8;
    let mut sig = 0.0;
    let erow = Mat::from_fn(row, 1, |_, _| 1.0);
    let mut lam = ecol.clone();
    let mut w = plam.clone();
    w = zipped!(w.as_ref()).map(|unzipped!(w_i)| 1.0 / *w_i);

    let psi_t = psi.transpose();
    let mut ptw = psi_t * &w;
    let mut max_ptw = f64::NEG_INFINITY;
    for i in 0..ptw.nrows() {
        if *ptw.get(i, 0) > max_ptw {
            max_ptw = *ptw.get(i, 0)
        }
    }
    let shrink = 2.0 * max_ptw;
    lam = scale(shrink) * lam;
    plam = scale(shrink) * plam;

    w = zipped!(w.as_ref()).map(|unzipped!(w_i)| *w_i / shrink);
    ptw = zipped!(ptw.as_ref()).map(|unzipped!(ptw_i)| *ptw_i / shrink);
    let mut y = &ecol - &ptw;
    let mut r =
        &erow - zipped!(w.as_ref(), plam.as_ref()).map(|unzipped!(w_i, plam_i)| *w_i * *plam_i);
    let mut norm_r = r.norm_max();
    let mut sum_log_plam = 0.0;
    let mut sum_log_w = 0.0;
    for i in 0..plam.nrows() {
        sum_log_plam += plam.get(i, 0).ln();
    }
    for i in 0..w.nrows() {
        sum_log_w += w.get(i, 0).ln();
    }
    let mut gap = (sum_log_w + sum_log_plam).abs() / (1. + sum_log_plam);
    let mut mu = (lam.transpose() * &y).get(0, 0) / col as f64;

    while mu > eps || norm_r > eps || gap > eps {
        let smu = sig * mu;
        let inner = zipped!(lam.as_ref(), y.as_ref()).map(|unzipped!(lam_i, y_i)| *lam_i / *y_i);
        let w_plam =
            zipped!(plam.as_ref(), w.as_ref()).map(|unzipped!(plam_i, w_i)| *plam_i / *w_i);
        let h = (&psi * inner.as_ref().col(0).column_vector_as_diagonal()) * &psi.transpose();
        let mut aux: Mat<f64> = Mat::zeros(row, row);
        for i in 0..row {
            let diag = aux.get_mut(i, i);
            *diag = *w_plam.get(i, 0);
        }
        let h = h + aux;
        let uph = h.cholesky(faer::Side::Lower).unwrap();
        let smuyinv = scale(smu)
            * zipped!(ecol.as_ref(), y.as_ref()).map(|unzipped!(ecol_i, y_i)| *ecol_i / *y_i);
        let rhsdw = zipped!(erow.as_ref(), w.as_ref()).map(|unzipped!(erow_i, w_i)| *erow_i / *w_i)
            - &psi * &smuyinv;
        let dw = uph.solve(&rhsdw);

        let dy = -psi.transpose() * &dw;

        let dlam = smuyinv
            - &lam
            - zipped!(inner.as_ref(), dy.as_ref()).map(|unzipped!(inner_i, dy_i)| *inner_i * *dy_i);
        let mut alfpri = -1.
            / (min(zipped!(dlam.as_ref(), lam.as_ref())
                .map(|unzipped!(dlam_i, lam_i)| *dlam_i / *lam_i))
            .min(-0.5));
        alfpri = (0.99995 * alfpri).min(1.0);
        let mut alfdual = -1.
            / (min(zipped!(dy.as_ref(), y.as_ref()).map(|unzipped!(dy_i, y_i)| *dy_i / *y_i))
                .min(-0.5));
        alfdual = alfdual.min(
            -1. / (min(zipped!(dw.as_ref(), w.as_ref()).map(|unzipped!(dw_i, w_i)| *dw_i / *w_i))
                .min(-0.5)),
        );
        alfdual = (0.99995 * alfdual).min(1.0);
        lam += scale(alfpri) * dlam;
        w += scale(alfdual) * &dw;
        y += scale(alfdual) * &dy;
        mu = (lam.transpose() * &y).get(0, 0) / col as f64;
        plam = &psi * &lam;
        r = &erow - zipped!(w.as_ref(), plam.as_ref()).map(|unzipped!(w_i, plam_i)| *w_i * *plam_i);
        ptw -= scale(alfdual) * dy;
        norm_r = r.norm_max();
        let mut sum_log_plam = 0.0;
        let mut sum_log_w = 0.0;

        for i in 0..plam.nrows() {
            sum_log_plam += plam.get(i, 0).ln();
        }
        for i in 0..w.nrows() {
            sum_log_w += w.get(i, 0).ln();
        }
        gap = (sum_log_w + sum_log_plam).abs() / (1. + sum_log_plam);
        if mu < eps && norm_r > eps {
            sig = 1.0;
        } else {
            sig = max(mat![[
                (1. - alfpri).powi(2),
                (1. - alfdual).powi(2),
                (norm_r - mu) / (norm_r + 100. * mu)
            ]])
            .min(0.3);
        }
    }
    lam = scale(1. / row as f64) * lam;
    let aux = psi * &lam;
    let mut obj = 0.0;

    for i in 0..aux.nrows() {
        obj += aux.get(i, 0).ln();
    }
    Ok((lam.as_ref().into_ndarray().column(0).to_owned(), obj))
}
fn min(mat: Mat<f64>) -> f64 {
    let mut min = f64::INFINITY;
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            if *mat.get(i, j) < min {
                min = *mat.get(i, j)
            }
        }
    }
    min
}
fn max(mat: Mat<f64>) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            if *mat.get(i, j) > max {
                max = *mat.get(i, j)
            }
        }
    }
    max
}
