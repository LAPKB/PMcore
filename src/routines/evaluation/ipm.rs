use std::error;

use linfa_linalg::{cholesky::Cholesky, triangular::SolveTriangular};
use ndarray::{array, Array, Array2, ArrayBase, Dim, OwnedRepr};
use ndarray_stats::{DeviationExt, QuantileExt};
type OneDimArray = ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;

/// Apply the Burke's Interior Point Method (IPM) to solve a specific optimization problem.
///
/// The Burke's IPM is an iterative optimization technique used for solving convex optimization
/// problems. It is applied to a matrix `psi`, iteratively updating variables and calculating
/// an objective function until convergence.
///
/// # Arguments
///
/// * `psi` - A reference to a 2D Array representing the input matrix for optimization.
///
/// # Returns
///
/// A `Result` containing a tuple with two elements:
///
/// * `lam` - An Array1<f64> representing the solution of the optimization problem.
/// * `obj` - A f64 value representing the objective function value at the solution.
///
/// # Errors
///
/// This function returns an error if any of the optimization steps encounter issues. The error
/// type is a boxed dynamic error (`Box<dyn error::Error>`).
///
/// # Example
///
/// ```
/// use your_module::{burke, OneDimArray};
/// use ndarray::{Array2, Array1};
///
/// // Define your input matrix 'psi' here.
///
/// // Solve the optimization problem using Burke's IPM.
/// let result = burke(&psi);
///
/// match result {
///     Ok((lam, obj)) => {
///         // Successfully solved the optimization problem using Burke's IPM.
///         // 'lam' contains the solution, and 'obj' contains the objective function value.
///     },
///     Err(err) => {
///         // Handle the error if optimization fails.
///         eprintln!("Error: {:?}", err);
///     },
/// }
/// ```
///
/// Note: This function applies the Interior Point Method (IPM) to iteratively update variables
/// until convergence, solving the convex optimization problem.
///
pub fn burke(
    psi: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> Result<(OneDimArray, f64), Box<dyn error::Error>> {
    let psi = psi.mapv(|x| x.abs());
    let (row, col) = psi.dim();
    // if row>col {
    //     return Err("The matrix PSI has row>col".into());
    // }
    if psi.min().unwrap() < &0.0 {
        return Err("PSI contains negative elements".into());
    }
    let ecol: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::ones(col);
    let mut plam = psi.dot(&ecol);
    // if plam.min().unwrap() <= &1e-15 {
    //     return Err("The vector psi*e has a non-positive entry".into());
    // }
    let eps = 1e-8;
    let mut sig = 0.;
    let erow: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::ones(row);
    let mut lam = ecol.clone();
    let mut w = 1. / &plam;
    let mut ptw = psi.t().dot(&w);
    let shrink = 2. * *ptw.max().unwrap();
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
        // log::info!("IPM cyle");
        let smu = sig * mu;
        let inner = &lam / &y; //divide(&lam, &y);
        let w_plam = &plam / &w; //divide(&plam, &w);
        let h = psi.dot(&Array2::from_diag(&inner)).dot(&psi.t()) + Array2::from_diag(&w_plam);
        let uph = h.cholesky()?;
        let uph = uph.t();
        let smuyinv = smu * (&ecol / &y);
        let rhsdw = &erow / &w - (psi.dot(&smuyinv));
        let a = rhsdw.clone().into_shape((rhsdw.len(), 1))?;
        //todo: cleanup this aux variable
        // //dbg!(uph.t().is_triangular(linfa_linalg::triangular::UPLO::Upper));
        // uph.solve_into(rhsdw);
        let x = uph
            .t()
            .solve_triangular(&a, linfa_linalg::triangular::UPLO::Lower)?;
        let dw_aux = uph.solve_triangular(&x, linfa_linalg::triangular::UPLO::Upper)?;
        let dw = dw_aux.column(0);
        let dy = -psi.t().dot(&dw);
        let dlam = smuyinv - &lam - inner * &dy;
        let mut alfpri = -1. / ((&dlam / &lam).min().unwrap().min(-0.5));
        alfpri = (0.99995 * alfpri).min(1.0);
        let mut alfdual = -1. / ((&dy / &y).min().unwrap().min(-0.5));
        alfdual = alfdual.min(-1. / (&dw / &w).min().unwrap().min(-0.5));
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
            .max()
            .unwrap()
            .min(0.3);
        }
    }
    lam /= row as f64;
    let obj = psi.dot(&lam).mapv(|x| x.ln()).sum();
    lam = &lam / lam.sum();
    Ok((lam, obj))
}

/// Computes the infinity norm (or maximum norm) of a 1-dimensional array.
///
/// The infinity norm of a 1-dimensional array is defined as the maximum absolute value of its elements.
///
/// # Arguments
///
/// * `a` - A 1-dimensional array (ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>) for which the infinity
///         norm is computed.
///
/// # Returns
///
/// A f64 value representing the infinity norm of the input array.
///
/// # Example
///
/// ```
/// use your_module::norm_inf;
/// use ndarray::{Array1, array};
///
/// // Define your 1-dimensional array 'a' here.
///
/// // Calculate the infinity norm.
/// let infinity_norm = norm_inf(a);
/// ```
///
/// In this example, `infinity_norm` will contain the maximum absolute value of the elements in array 'a'.
///
/// Note: This function calculates the infinity norm of a 1-dimensional array.
///
fn norm_inf(a: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>) -> f64 {
    let zeros: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::zeros(a.len());
    a.linf_dist(&zeros).unwrap()
}
