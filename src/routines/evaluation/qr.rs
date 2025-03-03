use faer_ext::*;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis};

use faer::linalg::solvers::ColPivQr;
use faer::perm::PermRef;
use faer::MatRef;

pub fn calculate_r(x: &Array2<f64>) -> (Array2<f64>, Vec<usize>) {
    // Normalize rows to sum to 1
    let mut n_x = x.clone();
    n_x.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| row /= row.sum());

    // Convert to faer matrix format
    let mat_x: MatRef<'_, f64> = n_x.view().into_faer();

    // Perform column pivoted QR decomposition
    let qr: ColPivQr<f64> = mat_x.col_piv_qr();

    // Extract the R matrix
    let r_mat: faer::Mat<f64> = qr.R().to_owned();
    let r = r_mat.as_ref().into_ndarray().to_owned();

    // Get the permutation information
    let perm: PermRef<'_, usize> = qr.P();
    let perm_vec: Vec<usize> = perm.as_ref().into();
    (r, perm)
}
