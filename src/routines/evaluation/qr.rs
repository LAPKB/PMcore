// use faer::{FaerMat, IntoFaer, IntoNdarray};
use faer_ext::*;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis};

pub fn calculate_r(x: &Array2<f64>) -> (Array2<f64>, Vec<usize>) {
    let mut n_x = x.clone();
    n_x.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| row /= row.sum());
    let mat_x = n_x.view().into_faer();
    let qr = mat_x.col_piv_qr();
    let r_mat = qr.compute_r();
    let (forward, _inverse) = qr.col_permutation().arrays();
    let r = r_mat.as_ref().into_ndarray().to_owned();
    let perm = Vec::from(forward);
    (r, perm)
}
