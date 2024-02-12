use faer::{Faer, IntoFaer, IntoNdarray};
use faer_core::{
    inner::{DenseOwn, DiagRef},
    mat, scale, unzipped, zip, zipped, Mat, Matrix,
};
use std::{borrow::BorrowMut, error, ops::Mul, os::unix::process};

use linfa_linalg::{cholesky::Cholesky, triangular::SolveTriangular};
use ndarray::{array, Array, Array2, ArrayBase, Dim, OwnedRepr};
use ndarray_stats::{DeviationExt, QuantileExt};
type OneDimArray = ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;

pub fn burke(psi: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>) {
    let (row, col) = psi.dim();
    // if psi.min().unwrap() < &0.0 {
    //     return Err("PSI contains negative elements".into());
    // }
    let psi: Matrix<DenseOwn<f64>> = psi.view().into_faer().to_owned();
    let ecol = Mat::from_fn(col, 1, |_, _| 1.0);
    let mut plam = &psi * &ecol; //row x 1
    let eps = 1e-8;
    let mut sig = 0.0;
    let erow = Mat::from_fn(row, 1, |_, _| 1.0);
    let mut lam = ecol.clone();
    let mut w = plam.clone();
    //This loop can be replaced to a zip call
    w = zipped!(w.as_ref()).map(|unzipped!(w_i)| 1.0 / *w_i);
    // for i in 0..w.nrows() {
    //     // for j in 0..w.ncols() {
    //     // w is row x 1, this for loop should not be needed
    //     let mut w_aux = w.get_mut(i, 0);
    //     *w_aux = 1.0 / *w_aux;
    //     // }
    // }

    let psi_t = psi.transpose();
    let mut ptw: Matrix<DenseOwn<f64>> = psi_t * &w;
    dbg!(&ptw);
    let mut max_ptw = f64::NEG_INFINITY;
    for i in 0..ptw.nrows() {
        if *ptw.get(i, 0) > max_ptw {
            max_ptw = *ptw.get(i, 0)
        }
    }
    let shrink = 2.0 * max_ptw;
    lam = scale(shrink) * lam;
    plam = scale(shrink) * plam;
    // w = scale(1.0 / shrink) * w;
    // ptw = scale(1.0 / shrink) * ptw;
    w = zipped!(w.as_ref()).map(|unzipped!(w_i)| *w_i / shrink);
    ptw = zipped!(ptw.as_ref()).map(|unzipped!(ptw_i)| *ptw_i / shrink);
    dbg!(&w);
    dbg!(&plam);
    dbg!(&erow);
    let mut y = &ecol - &ptw;
    let mut r =
        &erow - zipped!(w.as_ref(), plam.as_ref()).map(|unzipped!(w_i, plam_i)| *w_i * *plam_i);
    let mut norm_r = r.norm_max();
    dbg!(&y);
    dbg!(&r);
    dbg!(&norm_r);
    let mut sum_log_plam = 0.0;
    let mut sum_log_w = 0.0;
    dbg!(w.ncols());
    dbg!(w.nrows());
    dbg!(&w);
    for i in 0..plam.nrows() {
        sum_log_plam += plam.get(i, 0).ln();
    }
    for i in 0..w.nrows() {
        sum_log_w += w.get(i, 0).ln();
    }
    let mut gap = (sum_log_w + sum_log_plam).abs() / (1. + sum_log_plam);
    dbg!(sum_log_w);
    dbg!(gap);
    let mut mu = (lam.transpose() * &y).get(0, 0) / col as f64;

    dbg!(&mu);
    while mu > eps || norm_r > eps || gap > eps {
        let smu = sig * mu;
        let inner = zipped!(lam.as_ref(), y.as_ref()).map(|unzipped!(lam_i, y_i)| *lam_i / *y_i);
        dbg!(&inner);
        let w_plam =
            zipped!(plam.as_ref(), w.as_ref()).map(|unzipped!(plam_i, w_i)| *plam_i / *w_i);
        dbg!(&w_plam);
        let h: Matrix<DenseOwn<f64>> =
            (&psi * inner.as_ref().col(0).column_vector_as_diagonal()) * &psi.transpose();
        let mut aux: Matrix<DenseOwn<f64>> = Mat::zeros(row, row);
        for i in 0..row {
            let mut diag = aux.get_mut(i, i);
            *diag = *w_plam.get(i, 0);
        }
        let h = h + aux;
        dbg!(&h);
        std::process::abort();
    }

    // matrix_ops::Ok(())
}
