use faer::{Faer, IntoFaer, IntoNdarray};
use faer_core::{inner::DenseOwn, mat, scale, unzipped, zip, zipped, Mat, Matrix};
use std::{borrow::BorrowMut, error, ops::Mul};

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
    // matrix_ops::Ok(())
}
