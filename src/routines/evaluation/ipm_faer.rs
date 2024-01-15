use faer::{Faer, IntoFaer, IntoNdarray};
use faer_core::{matrix_ops, Mat};
use std::{borrow::BorrowMut, error, ops::Mul};

use linfa_linalg::{cholesky::Cholesky, triangular::SolveTriangular};
use ndarray::{array, Array, Array2, ArrayBase, Dim, OwnedRepr};
use ndarray_stats::{DeviationExt, QuantileExt};
type OneDimArray = ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;

pub fn burke(
    psi: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> Result<(OneDimArray, f64), Box<dyn error::Error>> {
    let (row, col) = psi.dim();
    if psi.min().unwrap() < &0.0 {
        return Err("PSI contains negative elements".into());
    }
    let psi = psi.view().into_faer();
    let ecol = Mat::from_fn(col, 1, |_, _| 1.0);
    let mut plam = psi * ecol; //row x 1
    let eps = 1e-8;
    let mut sig = 0.0;
    let erow = Mat::from_fn(row, 1, |_, _| 1.0);
    let mut lam = ecol;
    let mut w = plam.clone();
    for i in 0..w.nrows() {
        for j in 0..w.ncols() {
            // w is row x 1, this for loop should not be needed
            let mut w_aux = w.get_mut(i, j);
            *w_aux = 1.0 / *w_aux;
        }
    }
    let psi_t = Mat::from_fn(psi.ncols(), psi.nrows(), |i, j| psi[(j, i)]);
    let mut ptw = psi_t * w;
    matrix_ops::Ok(())
}
