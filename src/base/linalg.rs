use faer::{Faer, Mat};
use ndarray::Array2;

pub fn faer_qr_decomp(x: &Array2<f64>) -> (Array2<f64>, Vec<usize>) {
    let x = Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[[i, j]]);
    let qr = x.col_piv_qr();
    let r_mat = qr.compute_r();
    let (forward, _inverse) = qr.col_permutation().into_arrays();
    let mut r: Array2<f64> = Array2::zeros((r_mat.nrows(), r_mat.ncols()));
    //TODO: migrate all my matrix operations to faer or look for faer's own conversion implementation
    // https://github.com/sarah-ek/faer-rs/blob/main/faer/examples/conversions.rs
    for i in 0..r.nrows() {
        for j in 0..r.ncols() {
            r[[i, j]] = if i <= j { r_mat.read(i, j) } else { 0.0 }
        }
    }
    let perm = Vec::from(forward);
    (r, perm)
}
