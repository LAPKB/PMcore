use faer::{Faer, Mat};
use ndarray::Array2;

pub fn qr(x: &Array2<f64>) -> (Array2<f64>, Vec<usize>) {
    // TODO: we need more testing but this code seems to be not needed
    // if n_psi.ncols() > n_psi.nrows() {
    //     let nrows = n_psi.nrows();
    //     let ncols = n_psi.ncols();

    //     let diff = ncols - nrows;
    //     let zeros = Array2::<f64>::zeros((diff, ncols));
    //     let mut new_n_psi = Array2::<f64>::zeros((nrows + diff, ncols));
    //     new_n_psi.slice_mut(s![..nrows, ..]).assign(&n_psi);
    //     new_n_psi.slice_mut(s![nrows.., ..]).assign(&zeros);
    //     n_psi = new_n_psi;
    //     log::info!(
    //         "Cycle: {}. nspp>nsub. n_psi matrix has been expanded.",
    //         cycle
    //     );
    // }
    let x = Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[[i, j]]);
    let qr = x.col_piv_qr();
    let r_mat = qr.compute_r();
    let (forward, _inverse) = qr.col_permutation().into_arrays();
    //TODO: migrate all my matrix operations to faer or look for faer's own conversion implementation
    // https://github.com/sarah-ek/faer-rs/blob/main/faer/examples/conversions.rs
    // r_mat.as_ref().into_ndarray();
    let mut r: Array2<f64> = Array2::zeros((r_mat.nrows(), r_mat.ncols()));
    for i in 0..r.nrows() {
        for j in 0..r.ncols() {
            r[[i, j]] = if i <= j { r_mat.read(i, j) } else { 0.0 }
        }
    }
    let perm = Vec::from(forward);
    (r, perm)
}
