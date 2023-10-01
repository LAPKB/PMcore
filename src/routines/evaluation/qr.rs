use faer::{Faer, IntoFaer, IntoNdarray};
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis};

/// Calculates the R matrix and column permutation of the QR decomposition for a row-normalized matrix.
///
/// This function takes an input matrix `x` of type `Array2<f64>`, performs row normalization,
/// and then computes the QR decomposition. It returns two values:
///
/// # Arguments
///
/// * `x` - An Array2<f64> matrix representing the input data.
///
/// # Returns
///
/// A tuple containing:
///
/// * `r` - An Array2<f64> representing the R matrix of the QR decomposition.
/// * `perm` - A Vec<usize> containing the column permutation indices applied during the QR decomposition.
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// let x = Array2::<f64>::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
/// let (r, perm) = calculate_r(&x);
/// ```
///
/// In this example, `r` will contain the R matrix, and `perm` will hold the column permutation indices.
///
/// Note: This function assumes that the input matrix `x` is non-empty and has consistent dimensions.
/// ```
///
/// ```
pub fn calculate_r(x: &Array2<f64>) -> (Array2<f64>, Vec<usize>) {
    // TODO: we need more testing but this code seems not to be needed
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
    let mut n_x = x.clone();
    n_x.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| row /= row.sum());
    let mat_x = n_x.view().into_faer();
    let qr = mat_x.col_piv_qr();
    let r_mat = qr.compute_r();
    let (forward, _inverse) = qr.col_permutation().into_arrays();
    let r = r_mat.as_ref().into_ndarray().to_owned();
    let perm = Vec::from(forward);
    (r, perm)
}
