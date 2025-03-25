use crate::structs::psi::Psi;
use faer::linalg::solvers::ColPivQr;
use faer::Mat;

pub fn calculate_r(psi: &Psi) -> (Mat<f64>, Vec<usize>) {
    // Clone the matrix, as we will modify it
    let mut mat = psi.matrix().clone();

    // Normalize the rows to sum to 1
    mat.row_iter_mut().for_each(|row| {
        let row_sum: f64 = row.as_ref().iter().sum();
        row.iter_mut().for_each(|x| *x /= row_sum);
    });

    // Perform column pivoted QR decomposition
    let qr: ColPivQr<f64> = mat.col_piv_qr();

    // Extract the R matrix
    let r_mat: faer::Mat<f64> = qr.R().to_owned();

    // Get the permutation information
    let perm = qr.P().arrays().0.to_vec();
    (r_mat, perm)
}
