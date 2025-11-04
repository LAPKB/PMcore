use crate::structs::psi::Psi;
use anyhow::{bail, Result};
use faer::linalg::solvers::ColPivQr;
use faer::Mat;

/// Perform a QR decomposition on the Psi matrix
///
/// Normalizes each row of the matrix to sum to 1 before decomposition.
/// Returns the R matrix from QR decomposition and the column permutation vector.
///
/// # Arguments
/// * `psi` - The Psi matrix to decompose
///
/// # Returns
/// * Tuple containing the R matrix (as [faer::Mat<f64>]) and permutation vector (as [Vec<usize>])
/// * Error if any row in the matrix sums to zero
pub fn qrd(psi: &Psi) -> Result<(Mat<f64>, Vec<usize>)> {
    let mut mat = psi.matrix().to_owned();

    // Normalize the rows to sum to 1
    for (index, row) in mat.row_iter_mut().enumerate() {
        let row_sum: f64 = row.as_ref().iter().sum();

        // Check if the row sum is zero
        if row_sum.abs() == 0.0 {
            bail!("In psi, the row with index {} sums to zero", index);
        }
        row.iter_mut().for_each(|x| *x /= row_sum);
    }

    // Perform column pivoted QR decomposition
    let qr: ColPivQr<f64> = mat.col_piv_qr();

    // Extract the R matrix
    let r_mat: faer::Mat<f64> = qr.R().to_owned();

    // Get the permutation information
    let perm = qr.P().arrays().0.to_vec();
    Ok((r_mat, perm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        // Create a 2x2 identity matrix
        let mat: Mat<f64> = Mat::identity(10, 10);
        let psi = Psi::from(mat);

        // Perform the QR decomposition
        let (r_mat, perm) = qrd(&psi).unwrap();

        // Check that R is an identity matrix
        let expected_r_mat: Mat<f64> = Mat::identity(10, 10);
        assert_eq!(r_mat, expected_r_mat);

        // Check that the permutation is the identity
        assert_eq!(perm, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_with_zero_row_sum() {
        // Create a test matrix with a row that sums to zero
        let mat = Mat::from_fn(2, 2, |i, j| {
            if i == 0 && j == 0 {
                1.0
            } else if i == 0 && j == 1 {
                2.0
            } else {
                0.0 // Row that sums to zero for i == 1
            }
        });
        let psi = Psi::from(mat);

        // Perform the QR decomposition
        let result = qrd(&psi);

        // Confirm that the function returns an error
        assert!(result.is_err(), "Expected an error due to zero row sum");
    }

    #[test]
    fn test_empty_matrix() {
        // Create an empty Psi
        let mat = Mat::<f64>::new();
        let psi = Psi::from(mat);

        // Should not panic
        let (r_mat, perm) = qrd(&psi).unwrap();

        // Empty matrix should produce empty results
        assert_eq!(r_mat.nrows(), 0);
        assert_eq!(r_mat.ncols(), 0);
        assert_eq!(perm.len(), 0);
    }
}
