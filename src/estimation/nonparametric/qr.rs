use crate::estimation::nonparametric::Psi;
use anyhow::{bail, Result};
use faer::linalg::solvers::ColPivQr;
use faer::Mat;

/// Perform a QR decomposition on the Psi matrix.
pub fn qrd(psi: &Psi) -> Result<(Mat<f64>, Vec<usize>)> {
    let mut mat = psi.matrix().to_owned();

    for (index, row) in mat.row_iter_mut().enumerate() {
        let row_sum: f64 = row.as_ref().iter().sum();
        if row_sum.abs() == 0.0 {
            bail!("In psi, the row with index {} sums to zero", index);
        }
        row.iter_mut().for_each(|x| *x /= row_sum);
    }

    let qr: ColPivQr<f64> = mat.col_piv_qr();
    let r_mat: faer::Mat<f64> = qr.R().to_owned();
    let perm = qr.P().arrays().0.to_vec();
    Ok((r_mat, perm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let mat: Mat<f64> = Mat::identity(10, 10);
        let psi = Psi::from(mat);
        let (r_mat, perm) = qrd(&psi).unwrap();

        let expected_r_mat: Mat<f64> = Mat::identity(10, 10);
        assert_eq!(r_mat, expected_r_mat);
        assert_eq!(perm, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}