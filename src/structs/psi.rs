use anyhow::Result;
use faer::Mat;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::{Array2, ArrayView2};
use pharmsol::prelude::simulator::psi;
use pharmsol::Data;
use pharmsol::Equation;
use pharmsol::ErrorModels;

use super::theta::Theta;

/// [Psi] is a structure that holds the likelihood for each subject (row), for each support point (column)
#[derive(Debug, Clone, PartialEq)]
pub struct Psi {
    matrix: Mat<f64>,
}

impl Psi {
    pub fn new() -> Self {
        Psi { matrix: Mat::new() }
    }

    pub fn matrix(&self) -> &Mat<f64> {
        &self.matrix
    }

    pub fn nspp(&self) -> usize {
        self.matrix.nrows()
    }

    pub fn nsub(&self) -> usize {
        self.matrix.ncols()
    }

    /// Modify the [Psi::matrix] to only include the columns specified by `indices`
    pub(crate) fn filter_column_indices(&mut self, indices: &[usize]) {
        let matrix = self.matrix.to_owned();

        let new = Mat::from_fn(matrix.nrows(), indices.len(), |r, c| {
            *matrix.get(r, indices[c])
        });

        self.matrix = new;
    }

    /// Write the matrix to a CSV file
    pub fn write(&self, path: &str) {
        let mut writer = csv::Writer::from_path(path).unwrap();
        for row in self.matrix.row_iter() {
            writer
                .write_record(row.iter().map(|x| x.to_string()))
                .unwrap();
        }
    }
}

impl Default for Psi {
    fn default() -> Self {
        Psi::new()
    }
}

impl From<Array2<f64>> for Psi {
    fn from(array: Array2<f64>) -> Self {
        let matrix = array.view().into_faer().to_owned();
        Psi { matrix }
    }
}

impl From<Mat<f64>> for Psi {
    fn from(matrix: Mat<f64>) -> Self {
        Psi { matrix }
    }
}

impl From<ArrayView2<'_, f64>> for Psi {
    fn from(array_view: ArrayView2<'_, f64>) -> Self {
        let matrix = array_view.into_faer().to_owned();
        Psi { matrix }
    }
}

impl From<&Array2<f64>> for Psi {
    fn from(array: &Array2<f64>) -> Self {
        let matrix = array.view().into_faer().to_owned();
        Psi { matrix }
    }
}

pub fn calculate_psi(
    equation: &impl Equation,
    subjects: &Data,
    theta: &Theta,
    error_models: &ErrorModels,
    progress: bool,
    cache: bool,
) -> Result<Psi> {
    let psi_ndarray = psi(
        equation,
        subjects,
        &theta.matrix().clone().as_ref().into_ndarray().to_owned(),
        error_models,
        progress,
        cache,
    )?;

    Ok(psi_ndarray.view().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_from_array2() {
        // Create a test 2x3 array
        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let psi = Psi::from(array.clone());

        // Check dimensions
        assert_eq!(psi.nspp(), 2);
        assert_eq!(psi.nsub(), 3);

        // Check values by converting back to ndarray and comparing
        let result_array = psi.matrix().as_ref().into_ndarray();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(result_array[[i, j]], array[[i, j]]);
            }
        }
    }

    #[test]
    fn test_from_array2_ref() {
        // Create a test 3x2 array
        let array =
            Array2::from_shape_vec((3, 2), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();

        let psi = Psi::from(&array);

        // Check dimensions
        assert_eq!(psi.nspp(), 3);
        assert_eq!(psi.nsub(), 2);

        // Check values by converting back to ndarray and comparing
        let result_array = psi.matrix().as_ref().into_ndarray();
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(result_array[[i, j]], array[[i, j]]);
            }
        }
    }

    #[test]
    fn test_nspp() {
        // Test with a 4x2 matrix
        let array =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let psi = Psi::from(array);

        assert_eq!(psi.nspp(), 4);
    }

    #[test]
    fn test_nspp_empty() {
        // Test with empty matrix
        let psi = Psi::new();
        assert_eq!(psi.nspp(), 0);
    }

    #[test]
    fn test_nspp_single_row() {
        // Test with 1x3 matrix
        let array = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let psi = Psi::from(array);

        assert_eq!(psi.nspp(), 1);
    }

    #[test]
    fn test_nsub() {
        // Test with a 2x5 matrix
        let array = Array2::from_shape_vec(
            (2, 5),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let psi = Psi::from(array);

        assert_eq!(psi.nsub(), 5);
    }

    #[test]
    fn test_nsub_empty() {
        // Test with empty matrix
        let psi = Psi::new();
        assert_eq!(psi.nsub(), 0);
    }

    #[test]
    fn test_nsub_single_column() {
        // Test with 3x1 matrix
        let array = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let psi = Psi::from(array);

        assert_eq!(psi.nsub(), 1);
    }

    #[test]
    fn test_from_implementations_consistency() {
        // Test that both From implementations produce the same result
        let array = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]).unwrap();

        let psi_from_owned = Psi::from(array.clone());
        let psi_from_ref = Psi::from(&array);

        // Both should have the same dimensions
        assert_eq!(psi_from_owned.nspp(), psi_from_ref.nspp());
        assert_eq!(psi_from_owned.nsub(), psi_from_ref.nsub());

        // And the same values
        let owned_array = psi_from_owned.matrix().as_ref().into_ndarray();
        let ref_array = psi_from_ref.matrix().as_ref().into_ndarray();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(owned_array[[i, j]], ref_array[[i, j]]);
            }
        }
    }
}
