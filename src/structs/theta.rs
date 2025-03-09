use faer::Mat;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::{Array2, ArrayView2};

#[derive(Debug, Clone, PartialEq)]
pub struct Theta {
    matrix: Mat<f64>,
}

impl Theta {
    pub fn new() -> Self {
        Theta { matrix: Mat::new() }
    }

    /// Get the matrix containing parameter values
    ///
    /// The matrix is a 2D array where each row represents a support point, and each column a parameter
    pub fn matrix(&self) -> &Mat<f64> {
        &self.matrix
    }

    /// Get the number of support points, equal to the number of rows in the matrix
    pub fn nspp(&self) -> usize {
        self.matrix.nrows()
    }

    pub fn filter_indices(&mut self, indices: &[usize]) {
        let matrix = self.matrix.to_owned();

        let new = Mat::from_fn(matrix.nrows(), matrix.ncols(), |r, c| {
            *matrix.get(indices[r], c)
        });

        self.matrix = new;
    }
}

impl From<Array2<f64>> for Theta {
    fn from(array: Array2<f64>) -> Self {
        let matrix = array.view().into_faer().to_owned();
        Theta { matrix }
    }
}

impl From<Mat<f64>> for Theta {
    fn from(matrix: Mat<f64>) -> Self {
        Theta { matrix }
    }
}

impl From<ArrayView2<'_, f64>> for Theta {
    fn from(array_view: ArrayView2<'_, f64>) -> Self {
        let matrix = array_view.into_faer().to_owned();
        Theta { matrix }
    }
}

impl From<&Array2<f64>> for Theta {
    fn from(array: &Array2<f64>) -> Self {
        let matrix = array.view().into_faer().to_owned();
        Theta { matrix }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_filter_indices() {
        // Create a 4x2 matrix with recognizable values
        let matrix = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let mut theta = Theta::from(matrix);

        // Reorder to have rows 2, 0, 3, 1
        theta.filter_indices(&[1, 4]);

        // Expected result is a 4x2 matrix with the reordered rows
        let expected = mat![[1.0, 2.0], [7.0, 8.0]];

        assert_eq!(theta.matrix, expected);
    }
}
