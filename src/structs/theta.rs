use std::fmt::Debug;

use faer::Mat;
use faer_ext::IntoFaer;
use ndarray::{Array2, ArrayView2};

/// [Theta] is a structure that holds the support points
/// These represent the joint population parameter distribution
#[derive(Clone, PartialEq)]
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

    /// Modify the [Theta::matrix] to only include the rows specified by `indices`
    pub(crate) fn filter_indices(&mut self, indices: &[usize]) {
        let matrix = self.matrix.to_owned();

        let new = Mat::from_fn(indices.len(), matrix.ncols(), |r, c| {
            *matrix.get(indices[r], c)
        });

        self.matrix = new;
    }

    /// Forcibly add a support point to the matrix
    pub(crate) fn add_point(&mut self, spp: Vec<f64>) {
        self.matrix
            .resize_with(self.matrix.nrows() + 1, self.matrix.ncols(), |_, i| spp[i]);
    }

    /// Suggest a new support point to add to the matrix
    /// The point is only added if it is at least `min_dist` away from all existing support points
    /// and within the limits specified by `limits`
    pub(crate) fn suggest_point(&mut self, spp: Vec<f64>, min_dist: f64, limits: &[(f64, f64)]) {
        if self.check_point(&spp, min_dist, limits) {
            self.add_point(spp);
        }
    }

    /// Check if a point is at least `min_dist` away from all existing support points
    pub(crate) fn check_point(&self, spp: &Vec<f64>, min_dist: f64, limits: &[(f64, f64)]) -> bool {
        if self.matrix.nrows() == 0 {
            return true;
        }

        for row_idx in 0..self.matrix.nrows() {
            let mut squared_dist = 0.0;
            for (i, val) in spp.iter().enumerate() {
                // Normalized squared difference for this dimension
                let normalized_diff =
                    (val - self.matrix.get(row_idx, i)) / (limits[i].1 - limits[i].0);
                squared_dist += normalized_diff * normalized_diff;
            }
            let dist = squared_dist.sqrt();
            if dist <= min_dist {
                return false; // This point is too close to an existing point
            }
        }
        true // Point is sufficiently distant from all existing points
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

impl Debug for Theta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Write nspp and nsub
        println!();
        writeln!(f, "Theta contains {} support points\n", self.nspp())?;
        // Write the matrix
        self.matrix.row_iter().enumerate().for_each(|(index, row)| {
            write!(f, "{index}\t{:?}\n", row).unwrap();
        });
        Ok(())
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

        theta.filter_indices(&[0, 3]);

        // Expected result is a 2x2 matrix with filtered rows
        let expected = mat![[1.0, 2.0], [7.0, 8.0]];

        assert_eq!(theta.matrix, expected);
    }

    #[test]
    fn test_add_point() {
        let matrix = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let mut theta = Theta::from(matrix);

        theta.add_point(vec![7.0, 8.0]);

        let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        assert_eq!(theta.matrix, expected);
    }
}
