use std::fmt::Debug;

use faer::Mat;

use crate::prelude::Parameters;

/// [Theta] is a structure that holds the support points
/// These represent the joint population parameter distribution
///
/// Each row represents a support points, and each column a parameter
#[derive(Clone, PartialEq)]
pub struct Theta {
    matrix: Mat<f64>,
    parameters: Parameters,
}

impl Default for Theta {
    fn default() -> Self {
        Theta {
            matrix: Mat::new(),
            parameters: Parameters::new(),
        }
    }
}

impl Theta {
    pub fn new() -> Self {
        Theta::default()
    }

    pub(crate) fn from_parts(matrix: Mat<f64>, parameters: Parameters) -> Self {
        Theta { matrix, parameters }
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

    /// Get the parameter names
    pub fn param_names(&self) -> Vec<String> {
        self.parameters.names()
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
    pub(crate) fn add_point(&mut self, spp: &[f64]) {
        self.matrix
            .resize_with(self.matrix.nrows() + 1, self.matrix.ncols(), |_, i| spp[i]);
    }

    /// Suggest a new support point to add to the matrix
    /// The point is only added if it is at least `min_dist` away from all existing support points
    /// and within the limits specified by `limits`
    pub(crate) fn suggest_point(&mut self, spp: &[f64], min_dist: f64) {
        if self.check_point(spp, min_dist) {
            self.add_point(spp);
        }
    }

    /// Check if a point is at least `min_dist` away from all existing support points
    pub(crate) fn check_point(&self, spp: &[f64], min_dist: f64) -> bool {
        if self.matrix.nrows() == 0 {
            return true;
        }

        let limits = self.parameters.ranges();

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
        writeln!(f, "\nTheta contains {} support points\n", self.nspp())?;

        // Write the parameter names
        for name in self.parameters.names().iter() {
            write!(f, "\t{}", name)?;
        }
        writeln!(f)?;
        // Write the matrix
        self.matrix.row_iter().enumerate().for_each(|(index, row)| {
            write!(f, "{}", index).unwrap();
            for val in row.iter() {
                write!(f, "\t{:.2}", val).unwrap();
            }
            writeln!(f).unwrap();
        });
        Ok(())
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

        let parameters = Parameters::new().add("A", 0.0, 10.0).add("B", 0.0, 10.0);

        let mut theta = Theta::from_parts(matrix, parameters);

        theta.filter_indices(&[0, 3]);

        // Expected result is a 2x2 matrix with filtered rows
        let expected = mat![[1.0, 2.0], [7.0, 8.0]];

        assert_eq!(theta.matrix, expected);
    }

    #[test]
    fn test_add_point() {
        let matrix = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let parameters = Parameters::new().add("A", 0.0, 10.0).add("B", 0.0, 10.0);

        let mut theta = Theta::from_parts(matrix, parameters);

        theta.add_point(&[7.0, 8.0]);

        let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        assert_eq!(theta.matrix, expected);
    }
}
