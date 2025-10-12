use std::fmt::Debug;

use anyhow::{bail, Result};
use faer::Mat;
use serde::{Deserialize, Serialize};

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

    /// Create a new [Theta] from a matrix and [Parameters]
    ///
    /// It is important that the number of columns in the matrix matches the number of parameters
    /// in the [Parameters] object
    ///
    /// The order of parameters in the [Parameters] object should match the order of columns in the matrix
    pub fn from_parts(matrix: Mat<f64>, parameters: Parameters) -> Result<Self> {
        if matrix.ncols() != parameters.len() {
            bail!(
                "Number of columns in matrix ({}) does not match number of parameters ({})",
                matrix.ncols(),
                parameters.len()
            );
        }

        Ok(Theta { matrix, parameters })
    }

    /// Get the matrix containing parameter values
    ///
    /// The matrix is a 2D array where each row represents a support point, and each column a parameter
    pub fn matrix(&self) -> &Mat<f64> {
        &self.matrix
    }

    /// Get a mutable reference to the matrix
    pub fn matrix_mut(&mut self) -> &mut Mat<f64> {
        &mut self.matrix
    }

    /// Get the [Parameters] object associated with this [Theta]
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// Get a mutable reference to the [Parameters] object
    pub fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
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
    pub fn add_point(&mut self, spp: &[f64]) -> Result<()> {
        if spp.len() != self.matrix.ncols() {
            bail!(
                "Support point length ({}) does not match number of parameters ({})",
                spp.len(),
                self.matrix.ncols()
            );
        }

        self.matrix
            .resize_with(self.matrix.nrows() + 1, self.matrix.ncols(), |_, i| spp[i]);
        Ok(())
    }

    /// Suggest a new support point to add to the matrix
    /// The point is only added if it is at least `min_dist` away from all existing support points
    /// and within the limits specified by `limits`
    pub(crate) fn suggest_point(&mut self, spp: &[f64], min_dist: f64) -> Result<()> {
        if self.check_point(spp, min_dist) {
            self.add_point(spp)?;
        }
        Ok(())
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

    /// Write the theta matrix to a CSV writer
    /// Each row represents a support point, each column represents a parameter
    pub fn to_csv<W: std::io::Write>(&self, writer: W) -> Result<()> {
        let mut csv_writer = csv::Writer::from_writer(writer);

        // Write each row
        for i in 0..self.matrix.nrows() {
            let row: Vec<f64> = (0..self.matrix.ncols())
                .map(|j| *self.matrix.get(i, j))
                .collect();
            csv_writer.serialize(row)?;
        }

        csv_writer.flush()?;
        Ok(())
    }

    /// Read theta matrix from a CSV reader
    /// Each row represents a support point, each column represents a parameter
    /// Note: This only reads the matrix values, not the parameter metadata
    pub fn from_csv<R: std::io::Read>(reader: R) -> Result<Self> {
        let mut csv_reader = csv::Reader::from_reader(reader);
        let mut rows: Vec<Vec<f64>> = Vec::new();

        for result in csv_reader.deserialize() {
            let row: Vec<f64> = result?;
            rows.push(row);
        }

        if rows.is_empty() {
            bail!("CSV file is empty");
        }

        let nrows = rows.len();
        let ncols = rows[0].len();

        // Verify all rows have the same length
        for (i, row) in rows.iter().enumerate() {
            if row.len() != ncols {
                bail!("Row {} has {} columns, expected {}", i, row.len(), ncols);
            }
        }

        // Create matrix from rows
        let mat = Mat::from_fn(nrows, ncols, |i, j| rows[i][j]);

        // Create empty parameters - user will need to set these separately
        let parameters = Parameters::new();

        Ok(Theta::from_parts(mat, parameters)?)
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

impl Serialize for Theta {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;

        let mut seq = serializer.serialize_seq(Some(self.matrix.nrows()))?;

        // Serialize each row as a vector
        for i in 0..self.matrix.nrows() {
            let row: Vec<f64> = (0..self.matrix.ncols())
                .map(|j| *self.matrix.get(i, j))
                .collect();
            seq.serialize_element(&row)?;
        }

        seq.end()
    }
}

impl<'de> Deserialize<'de> for Theta {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{SeqAccess, Visitor};
        use std::fmt;

        struct ThetaVisitor;

        impl<'de> Visitor<'de> for ThetaVisitor {
            type Value = Theta;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of rows (vectors of f64)")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut rows: Vec<Vec<f64>> = Vec::new();

                while let Some(row) = seq.next_element::<Vec<f64>>()? {
                    rows.push(row);
                }

                if rows.is_empty() {
                    return Err(serde::de::Error::custom("Empty matrix not allowed"));
                }

                let nrows = rows.len();
                let ncols = rows[0].len();

                // Verify all rows have the same length
                for (i, row) in rows.iter().enumerate() {
                    if row.len() != ncols {
                        return Err(serde::de::Error::custom(format!(
                            "Row {} has {} columns, expected {}",
                            i,
                            row.len(),
                            ncols
                        )));
                    }
                }

                // Create matrix from rows
                let mat = Mat::from_fn(nrows, ncols, |i, j| rows[i][j]);

                // Create empty parameters - user will need to set these separately
                let parameters = Parameters::new();

                Theta::from_parts(mat, parameters).map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_seq(ThetaVisitor)
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

        let mut theta = Theta::from_parts(matrix, parameters).unwrap();

        theta.filter_indices(&[0, 3]);

        // Expected result is a 2x2 matrix with filtered rows
        let expected = mat![[1.0, 2.0], [7.0, 8.0]];

        assert_eq!(theta.matrix, expected);
    }

    #[test]
    fn test_add_point() {
        let matrix = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let parameters = Parameters::new().add("A", 0.0, 10.0).add("B", 0.0, 10.0);

        let mut theta = Theta::from_parts(matrix, parameters).unwrap();

        theta.add_point(&[7.0, 8.0]).unwrap();

        let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        assert_eq!(theta.matrix, expected);
    }

    #[test]
    fn test_suggest_point() {
        let matrix = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let parameters = Parameters::new().add("A", 0.0, 10.0).add("B", 0.0, 10.0);
        let mut theta = Theta::from_parts(matrix, parameters).unwrap();
        theta.suggest_point(&[7.0, 8.0], 0.2).unwrap();
        let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        assert_eq!(theta.matrix, expected);

        // Suggest a point that is too close
        theta.suggest_point(&[7.1, 8.1], 0.2).unwrap();
        // The point should not be added
        assert_eq!(theta.matrix.nrows(), 4);
    }

    #[test]
    fn test_param_names() {
        let matrix = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let parameters = Parameters::new().add("A", 0.0, 10.0).add("B", 0.0, 10.0);

        let theta = Theta::from_parts(matrix, parameters).unwrap();
        let names = theta.param_names();
        assert_eq!(names, vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn test_set_matrix() {
        let matrix = mat![[1.0, 2.0], [3.0, 4.0]];
        let parameters = Parameters::new().add("A", 0.0, 10.0).add("B", 0.0, 10.0);
        let mut theta = Theta::from_parts(matrix, parameters).unwrap();

        let new_matrix = mat![[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        theta.matrix_mut().clone_from(&new_matrix);

        assert_eq!(theta.matrix(), &new_matrix);
    }
}
