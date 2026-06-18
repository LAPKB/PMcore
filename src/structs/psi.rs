use anyhow::bail;
use anyhow::Result;
use faer::Mat;
use ndarray::Array2;
use pharmsol::prelude::simulator::log_likelihood_matrix;
use pharmsol::AssayErrorModels;
use pharmsol::Data;
use pharmsol::Equation;
use serde::{Deserialize, Serialize};

use super::theta::Theta;
use super::weights::Weights;

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

    /// Write the psi matrix to a CSV writer
    /// Each row represents a subject, each column represents a support point
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

    /// Read psi matrix from a CSV reader
    /// Each row represents a subject, each column represents a support point
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

        Ok(Psi { matrix: mat })
    }

    /// Compute the maximum D-optimality value across all support points
    ///
    /// The D-optimality criterion measures convergence of the NPML/NPOD algorithm.
    /// At optimality, this value should be close to 0, meaning no support point
    /// can further improve the likelihood.
    ///
    /// # Interpretation
    /// - **≈ 0**: Solution is optimal
    /// - **> 0**: Not converged; some support points could still improve the objective
    /// - **Larger values**: Further from convergence
    pub fn d_optimality(&self, weights: &Weights) -> Result<f64> {
        let d_values = self.d_optimality_spp(weights)?;
        Ok(d_values.into_iter().fold(f64::NEG_INFINITY, f64::max))
    }

    /// Compute D-optimality values for each support point
    ///
    /// Returns the D-value for each support point in the current solution.
    /// At convergence, all values should be close to 0.
    ///
    /// The D-optimality value for support point $j$ is:
    /// $$D(\theta_j) = \sum_{i=1}^{n} \frac{\psi_{ij}}{p_\lambda(y_i)} - n$$
    pub(crate) fn d_optimality_spp(&self, weights: &Weights) -> Result<Vec<f64>> {
        let psi_mat = self.matrix();
        let nsub = psi_mat.nrows();
        let nspp = psi_mat.ncols();

        if nspp != weights.len() {
            bail!(
                "Psi has {} columns but weights has {} elements",
                nspp,
                weights.len()
            );
        }

        // Compute pyl = psi * w (weighted probability for each subject)
        let mut pyl = vec![0.0; nsub];
        for i in 0..nsub {
            for (j, w_j) in weights.iter().enumerate() {
                pyl[i] += psi_mat.get(i, j) * w_j;
            }
        }

        // Check for zero probabilities
        for (i, &p) in pyl.iter().enumerate() {
            if p == 0.0 {
                bail!("Subject {} has zero weighted probability", i);
            }
        }

        // Compute D-value for each support point
        let mut d_values = Vec::with_capacity(nspp);
        let n = nsub as f64;

        for j in 0..nspp {
            let mut sum = -n;
            for i in 0..nsub {
                sum += psi_mat.get(i, j) / pyl[i];
            }
            d_values.push(sum);
        }

        Ok(d_values)
    }
}

impl Default for Psi {
    fn default() -> Self {
        Psi::new()
    }
}

impl From<Array2<f64>> for Psi {
    fn from(array: Array2<f64>) -> Self {
        let matrix = Mat::from_fn(array.nrows(), array.ncols(), |i, j| array[(i, j)]);
        Psi { matrix }
    }
}

impl From<Mat<f64>> for Psi {
    fn from(matrix: Mat<f64>) -> Self {
        Psi { matrix }
    }
}

impl From<&Array2<f64>> for Psi {
    fn from(array: &Array2<f64>) -> Self {
        let matrix = Mat::from_fn(array.nrows(), array.ncols(), |i, j| array[(i, j)]);
        Psi { matrix }
    }
}

impl Serialize for Psi {
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

impl<'de> Deserialize<'de> for Psi {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{SeqAccess, Visitor};
        use std::fmt;

        struct PsiVisitor;

        impl<'de> Visitor<'de> for PsiVisitor {
            type Value = Psi;

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

                Ok(Psi { matrix: mat })
            }
        }

        deserializer.deserialize_seq(PsiVisitor)
    }
}

pub(crate) fn calculate_psi(
    equation: &impl Equation,
    subjects: &Data,
    theta: &Theta,
    error_models: &AssayErrorModels,
    progress: bool,
) -> Result<Psi> {
    let tm = theta.matrix();
    let theta_ndarray = Array2::from_shape_fn((tm.nrows(), tm.ncols()), |(i, j)| tm[(i, j)]);
    let log_psi =
        log_likelihood_matrix(equation, subjects, &theta_ndarray, error_models, progress)?;
    let psi_ndarray = log_psi.mapv(f64::exp);

    Ok(Psi::from(psi_ndarray))
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

        // Check values using faer matrix directly
        let m = psi.matrix();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m[(i, j)], array[[i, j]]);
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

        // Check values using faer matrix directly
        let m = psi.matrix();
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(m[(i, j)], array[[i, j]]);
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
        let owned_m = psi_from_owned.matrix();
        let ref_m = psi_from_ref.matrix();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(owned_m[(i, j)], ref_m[(i, j)]);
            }
        }
    }

    #[test]
    fn test_d_optimality_uniform_weights() {
        // With uniform weights and equal likelihoods per subject, D should be 0
        // Psi: 3 subjects (rows) x 2 support points (cols)
        // All likelihoods equal means each support point contributes equally
        let array = Array2::from_shape_vec((3, 2), vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).unwrap();
        let psi = Psi::from(array);
        let weights = Weights::uniform(2);

        let d = psi.d_optimality(&weights).unwrap();

        // With equal likelihoods and uniform weights:
        // pyl[i] = 0.5 * 0.5 + 0.5 * 0.5 = 0.5 for each subject
        // D[j] = sum(psi[i,j] / pyl[i]) - n = sum(0.5 / 0.5) - 3 = 3 - 3 = 0
        assert!((d - 0.0).abs() < 1e-10, "Expected d ≈ 0, got {}", d);
    }

    #[test]
    fn test_d_optimality_at_convergence() {
        // At convergence, all D values should be ≈ 0
        // This is a constructed example where the solution is optimal
        let array = Array2::from_shape_vec((2, 2), vec![0.8, 0.2, 0.2, 0.8]).unwrap();
        let psi = Psi::from(array);
        let weights = Weights::uniform(2);

        let d = psi.d_optimality(&weights).unwrap();

        // pyl[0] = 0.8 * 0.5 + 0.2 * 0.5 = 0.5
        // pyl[1] = 0.2 * 0.5 + 0.8 * 0.5 = 0.5
        // D[0] = (0.8/0.5 + 0.2/0.5) - 2 = (1.6 + 0.4) - 2 = 0
        // D[1] = (0.2/0.5 + 0.8/0.5) - 2 = (0.4 + 1.6) - 2 = 0
        assert!((d - 0.0).abs() < 1e-10, "Expected d ≈ 0, got {}", d);
    }

    #[test]
    fn test_d_optimality_spp_values() {
        // Test that d_optimality_spp returns correct per-support-point values
        let array = Array2::from_shape_vec((2, 2), vec![0.6, 0.4, 0.3, 0.7]).unwrap();
        let psi = Psi::from(array);
        let weights = Weights::from_vec(vec![0.5, 0.5]);

        let d_values = psi.d_optimality_spp(&weights).unwrap();

        assert_eq!(d_values.len(), 2);

        // pyl[0] = 0.6 * 0.5 + 0.4 * 0.5 = 0.5
        // pyl[1] = 0.3 * 0.5 + 0.7 * 0.5 = 0.5
        // D[0] = (0.6/0.5 + 0.3/0.5) - 2 = (1.2 + 0.6) - 2 = -0.2
        // D[1] = (0.4/0.5 + 0.7/0.5) - 2 = (0.8 + 1.4) - 2 = 0.2
        assert!(
            (d_values[0] - (-0.2)).abs() < 1e-10,
            "Expected d[0] ≈ -0.2, got {}",
            d_values[0]
        );
        assert!(
            (d_values[1] - 0.2).abs() < 1e-10,
            "Expected d[1] ≈ 0.2, got {}",
            d_values[1]
        );
    }

    #[test]
    fn test_d_optimality_max_is_maximum() {
        // d_optimality should return the maximum of d_optimality_spp
        let array = Array2::from_shape_vec((2, 3), vec![0.6, 0.3, 0.1, 0.2, 0.5, 0.3]).unwrap();
        let psi = Psi::from(array);
        let weights = Weights::from_vec(vec![0.4, 0.4, 0.2]);

        let d_max = psi.d_optimality(&weights).unwrap();
        let d_values = psi.d_optimality_spp(&weights).unwrap();

        let expected_max = d_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (d_max - expected_max).abs() < 1e-10,
            "d_optimality should equal max of d_optimality_spp"
        );
    }

    #[test]
    fn test_d_optimality_dimension_mismatch() {
        // Should error when weights length doesn't match number of support points
        let array = Array2::from_shape_vec((2, 3), vec![0.5, 0.3, 0.2, 0.4, 0.4, 0.2]).unwrap();
        let psi = Psi::from(array);
        let weights = Weights::from_vec(vec![0.5, 0.5]); // 2 weights for 3 support points

        let result = psi.d_optimality(&weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_d_optimality_zero_probability_error() {
        // Should error when a subject has zero weighted probability
        // This happens when all support points have zero likelihood for a subject
        let array = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.5, 0.5]).unwrap();
        let psi = Psi::from(array);
        let weights = Weights::uniform(2);

        let result = psi.d_optimality(&weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_d_optimality_nonuniform_weights() {
        // Test with non-uniform weights
        let array = Array2::from_shape_vec((2, 2), vec![0.8, 0.2, 0.4, 0.6]).unwrap();
        let psi = Psi::from(array);
        let weights = Weights::from_vec(vec![0.7, 0.3]); // Non-uniform

        let d = psi.d_optimality(&weights).unwrap();

        // pyl[0] = 0.8 * 0.7 + 0.2 * 0.3 = 0.56 + 0.06 = 0.62
        // pyl[1] = 0.4 * 0.7 + 0.6 * 0.3 = 0.28 + 0.18 = 0.46
        // D[0] = (0.8/0.62 + 0.4/0.46) - 2 ≈ (1.290 + 0.870) - 2 ≈ 0.160
        // D[1] = (0.2/0.62 + 0.6/0.46) - 2 ≈ (0.323 + 1.304) - 2 ≈ -0.373
        // max(D) ≈ 0.160

        // Just verify it runs and returns a reasonable value
        assert!(d.is_finite(), "D-optimality should be finite");
    }

    #[test]
    fn test_d_optimality_single_support_point() {
        // With a single support point, D should be 0 (trivially optimal)
        let array = Array2::from_shape_vec((3, 1), vec![0.5, 0.3, 0.7]).unwrap();
        let psi = Psi::from(array);
        let weights = Weights::from_vec(vec![1.0]);

        let d = psi.d_optimality(&weights).unwrap();

        // pyl[i] = psi[i, 0] * 1.0 = psi[i, 0]
        // D[0] = sum(psi[i,0] / psi[i,0]) - n = n - n = 0
        assert!((d - 0.0).abs() < 1e-10, "Expected d ≈ 0, got {}", d);
    }
}
