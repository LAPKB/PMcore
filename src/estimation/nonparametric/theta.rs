use std::fmt::Debug;

use anyhow::{bail, Result};
use faer::Mat;
use serde::{Deserialize, Serialize};

use super::weights::Weights;
use crate::model::ParameterSpace;

/// [Theta] is a structure that holds the support points
/// These represent the joint population parameter distribution
///
/// Each row represents a support points, and each column a parameter
#[derive(Clone, PartialEq)]
pub struct Theta {
    matrix: Mat<f64>,
    parameters: ParameterSpace,
}

impl Default for Theta {
    fn default() -> Self {
        Theta {
            matrix: Mat::new(),
            parameters: ParameterSpace::new(),
        }
    }
}

impl Theta {
    pub fn new() -> Self {
        Theta::default()
    }

    /// Create a new [Theta] from a matrix and [ParameterSpace]
    ///
    /// It is important that the number of columns in the matrix matches the number of parameters
    /// in the [ParameterSpace]
    ///
    /// The order of parameters in the [ParameterSpace] should match the order of columns in the matrix
    pub fn from_parts(matrix: Mat<f64>, parameters: impl Into<ParameterSpace>) -> Result<Self> {
        let parameters = parameters.into();
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

    /// Get the [ParameterSpace] associated with this [Theta]
    pub fn parameters(&self) -> &ParameterSpace {
        &self.parameters
    }

    /// Get a mutable reference to the [ParameterSpace]
    pub fn parameters_mut(&mut self) -> &mut ParameterSpace {
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

        let limits = self
            .parameters
            .finite_ranges()
            .expect("theta requires finite parameter bounds");

        for row_idx in 0..self.matrix.nrows() {
            let mut squared_dist = 0.0;
            for (i, val) in spp.iter().enumerate() {
                let normalized_diff =
                    (val - self.matrix.get(row_idx, i)) / (limits[i].1 - limits[i].0);
                squared_dist += normalized_diff * normalized_diff;
            }
            let dist = squared_dist.sqrt();
            if dist <= min_dist {
                return false;
            }
        }
        true
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

    /// Write the matrix to a CSV file with weights
    pub fn write_with_weights(&self, path: &str, weights: &Weights) -> Result<()> {
        if self.nspp() != weights.len() {
            bail!(
                "Number of support points ({}) does not match number of weights ({})",
                self.nspp(),
                weights.len()
            );
        }

        let mut writer = csv::Writer::from_path(path)?;

        let header: Vec<String> = self
            .parameters
            .names()
            .iter()
            .cloned()
            .chain(std::iter::once("prob".to_string()))
            .collect();

        writer.write_record(header)?;

        for (row_idx, row) in self.matrix.row_iter().enumerate() {
            let mut record: Vec<String> = row.iter().map(|x| x.to_string()).collect();
            record.push(weights[row_idx].to_string());
            writer.write_record(record)?;
        }
        Ok(())
    }

    /// Write the theta matrix to a CSV writer
    /// Each row represents a support point, each column represents a parameter
    pub fn to_csv<W: std::io::Write>(&self, writer: W) -> Result<()> {
        let mut csv_writer = csv::Writer::from_writer(writer);

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

        for (i, row) in rows.iter().enumerate() {
            if row.len() != ncols {
                bail!("Row {} has {} columns, expected {}", i, row.len(), ncols);
            }
        }

        let mat = Mat::from_fn(nrows, ncols, |i, j| rows[i][j]);
        let parameters = ParameterSpace::new();

        Theta::from_parts(mat, parameters)
    }
}

impl Debug for Theta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\nTheta contains {} support points\n", self.nspp())?;

        for name in self.parameters.names().iter() {
            write!(f, "\t{}", name)?;
        }
        writeln!(f)?;
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
        use serde::ser::SerializeStruct;

        let rows: Vec<Vec<f64>> = (0..self.matrix.nrows())
            .map(|i| (0..self.matrix.ncols()).map(|j| *self.matrix.get(i, j)).collect())
            .collect();

        let mut state = serializer.serialize_struct("Theta", 2)?;
        state.serialize_field("matrix", &rows)?;
        state.serialize_field("parameters", &self.parameters)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Theta {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ThetaSerde {
            matrix: Vec<Vec<f64>>,
            parameters: ParameterSpace,
        }

        let decoded = ThetaSerde::deserialize(deserializer)?;

        if decoded.matrix.is_empty() {
            return Ok(Self {
                matrix: Mat::new(),
                parameters: decoded.parameters,
            });
        }

        let nrows = decoded.matrix.len();
        let ncols = decoded.matrix[0].len();
        for (index, row) in decoded.matrix.iter().enumerate() {
            if row.len() != ncols {
                return Err(serde::de::Error::custom(format!(
                    "Row {} has {} columns, expected {}",
                    index,
                    row.len(),
                    ncols
                )));
            }
        }

        let matrix = Mat::from_fn(nrows, ncols, |i, j| decoded.matrix[i][j]);
        Self::from_parts(matrix, decoded.parameters).map_err(serde::de::Error::custom)
    }
}