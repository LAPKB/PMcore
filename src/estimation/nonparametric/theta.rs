use std::{fmt::Debug, fs::File, path::Path};

use anyhow::{bail, Context, Result};
use faer::Mat;
use serde::{Deserialize, Serialize};

use super::sampling::{self, latin, sobol};
use super::weights::Weights;
use crate::model::{BoundedParameter, ParameterSpace};

/// [Theta] is a structure that holds the support points
/// These represent the joint population parameter distribution
///
/// Each row represents a support points, and each column a parameter
#[derive(Clone, PartialEq)]
pub struct Theta {
    matrix: Mat<f64>,
    parameters: ParameterSpace<BoundedParameter>,
}

impl Default for Theta {
    fn default() -> Self {
        Theta {
            matrix: Mat::new(),
            parameters: ParameterSpace::<BoundedParameter>::new(),
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
    pub fn from_parts(
        matrix: Mat<f64>,
        parameters: ParameterSpace<BoundedParameter>,
    ) -> Result<Self> {
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
    pub fn parameters(&self) -> &ParameterSpace<BoundedParameter> {
        &self.parameters
    }

    /// Get a mutable reference to the [ParameterSpace]
    pub fn parameters_mut(&mut self) -> &mut ParameterSpace<BoundedParameter> {
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

        let limits = self.parameters.finite_ranges();

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

    /// Create a new Theta with an additional parameter column.
    ///
    /// All existing rows keep their values. The new column is filled with
    /// `initial_value` for every row. Returns a new `Theta` — does not
    /// mutate in place.
    ///
    /// # Errors
    ///
    /// - `name` already exists in the parameter space
    /// - `lower` or `upper` are non-finite
    /// - `lower >= upper`
    pub fn with_added_parameter(
        &self,
        name: &str,
        lower: f64,
        upper: f64,
        initial_value: f64,
    ) -> Result<Theta> {
        // Validate uniqueness
        if self.parameters().iter().any(|p| p.name.as_str() == name) {
            bail!("parameter '{}' already exists in theta", name);
        }

        // Validate bounds
        if !lower.is_finite() || !upper.is_finite() {
            bail!(
                "bounds must be finite for parameter '{}': [{}, {}]",
                name,
                lower,
                upper
            );
        }
        if lower >= upper {
            bail!(
                "lower bound ({}) must be strictly less than upper bound ({}) for parameter '{}'",
                lower,
                upper,
                name
            );
        }

        let (nrows, ncols) = (self.matrix().nrows(), self.matrix().ncols());
        let new_matrix = faer::Mat::from_fn(nrows, ncols + 1, |r, c| {
            if c < ncols {
                self.matrix()[(r, c)]
            } else {
                initial_value
            }
        });

        let new_params = self.parameters().clone().add(name, lower, upper);

        Theta::from_parts(new_matrix, new_params)
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
        let parameters = ParameterSpace::<BoundedParameter>::new();

        Theta::from_parts(mat, parameters)
    }

    /// Generate a starting grid of `points` support points over `parameters`
    /// using a Sobol sequence and the default seed ([`sampling::DEFAULT_SEED`]).
    ///
    /// The returned [Theta] carries `parameters`, so the chosen grid is explicit
    /// and self-describing.
    pub fn sobol(parameters: &ParameterSpace<BoundedParameter>, points: usize) -> Result<Self> {
        Self::sobol_with_seed(parameters, points, sampling::DEFAULT_SEED)
    }

    /// Generate a starting grid over `parameters` using a Sobol sequence with the
    /// default number of support points ([`sampling::DEFAULT_POINTS`]) and the
    /// default seed ([`sampling::DEFAULT_SEED`]).
    pub fn sobol_default(parameters: &ParameterSpace<BoundedParameter>) -> Result<Self> {
        Self::sobol(parameters, sampling::DEFAULT_POINTS)
    }

    /// Like [`Theta::sobol`], with an explicit seed for the quasi-random sequence.
    pub fn sobol_with_seed(
        parameters: &ParameterSpace<BoundedParameter>,
        points: usize,
        seed: usize,
    ) -> Result<Self> {
        validate_bounds(parameters)?;
        sobol::generate(parameters, points, seed)
    }

    /// Generate a starting grid of `points` support points over `parameters`
    /// using Latin Hypercube Sampling and the default seed ([`sampling::DEFAULT_SEED`]).
    ///
    /// The returned [Theta] carries `parameters`, so the chosen grid is explicit
    /// and self-describing.
    pub fn latin(parameters: &ParameterSpace<BoundedParameter>, points: usize) -> Result<Self> {
        Self::latin_with_seed(parameters, points, sampling::DEFAULT_SEED)
    }

    /// Like [`Theta::latin`], with an explicit seed for the quasi-random sequence.
    pub fn latin_with_seed(
        parameters: &ParameterSpace<BoundedParameter>,
        points: usize,
        seed: usize,
    ) -> Result<Self> {
        validate_bounds(parameters)?;
        latin::generate(parameters, points, seed)
    }

    pub fn from_file(
        path: impl AsRef<Path>,
        parameters: &ParameterSpace<BoundedParameter>,
    ) -> Result<(Theta, Option<Weights>)> {
        let path = path.as_ref();
        tracing::info!("Reading prior from {}", path.display());
        let file = File::open(path).context(format!(
            "Unable to open the prior file '{}'",
            path.display()
        ))?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let mut parameter_names: Vec<String> = reader
            .headers()?
            .clone()
            .into_iter()
            .map(|s| s.trim().to_owned())
            .collect();

        let prob_index = parameter_names.iter().position(|name| name == "prob");
        if let Some(index) = prob_index {
            parameter_names.remove(index);
        }

        let random_names: Vec<String> = parameters.names();

        let mut reordered_indices: Vec<usize> = Vec::new();
        for random_name in &random_names {
            match parameter_names.iter().position(|name| name == random_name) {
                Some(index) => {
                    let adjusted_index = if let Some(prob_idx) = prob_index {
                        if index >= prob_idx {
                            index + 1
                        } else {
                            index
                        }
                    } else {
                        index
                    };
                    reordered_indices.push(adjusted_index);
                }
                None => bail!("Parameter {} is not present in the CSV file.", random_name),
            }
        }

        if parameter_names.len() > random_names.len() {
            let extra_parameters: Vec<&String> = parameter_names.iter().collect();
            bail!(
                "Found parameters in the prior not present in configuration: {:?}",
                extra_parameters
            );
        }

        let mut theta_values = Vec::new();
        let mut prob_values = Vec::new();

        for result in reader.records() {
            let record = result.unwrap();
            let values: Vec<f64> = reordered_indices
                .iter()
                .map(|&i| record[i].parse::<f64>().unwrap())
                .collect();
            theta_values.push(values);

            if let Some(prob_idx) = prob_index {
                let prob_value: f64 = record[prob_idx].parse::<f64>().unwrap();
                prob_values.push(prob_value);
            }
        }

        let n_points = theta_values.len();
        let n_params = random_names.len();
        let theta_values: Vec<f64> = theta_values.into_iter().flatten().collect();
        let theta_matrix: Mat<f64> =
            Mat::from_fn(n_points, n_params, |i, j| theta_values[i * n_params + j]);

        let theta = Theta::from_parts(theta_matrix, parameters.clone())?;
        let weights = if !prob_values.is_empty() {
            Some(Weights::from_vec(prob_values))
        } else {
            None
        };

        Ok((theta, weights))
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
            .map(|i| {
                (0..self.matrix.ncols())
                    .map(|j| *self.matrix.get(i, j))
                    .collect()
            })
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
            parameters: ParameterSpace<BoundedParameter>,
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

/// Validates that every parameter has a strictly-ordered, finite bound interval.
fn validate_bounds(parameters: &ParameterSpace<BoundedParameter>) -> Result<()> {
    for parameter in parameters.iter() {
        if parameter.lower >= parameter.upper {
            bail!(
                "Parameter '{}' has invalid bounds: [{}, {}]. Lower bound must be less than upper bound.",
                parameter.name,
                parameter.lower,
                parameter.upper
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn parameters() -> ParameterSpace<BoundedParameter> {
        ParameterSpace::<BoundedParameter>::new()
            .add("ke", 0.1, 1.0)
            .add("v", 5.0, 50.0)
    }

    fn temp_csv_path() -> String {
        format!("test_temp_theta_{}.csv", rand::random::<u32>())
    }

    #[test]
    fn sobol_generates_expected_shape() {
        let theta = Theta::sobol_with_seed(&parameters(), 10, 42).unwrap();
        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 2);
    }

    #[test]
    fn latin_generates_expected_shape() {
        let theta = Theta::latin(&parameters(), 10).unwrap();
        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 2);
    }

    #[test]
    fn sampling_rejects_invalid_bounds() {
        let bad = ParameterSpace::<BoundedParameter>::new().add("ke", 1.0, 1.0);
        let err = Theta::sobol(&bad, 10).unwrap_err();
        assert!(err.to_string().contains("invalid bounds"));
    }

    #[test]
    fn from_file_parses_weights_and_reorders_columns() {
        let path = temp_csv_path();
        fs::write(&path, "v,ke,prob\n10.0,0.5,0.3\n15.0,0.7,0.7\n").unwrap();

        let (theta, weights) = Theta::from_file(&path, &parameters()).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(theta.nspp(), 2);
        assert_eq!(theta.matrix()[(0, 0)], 0.5);
        assert_eq!(theta.matrix()[(0, 1)], 10.0);

        let weights = weights.expect("weights should be parsed from prob column");
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0], 0.3);
        assert_eq!(weights[1], 0.7);
    }

    #[test]
    fn from_file_rejects_extra_parameters() {
        let path = temp_csv_path();
        fs::write(&path, "ke,v,extra\n0.5,10.0,1.0\n").unwrap();

        let err = Theta::from_file(&path, &parameters()).unwrap_err();
        let _ = fs::remove_file(&path);

        assert!(err
            .to_string()
            .contains("Found parameters in the prior not present in configuration"));
    }
}
