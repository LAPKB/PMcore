use std::fmt::Debug;

use anyhow::{bail, Context, Result};
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

    pub(crate) fn from_parts(matrix: Mat<f64>, parameters: Parameters) -> Self {
        Theta { matrix, parameters }
    }

    /// Create a new [Theta] using Sobol sequence sampling
    ///
    /// # Arguments
    ///
    /// * `parameters` - The [Parameters] struct, which contains the parameters to be sampled
    /// * `points` - The number of points to generate, i.e. the number of rows in the matrix
    /// * `seed` - The seed for the Sobol sequence generator
    ///
    /// # Returns
    ///
    /// A [Result] containing the [Theta] structure with the support point matrix
    pub fn from_sobol(parameters: Parameters, points: usize, seed: usize) -> Result<Self> {
        // Validate parameter bounds
        Self::validate_parameters(&parameters)?;

        let seed = seed as u32;
        let params: Vec<(String, f64, f64)> = parameters
            .iter()
            .map(|p| (p.name.clone(), p.lower, p.upper))
            .collect();

        let rand_matrix = Mat::from_fn(points, params.len(), |i, j| {
            let unscaled =
                sobol_burley::sample((i).try_into().unwrap(), j.try_into().unwrap(), seed) as f64;
            let (_name, lower, upper) = params.get(j).unwrap();
            lower + unscaled * (upper - lower)
        });

        let theta = Theta::from_parts(rand_matrix, parameters);
        Ok(theta)
    }

    /// Create a new [Theta] using Latin Hypercube sampling
    ///
    /// # Arguments
    ///
    /// * `parameters` - The [Parameters] struct, which contains the parameters to be sampled
    /// * `points` - The number of points to generate, i.e. the number of rows in the matrix
    /// * `seed` - The seed for the random number generator
    ///
    /// # Returns
    ///
    /// A [Result] containing the [Theta] structure with the support point matrix
    pub fn from_latin(parameters: Parameters, points: usize, seed: usize) -> Result<Self> {
        use rand::prelude::*;
        use rand::rngs::StdRng;
        use rand::Rng;

        // Validate parameter bounds
        Self::validate_parameters(&parameters)?;

        let params: Vec<(String, f64, f64)> = parameters
            .iter()
            .map(|p| (p.name.clone(), p.lower, p.upper))
            .collect();

        // Initialize random number generator with the provided seed
        let mut rng = StdRng::seed_from_u64(seed as u64);

        // Create and shuffle intervals for each parameter
        let mut intervals = Vec::new();
        for _ in 0..params.len() {
            let mut param_intervals: Vec<f64> = (0..points).map(|i| i as f64).collect();
            param_intervals.shuffle(&mut rng);
            intervals.push(param_intervals);
        }

        let rand_matrix = Mat::from_fn(points, params.len(), |i, j| {
            // Get the interval for this parameter and point
            let interval = intervals[j][i];
            let random_offset = rng.random::<f64>();
            // Calculate normalized value in [0,1]
            let unscaled = (interval + random_offset) / points as f64;
            // Scale to parameter range
            let (_name, lower, upper) = params.get(j).unwrap();
            lower + unscaled * (upper - lower)
        });

        let theta = Theta::from_parts(rand_matrix, parameters);
        Ok(theta)
    }

    /// Create a new [Theta] by reading from a CSV file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the CSV file containing the prior distribution
    /// * `parameters` - The [Parameters] struct defining expected parameters
    ///
    /// # Returns
    ///
    /// A [Result] containing the [Theta] structure with the support point matrix
    pub fn from_file<P: AsRef<str>>(path: P, parameters: Parameters) -> Result<Self> {
        use std::fs::File;

        // Validate parameter bounds
        Self::validate_parameters(&parameters)?;

        let path = path.as_ref();
        tracing::info!("Reading prior from {}", path);
        let file = File::open(path).context(format!("Unable to open the prior file '{}'", path))?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let mut parameter_names: Vec<String> = reader
            .headers()?
            .clone()
            .into_iter()
            .map(|s| s.trim().to_owned())
            .collect();

        // Remove "prob" column if present
        if let Some(index) = parameter_names.iter().position(|name| name == "prob") {
            parameter_names.remove(index);
        }

        // Check and reorder parameters to match names in parameters
        let random_names: Vec<String> = parameters.names();

        let mut reordered_indices: Vec<usize> = Vec::new();
        for random_name in &random_names {
            match parameter_names.iter().position(|name| name == random_name) {
                Some(index) => {
                    reordered_indices.push(index);
                }
                None => {
                    bail!("Parameter {} is not present in the CSV file.", random_name);
                }
            }
        }

        // Check if there are remaining parameters not present in parameters
        if parameter_names.len() > random_names.len() {
            let extra_parameters: Vec<&String> = parameter_names.iter().collect();
            bail!(
                "Found parameters in the prior not present in configuration: {:?}",
                extra_parameters
            );
        }

        // Read parameter values row by row, keeping only those associated with the reordered parameters
        let mut theta_values = Vec::new();
        for result in reader.records() {
            let record = result.unwrap();
            let values: Vec<f64> = reordered_indices
                .iter()
                .map(|&i| record[i].parse::<f64>().unwrap())
                .collect();
            theta_values.push(values);
        }

        let n_points = theta_values.len();
        let n_params = random_names.len();

        // Convert nested Vec into a single Vec
        let theta_values: Vec<f64> = theta_values.into_iter().flatten().collect();

        let theta_matrix: Mat<f64> =
            Mat::from_fn(n_points, n_params, |i, j| theta_values[i * n_params + j]);

        let theta = Theta::from_parts(theta_matrix, parameters);

        Ok(theta)
    }

    /// Validate parameter bounds to ensure they are finite and lower < upper
    fn validate_parameters(parameters: &Parameters) -> Result<()> {
        for param in parameters.iter() {
            if param.lower.is_infinite() || param.upper.is_infinite() {
                bail!(
                    "Parameter '{}' has infinite bounds: [{}, {}]",
                    param.name,
                    param.lower,
                    param.upper
                );
            }

            // Ensure that the lower bound is less than the upper bound
            if param.lower >= param.upper {
                bail!(
                    "Parameter '{}' has invalid bounds: [{}, {}]. Lower bound must be less than upper bound.",
                    param.name,
                    param.lower,
                    param.upper
                );
            }
        }
        Ok(())
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

        Ok(Theta::from_parts(mat, parameters))
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

                Ok(Theta::from_parts(mat, parameters))
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

    #[test]
    fn test_suggest_point() {
        let matrix = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let parameters = Parameters::new().add("A", 0.0, 10.0).add("B", 0.0, 10.0);
        let mut theta = Theta::from_parts(matrix, parameters);
        theta.suggest_point(&[7.0, 8.0], 0.2);
        let expected = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        assert_eq!(theta.matrix, expected);

        // Suggest a point that is too close
        theta.suggest_point(&[7.1, 8.1], 0.2);
        // The point should not be added
        assert_eq!(theta.matrix.nrows(), 4);
    }

    #[test]
    fn test_param_names() {
        let matrix = mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let parameters = Parameters::new().add("A", 0.0, 10.0).add("B", 0.0, 10.0);

        let theta = Theta::from_parts(matrix, parameters);
        let names = theta.param_names();
        assert_eq!(names, vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn test_from_sobol() {
        let parameters = Parameters::new().add("ke", 0.1, 1.0).add("v", 5.0, 50.0);

        let theta = Theta::from_sobol(parameters, 10, 42).unwrap();

        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 2);

        // Verify values are within bounds
        for i in 0..theta.matrix().nrows() {
            let ke_val = *theta.matrix().get(i, 0);
            let v_val = *theta.matrix().get(i, 1);
            assert!(
                ke_val >= 0.1 && ke_val <= 1.0,
                "ke value {} out of bounds",
                ke_val
            );
            assert!(
                v_val >= 5.0 && v_val <= 50.0,
                "v value {} out of bounds",
                v_val
            );
        }
    }

    #[test]
    fn test_from_latin() {
        let parameters = Parameters::new().add("ke", 0.1, 1.0).add("v", 5.0, 50.0);

        let theta = Theta::from_latin(parameters, 15, 123).unwrap();

        assert_eq!(theta.nspp(), 15);
        assert_eq!(theta.matrix().ncols(), 2);

        // Verify values are within bounds
        for i in 0..theta.matrix().nrows() {
            let ke_val = *theta.matrix().get(i, 0);
            let v_val = *theta.matrix().get(i, 1);
            assert!(
                ke_val >= 0.1 && ke_val <= 1.0,
                "ke value {} out of bounds",
                ke_val
            );
            assert!(
                v_val >= 5.0 && v_val <= 50.0,
                "v value {} out of bounds",
                v_val
            );
        }
    }

    #[test]
    fn test_from_file_valid() {
        use std::fs;

        let csv_content = "ke,v\n0.2,10.0\n0.5,25.0\n0.8,40.0\n";
        let temp_path = format!("test_temp_{}.csv", rand::random::<u32>());
        fs::write(&temp_path, csv_content).unwrap();

        let parameters = Parameters::new().add("ke", 0.1, 1.0).add("v", 5.0, 50.0);
        let theta = Theta::from_file(&temp_path, parameters).unwrap();

        assert_eq!(theta.nspp(), 3);
        assert_eq!(theta.matrix().ncols(), 2);

        // Clean up
        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_from_file_with_prob_column() {
        use std::fs;

        let csv_content = "ke,v,prob\n0.2,10.0,0.5\n0.5,25.0,0.3\n0.8,40.0,0.2\n";
        let temp_path = format!("test_temp_{}.csv", rand::random::<u32>());
        fs::write(&temp_path, csv_content).unwrap();

        let parameters = Parameters::new().add("ke", 0.1, 1.0).add("v", 5.0, 50.0);
        let theta = Theta::from_file(&temp_path, parameters).unwrap();

        assert_eq!(theta.nspp(), 3);
        assert_eq!(theta.matrix().ncols(), 2);

        // Clean up
        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_from_sobol_infinite_bounds() {
        let parameters = Parameters::new()
            .add("ke", f64::NEG_INFINITY, 1.0)
            .add("v", 5.0, 50.0);

        let result = Theta::from_sobol(parameters, 10, 42);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("infinite bounds"));
    }

    #[test]
    fn test_from_latin_invalid_bounds() {
        let parameters = Parameters::new()
            .add("ke", 1.0, 0.5) // Invalid: lower >= upper
            .add("v", 5.0, 50.0);

        let result = Theta::from_latin(parameters, 10, 42);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid bounds"));
    }

    #[test]
    fn test_from_file_missing_parameter() {
        use std::fs;

        let csv_content = "ke\n0.2\n0.5\n0.8\n";
        let temp_path = format!("test_temp_{}.csv", rand::random::<u32>());
        fs::write(&temp_path, csv_content).unwrap();

        let parameters = Parameters::new().add("ke", 0.1, 1.0).add("v", 5.0, 50.0);
        let result = Theta::from_file(&temp_path, parameters);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Parameter v is not present"));

        // Clean up
        let _ = fs::remove_file(&temp_path);
    }

    #[test]
    fn test_from_file_nonexistent() {
        let parameters = Parameters::new().add("ke", 0.1, 1.0);
        let result = Theta::from_file("nonexistent_file.csv", parameters);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unable to open the prior file"));
    }
}
