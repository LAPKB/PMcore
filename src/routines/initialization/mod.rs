use std::fs::File;

use crate::structs::theta::Theta;
use anyhow::{bail, Context, Result};
use faer::Mat;
use serde::{Deserialize, Serialize};

use crate::routines::settings::Settings;

pub mod latin;
pub mod sobol;

/// The sampler used to generate the grid of support points
///
/// The sampler can be one of the following:
///
/// - `Sobol`: Generates a Sobol sequence
/// - `Latin`: Generates a Latin hypercube
/// - `File`: Reads the prior distribution from a CSV file
#[derive(Debug, Deserialize, Clone, Serialize)]
pub enum Prior {
    Sobol(usize, usize),
    Latin(usize, usize),
    File(String),
    #[serde(skip)]
    Theta(Theta),
}

impl Prior {
    pub fn sobol(points: usize, seed: usize) -> Prior {
        Prior::Sobol(points, seed)
    }

    /// Get the number of initial support points
    ///
    /// This function returns the number of points for Sobol and Latin samplers,
    /// and returns `None` for file-based priors since they do not have a fixed number of points.
    /// For custom priors ([Prior::Theta]), it returns the number of support points in the original [Theta] structure.
    pub fn points(&self) -> Option<usize> {
        match self {
            Prior::Sobol(points, _) => Some(*points),
            Prior::Latin(points, _) => Some(*points),
            Prior::File(_) => None, // File-based prior does not have a fixed number of points
            Prior::Theta(theta) => Some(theta.nspp()),
        }
    }

    /// Get the seed used for the random number generator
    ///
    /// This function returns the seed for Sobol and Latin samplers,
    /// and returns `None` for file-based priors since they do not have a fixed seed.
    /// For custom priors ([Prior::Theta]), it returns `None` as they do not have a fixed seed.
    pub fn seed(&self) -> Option<usize> {
        match self {
            Prior::Sobol(_, seed) => Some(*seed),
            Prior::Latin(_, seed) => Some(*seed),
            Prior::File(_) => None, // "File-based prior does not have a fixed seed"
            Prior::Theta(_) => None, // Custom prior does not have a fixed seed
        }
    }
}

impl Default for Prior {
    fn default() -> Self {
        Prior::Sobol(2028, 22)
    }
}

/// This function generates the grid of support points according to the sampler specified in the [Settings]
pub fn sample_space(settings: &Settings) -> Result<Theta> {
    // Ensure that the parameter ranges are not infinite
    for param in settings.parameters().iter() {
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

    // Otherwise, parse the sampler type and generate the grid
    let prior = match settings.prior() {
        Prior::Sobol(points, seed) => sobol::generate(settings.parameters(), *points, *seed)?,
        Prior::Latin(points, seed) => latin::generate(settings.parameters(), *points, *seed)?,
        Prior::File(ref path) => parse_prior(path, settings)?,
        Prior::Theta(ref theta) => {
            // If a custom prior is provided, return it directly
            return Ok(theta.clone());
        }
    };
    Ok(prior)
}

/// This function reads the prior distribution from a file
pub fn parse_prior(path: &String, settings: &Settings) -> Result<Theta> {
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

    // Check and reorder parameters to match names in settings.parsed.random
    let random_names: Vec<String> = settings.parameters().names();

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

    // Check if there are remaining parameters not present in settings.parsed.random
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

    let theta = Theta::from_parts(theta_matrix, settings.parameters().clone())?;

    Ok(theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use pharmsol::{ErrorModel, ErrorModels, ErrorPoly};
    use std::fs;

    fn create_test_settings() -> Settings {
        let parameters = Parameters::new().add("ke", 0.1, 1.0).add("v", 5.0, 50.0);

        let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
        let ems = ErrorModels::new().add(0, em).unwrap();

        Settings::builder()
            .set_algorithm(Algorithm::NPAG)
            .set_parameters(parameters)
            .set_error_models(ems)
            .build()
    }

    fn create_temp_csv_file(content: &str) -> String {
        let temp_path = format!("test_temp_{}.csv", rand::random::<u32>());
        fs::write(&temp_path, content).unwrap();
        temp_path
    }

    fn cleanup_temp_file(path: &str) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_prior_sobol_creation() {
        let prior = Prior::sobol(100, 42);
        assert_eq!(prior.points(), Some(100));
        assert_eq!(prior.seed(), Some(42));
    }

    #[test]
    fn test_prior_latin_creation() {
        let prior = Prior::Latin(50, 123);
        assert_eq!(prior.points(), Some(50));
        assert_eq!(prior.seed(), Some(123));
    }

    #[test]
    fn test_prior_default() {
        let prior = Prior::default();
        assert_eq!(prior.points(), Some(2028));
        assert_eq!(prior.seed(), Some(22));
    }

    #[test]
    fn test_prior_file_points() {
        let prior = Prior::File("test.csv".to_string());
        assert_eq!(prior.points(), None);
    }

    #[test]
    fn test_prior_file_seed() {
        let prior = Prior::File("test.csv".to_string());
        assert_eq!(prior.seed(), None);
    }

    #[test]
    fn test_sample_space_sobol() {
        let mut settings = create_test_settings();
        settings.set_prior(Prior::sobol(10, 42));

        let result = sample_space(&settings);
        assert!(result.is_ok());

        let theta = result.unwrap();
        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 2);
    }

    #[test]
    fn test_sample_space_latin() {
        let mut settings = create_test_settings();
        settings.set_prior(Prior::Latin(15, 123));

        let result = sample_space(&settings);
        assert!(result.is_ok());

        let theta = result.unwrap();
        assert_eq!(theta.nspp(), 15);
        assert_eq!(theta.matrix().ncols(), 2);
    }

    #[test]
    fn test_sample_space_custom_theta() {
        let mut settings = create_test_settings();

        // Create a custom theta
        let parameters = settings.parameters().clone();
        let matrix = faer::Mat::from_fn(3, 2, |i, j| (i + j) as f64);
        let custom_theta = Theta::from_parts(matrix, parameters).unwrap();

        let prior = Prior::Theta(custom_theta.clone());
        settings.set_prior(Prior::Theta(custom_theta.clone()));

        let result = sample_space(&settings);
        assert!(result.is_ok());

        let theta = result.unwrap();
        assert_eq!(theta.nspp(), 3);
        assert_eq!(theta.matrix().ncols(), 2);
        assert_eq!(theta, custom_theta);
        assert!(prior.points() == Some(3));
    }

    #[test]
    fn test_sample_space_infinite_bounds_error() {
        let parameters = Parameters::new()
            .add("ke", f64::NEG_INFINITY, 1.0) // Invalid: infinite lower bound
            .add("v", 5.0, 50.0);

        let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
        let ems = ErrorModels::new().add(0, em).unwrap();

        let mut settings = Settings::builder()
            .set_algorithm(Algorithm::NPAG)
            .set_parameters(parameters)
            .set_error_models(ems)
            .build();

        settings.set_prior(Prior::sobol(10, 42));

        let result = sample_space(&settings);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("infinite bounds"));
    }

    #[test]
    fn test_sample_space_invalid_bounds_error() {
        let parameters = Parameters::new()
            .add("ke", 1.0, 0.5) // Invalid: lower bound >= upper bound
            .add("v", 5.0, 50.0);

        let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
        let ems = ErrorModels::new().add(0, em).unwrap();

        let mut settings = Settings::builder()
            .set_algorithm(Algorithm::NPAG)
            .set_parameters(parameters)
            .set_error_models(ems)
            .build();

        settings.set_prior(Prior::sobol(10, 42));

        let result = sample_space(&settings);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid bounds"));
    }

    #[test]
    fn test_parse_prior_valid_file() {
        let csv_content = "ke,v\n0.1,10.0\n0.2,15.0\n0.3,20.0\n";
        let temp_path = create_temp_csv_file(csv_content);

        let settings = create_test_settings();

        let result = parse_prior(&temp_path, &settings);
        assert!(result.is_ok());

        let theta = result.unwrap();
        assert_eq!(theta.nspp(), 3);
        assert_eq!(theta.matrix().ncols(), 2);

        cleanup_temp_file(&temp_path);
    }

    #[test]
    fn test_parse_prior_with_prob_column() {
        let csv_content = "ke,v,prob\n0.1,10.0,0.5\n0.2,15.0,0.3\n0.3,20.0,0.2\n";
        let temp_path = create_temp_csv_file(csv_content);

        let settings = create_test_settings();

        let result = parse_prior(&temp_path, &settings);
        assert!(result.is_ok());

        let theta = result.unwrap();
        assert_eq!(theta.nspp(), 3);
        assert_eq!(theta.matrix().ncols(), 2);

        cleanup_temp_file(&temp_path);
    }

    #[test]
    fn test_parse_prior_missing_parameter() {
        let csv_content = "ke\n0.1\n0.2\n0.3\n";
        let temp_path = create_temp_csv_file(csv_content);

        let settings = create_test_settings();

        let result = parse_prior(&temp_path, &settings);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Parameter v is not present"));

        cleanup_temp_file(&temp_path);
    }

    #[test]
    fn test_parse_prior_extra_parameters() {
        let csv_content = "ke,v,extra_param\n0.1,10.0,1.0\n0.2,15.0,2.0\n0.3,20.0,3.0\n";
        let temp_path = create_temp_csv_file(csv_content);

        let settings = create_test_settings();

        let result = parse_prior(&temp_path, &settings);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Found parameters in the prior not present in configuration"));

        cleanup_temp_file(&temp_path);
    }

    #[test]
    fn test_parse_prior_nonexistent_file() {
        let settings = create_test_settings();
        let file_path = "nonexistent_file.csv".to_string();

        let result = parse_prior(&file_path, &settings);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unable to open the prior file"));
    }

    #[test]
    fn test_parse_prior_reordered_columns() {
        let csv_content = "v,ke\n10.0,0.1\n15.0,0.2\n20.0,0.3\n";
        let temp_path = create_temp_csv_file(csv_content);

        let settings = create_test_settings();

        let result = parse_prior(&temp_path, &settings);
        assert!(result.is_ok());

        let theta = result.unwrap();
        assert_eq!(theta.nspp(), 3);
        assert_eq!(theta.matrix().ncols(), 2);

        // Verify the values are correctly reordered (ke should be first, v second)
        let matrix = theta.matrix();
        assert!((matrix[(0, 0)] - 0.1).abs() < 1e-10); // First row, ke value
        assert!((matrix[(0, 1)] - 10.0).abs() < 1e-10); // First row, v value

        cleanup_temp_file(&temp_path);
    }

    #[test]
    fn test_sample_space_file_based() {
        let csv_content = "ke,v\n0.1,10.0\n0.2,15.0\n0.3,20.0\n";
        let temp_path = create_temp_csv_file(csv_content);

        let mut settings = create_test_settings();
        settings.set_prior(Prior::File(temp_path.clone()));

        let result = sample_space(&settings);
        assert!(result.is_ok());

        let theta = result.unwrap();
        assert_eq!(theta.nspp(), 3);
        assert_eq!(theta.matrix().ncols(), 2);

        cleanup_temp_file(&temp_path);
    }

    #[test]
    fn test_prior_theta_no_seed_panic() {
        let parameters = Parameters::new().add("ke", 0.1, 1.0);
        let matrix = faer::Mat::from_fn(1, 1, |_, _| 0.5);
        let theta = Theta::from_parts(matrix, parameters).unwrap();
        let prior = Prior::Theta(theta);

        assert_eq!(prior.seed(), None, "Theta prior should not have a seed");
    }
}
