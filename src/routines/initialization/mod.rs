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

    pub fn get_points(&self) -> usize {
        match self {
            Prior::Sobol(points, _) => *points,
            Prior::Latin(points, _) => *points,
            Prior::File(_) => {
                unimplemented!("File-based prior does not have a fixed number of points")
            }
            Prior::Theta(theta) => theta.nspp(),
        }
    }

    pub fn get_seed(&self) -> usize {
        match self {
            Prior::Sobol(_, seed) => *seed,
            Prior::Latin(_, seed) => *seed,
            Prior::File(_) => unimplemented!("File-based prior does not have a fixed seed"),
            Prior::Theta(_) => {
                unimplemented!("Custom prior does not have a fixed seed")
            }
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

    let theta = Theta::from_parts(theta_matrix, settings.parameters().clone());

    Ok(theta)
}
