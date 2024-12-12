use std::fs::File;

use crate::prelude::data::Data;
use crate::prelude::simulator::Equation;
use anyhow::{bail, Context, Result};
use ndarray::Array2;
use pharmsol::{prelude::EstimateTheta, SupportPoint};

use crate::prelude::settings::Settings;

pub mod latin;
pub mod sobol;

/// This function generates the grid of support points according to the sampler specified in the [Settings]
pub fn sample_space(settings: &Settings, data: &Data, eqn: &impl Equation) -> Result<Array2<f64>> {
    // Get the ranges of the random parameters
    let ranges = settings.random.ranges();
    let parameters = settings.random.names();

    // If a prior file is provided, read it and return
    if settings.prior.file.is_some() {
        let prior = parse_prior(
            settings.prior.file.as_ref().unwrap(),
            &settings.random.names(),
        )?;
        return Ok(prior);
    }

    // Otherwise, parse the sampler type and generate the grid
    let prior = match settings.prior.sampler.as_str() {
        "sobol" => sobol::generate(settings.prior.points, &ranges, settings.prior.seed)?,
        "latin" => latin::generate(settings.prior.points, &ranges, settings.prior.seed)?,
        "osat" => {
            let mut point = vec![];
            for range in ranges {
                point.push((range.1 - range.0) / 2.0);
            }
            let spp = SupportPoint::from_vec(point, parameters);
            data.estimate_theta(eqn, &spp)
        }
        _ => {
            bail!(
                "Unknown sampler specified in settings: {}",
                settings.prior.sampler
            );
        }
    };
    Ok(prior)
}

/// This function reads the prior distribution from a file
pub fn parse_prior(path: &String, names: &Vec<String>) -> Result<Array2<f64>> {
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
    let random_names: Vec<String> = names.clone();

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

    Ok(Array2::from_shape_vec((n_points, n_params), theta_values)?)
}
