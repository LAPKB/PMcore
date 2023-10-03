use std::fs::File;

use ndarray::Array2;

use crate::prelude::settings::run::Data;

pub mod sobol;

pub fn sample_space(settings: &Data, ranges: &Vec<(f64, f64)>) -> Array2<f64> {
    match &settings.parsed.paths.prior_dist {
        Some(prior_path) => {
            let file = File::open(prior_path).unwrap();
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(true)
                .from_reader(file);

            let mut parameter_names: Vec<String> = reader
                .headers()
                .unwrap()
                .clone()
                .into_iter()
                .map(|s| s.trim().to_owned())
                .collect();

            // Remove "prob" column if present
            if let Some(index) = parameter_names.iter().position(|name| name == "prob") {
                parameter_names.remove(index);
            }

            // Check and reorder parameters to match names in settings.parsed.random
            let random_names: Vec<String> = settings
                .parsed
                .random
                .iter()
                .map(|(name, _)| name.clone())
                .collect();

            let mut reordered_indices: Vec<usize> = Vec::new();
            for random_name in &random_names {
                match parameter_names.iter().position(|name| name == random_name) {
                    Some(index) => {
                        reordered_indices.push(index);
                    }
                    None => {
                        panic!("Parameter {} is not present in the CSV file.", random_name);
                    }
                }
            }

            // Check if there are remaining parameters not present in settings.parsed.random
            if parameter_names.len() > random_names.len() {
                let extra_parameters: Vec<&String> = parameter_names.iter().collect();
                panic!(
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

            Array2::from_shape_vec((n_points, n_params), theta_values)
                .expect("Failed to create theta Array2")
        }
        None => sobol::generate(
            settings.parsed.config.init_points,
            ranges,
            settings.parsed.config.seed,
        ),
    }
}
