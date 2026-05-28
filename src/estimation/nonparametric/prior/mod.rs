use std::fs::File;

use crate::estimation::nonparametric::{Theta, Weights};
use crate::model::NonParametricParameters;
use anyhow::{bail, Context, Result};
use faer::Mat;
use serde::{Deserialize, Serialize};

pub mod latin;
pub mod sobol;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum Prior {
    /// Generate support points using a Sobol sequence
    Sobol(usize, usize),
    /// Generate support points using Latin Hypercube Sampling
    Latin(usize, usize),
    /// Use a predefined set of support points provided as a [Theta]
    ///
    /// Note that the parameters of the [Theta] must match the parameters of the estimation problem
    Theta(Theta),
}

impl Serialize for Prior {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        enum Ref {
            Sobol { points: usize, seed: usize },
            Latin { points: usize, seed: usize },
        }
        let wire = match self {
            Prior::Sobol(points, seed) => Ref::Sobol {
                points: *points,
                seed: *seed,
            },
            Prior::Latin(points, seed) => Ref::Latin {
                points: *points,
                seed: *seed,
            },

            Prior::Theta(_) => {
                return Err(serde::ser::Error::custom(
                    "Prior::Theta cannot be serialized",
                ))
            }
        };
        wire.serialize(serializer)
    }
}

impl Prior {
    pub fn sobol(points: usize, seed: usize) -> Prior {
        Prior::Sobol(points, seed)
    }

    pub fn points(&self) -> Option<usize> {
        match self {
            Prior::Sobol(points, _) => Some(*points),
            Prior::Latin(points, _) => Some(*points),
            Prior::Theta(theta) => Some(theta.nspp()),
        }
    }

    pub fn seed(&self) -> Option<usize> {
        match self {
            Prior::Sobol(_, seed) => Some(*seed),
            Prior::Latin(_, seed) => Some(*seed),
            Prior::Theta(_) => None,
        }
    }

    /// Generates the initial support points (theta) based on the specified prior configuration
    ///
    /// If a Prior::Theta is provided, it will be returned directly. For Sobol and Latin, the support points will be generated based on the number of points and seed.
    pub fn theta(&self, parameters: &NonParametricParameters) -> Result<Theta> {
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

        let prior = match self {
            Prior::Sobol(points, seed) => sobol::generate(parameters, *points, *seed)?,
            Prior::Latin(points, seed) => latin::generate(parameters, *points, *seed)?,
            Prior::Theta(theta) => theta.clone(),
        };

        Ok(prior)
    }
}

impl Default for Prior {
    fn default() -> Self {
        Prior::Sobol(2028, 22)
    }
}

pub fn read_prior(
    path: impl AsRef<str>,
    parameters: &NonParametricParameters,
) -> Result<(Theta, Option<Weights>)> {
    let path = path.as_ref().to_string();
    parse_prior_for_parameters(&path, &parameters)
}

pub(crate) fn parse_prior_for_parameters(
    path: &String,
    parameters: &NonParametricParameters,
) -> Result<(Theta, Option<Weights>)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BoundedParameter;
    use std::fs;

    fn parameters() -> NonParametricParameters {
        NonParametricParameters::new()
            .add(BoundedParameter::new("ke", 0.1, 1.0))
            .add(BoundedParameter::new("v", 5.0, 50.0))
    }

    fn temp_csv_path() -> String {
        format!("test_temp_prior_{}.csv", rand::random::<u32>())
    }

    #[test]
    fn sample_space_generates_expected_shape() {
        let theta = Prior::sobol(10, 42).theta(&parameters()).unwrap();
        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 2);
    }

    #[test]
    fn read_prior_parses_weights_and_reorders_columns() {
        let path = temp_csv_path();
        fs::write(&path, "v,ke,prob\n10.0,0.5,0.3\n15.0,0.7,0.7\n").unwrap();

        let (theta, weights) = read_prior(&path, &parameters()).unwrap();
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
    fn read_prior_rejects_extra_parameters() {
        let path = temp_csv_path();
        fs::write(&path, "ke,v,extra\n0.5,10.0,1.0\n").unwrap();

        let err = read_prior(&path, &parameters()).unwrap_err();
        let _ = fs::remove_file(&path);

        assert!(err
            .to_string()
            .contains("Found parameters in the prior not present in configuration"));
    }
}
