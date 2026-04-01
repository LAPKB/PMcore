use std::fs::File;

use crate::estimation::nonparametric::{Theta, Weights};
use crate::model::{ParameterDomain, ParameterSpace};
use anyhow::{bail, Context, Result};
use faer::Mat;
use serde::{Deserialize, Serialize};

pub mod latin;
pub mod sobol;

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

    pub fn points(&self) -> Option<usize> {
        match self {
            Prior::Sobol(points, _) => Some(*points),
            Prior::Latin(points, _) => Some(*points),
            Prior::File(_) => None,
            Prior::Theta(theta) => Some(theta.nspp()),
        }
    }

    pub fn seed(&self) -> Option<usize> {
        match self {
            Prior::Sobol(_, seed) => Some(*seed),
            Prior::Latin(_, seed) => Some(*seed),
            Prior::File(_) => None,
            Prior::Theta(_) => None,
        }
    }
}

impl Default for Prior {
    fn default() -> Self {
        Prior::Sobol(2028, 22)
    }
}

pub fn read_prior(
    path: impl AsRef<str>,
    parameters: impl Into<ParameterSpace>,
) -> Result<(Theta, Option<Weights>)> {
    let path = path.as_ref().to_string();
    parse_prior_for_parameters(&path, parameters)
}

pub(crate) fn sample_space_for_parameters(
    parameters: impl Into<ParameterSpace>,
    prior: &Prior,
) -> Result<Theta> {
    let parameter_space = parameters.into();

    for parameter in parameter_space.iter() {
        let (lower, upper) = match parameter.domain {
            ParameterDomain::Bounded { lower, upper } => (lower, upper),
            ParameterDomain::Positive {
                lower: Some(lower),
                upper: Some(upper),
            }
            | ParameterDomain::Unbounded {
                lower: Some(lower),
                upper: Some(upper),
            } => (lower, upper),
            _ => bail!(
                "Parameter '{}' is missing finite bounds required for nonparametric initialization",
                parameter.name
            ),
        };

        if lower.is_infinite() || upper.is_infinite() {
            bail!(
                "Parameter '{}' has infinite bounds: [{}, {}]",
                parameter.name,
                lower,
                upper
            );
        }

        if lower >= upper {
            bail!(
                "Parameter '{}' has invalid bounds: [{}, {}]. Lower bound must be less than upper bound.",
                parameter.name,
                lower,
                upper
            );
        }
    }

    let prior = match prior {
        Prior::Sobol(points, seed) => sobol::generate(&parameter_space, *points, *seed)?,
        Prior::Latin(points, seed) => latin::generate(&parameter_space, *points, *seed)?,
        Prior::File(path) => parse_prior_for_parameters(path, &parameter_space)?.0,
        Prior::Theta(theta) => return Ok(theta.clone()),
    };
    Ok(prior)
}

pub(crate) fn parse_prior_for_parameters(
    path: &String,
    parameters: impl Into<ParameterSpace>,
) -> Result<(Theta, Option<Weights>)> {
    let parameters = parameters.into();
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
    use crate::model::{ParameterSpace, ParameterSpec};
    use std::fs;

    fn parameter_space() -> ParameterSpace {
        ParameterSpace::new()
            .add(ParameterSpec::bounded("ke", 0.1, 1.0))
            .add(ParameterSpec::bounded("v", 5.0, 50.0))
    }

    fn temp_csv_path() -> String {
        format!("test_temp_prior_{}.csv", rand::random::<u32>())
    }

    #[test]
    fn prior_metadata_accessors() {
        let sobol = Prior::sobol(100, 42);
        assert_eq!(sobol.points(), Some(100));
        assert_eq!(sobol.seed(), Some(42));

        let latin = Prior::Latin(50, 7);
        assert_eq!(latin.points(), Some(50));
        assert_eq!(latin.seed(), Some(7));

        let file = Prior::File("prior.csv".to_string());
        assert_eq!(file.points(), None);
        assert_eq!(file.seed(), None);
    }

    #[test]
    fn sample_space_generates_expected_shape() {
        let theta = sample_space_for_parameters(parameter_space(), &Prior::sobol(10, 42)).unwrap();
        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 2);
    }

    #[test]
    fn sample_space_returns_custom_theta_verbatim() {
        let parameters = parameter_space();
        let matrix = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
        let custom = Theta::from_parts(matrix, parameters).unwrap();

        let theta =
            sample_space_for_parameters(parameter_space(), &Prior::Theta(custom.clone())).unwrap();
        assert_eq!(theta.matrix(), custom.matrix());
    }

    #[test]
    fn read_prior_parses_weights_and_reorders_columns() {
        let path = temp_csv_path();
        fs::write(&path, "v,ke,prob\n10.0,0.5,0.3\n15.0,0.7,0.7\n").unwrap();

        let (theta, weights) = read_prior(&path, parameter_space()).unwrap();
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

        let err = read_prior(&path, parameter_space()).unwrap_err();
        let _ = fs::remove_file(&path);

        assert!(err
            .to_string()
            .contains("Found parameters in the prior not present in configuration"));
    }
}
