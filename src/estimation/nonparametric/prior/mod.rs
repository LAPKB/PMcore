use crate::estimation::nonparametric::Theta;
use crate::model::{BoundedParameter, ParameterSpace};
use anyhow::{bail, Result};

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
    pub fn theta(&self, parameters: &ParameterSpace<BoundedParameter>) -> Result<Theta> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Parameter;
    use std::fs;

    fn parameters() -> ParameterSpace<BoundedParameter> {
        ParameterSpace::<BoundedParameter>::new()
            .add(Parameter::bounded("ke", 0.1, 1.0))
            .add(Parameter::bounded("v", 5.0, 50.0))
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
    fn read_prior_rejects_extra_parameters() {
        let path = temp_csv_path();
        fs::write(&path, "ke,v,extra\n0.5,10.0,1.0\n").unwrap();

        let err = Theta::from_file(&path, &parameters()).unwrap_err();
        let _ = fs::remove_file(&path);

        assert!(err
            .to_string()
            .contains("Found parameters in the prior not present in configuration"));
    }
}
