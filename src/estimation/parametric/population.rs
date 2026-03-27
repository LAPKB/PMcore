//! Parametric population representation.

use anyhow::{bail, Result};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};

use crate::model::{ParameterDomain, ParameterSpace};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum CovarianceStructure {
    #[default]
    Full,
    Diagonal,
    BlockDiagonal(Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct Population {
    mu: Col<f64>,
    omega: Mat<f64>,
    parameters: ParameterSpace,
    structure: CovarianceStructure,
}

impl Population {
    pub fn new(
        mu: Col<f64>,
        omega: Mat<f64>,
        parameters: impl Into<ParameterSpace>,
    ) -> Result<Self> {
        let parameters = parameters.into();
        let n = mu.nrows();

        if omega.nrows() != omega.ncols() {
            bail!(
                "Covariance matrix must be square, got {}x{}",
                omega.nrows(),
                omega.ncols()
            );
        }

        if omega.nrows() != n {
            bail!(
                "Covariance matrix dimension ({}) must match mean vector length ({})",
                omega.nrows(),
                n
            );
        }

        if parameters.len() != n {
            bail!(
                "Number of parameters ({}) must match mean vector length ({})",
                parameters.len(),
                n
            );
        }

        Ok(Self {
            mu,
            omega,
            parameters,
            structure: CovarianceStructure::Full,
        })
    }

    pub fn new_diagonal(
        mu: Col<f64>,
        variances: Col<f64>,
        parameters: impl Into<ParameterSpace>,
    ) -> Result<Self> {
        let n = mu.nrows();

        if variances.nrows() != n {
            bail!(
                "Variances length ({}) must match mean vector length ({})",
                variances.nrows(),
                n
            );
        }

        let omega = Mat::from_fn(n, n, |i, j| if i == j { variances[i] } else { 0.0 });

        let mut pop = Self::new(mu, omega, parameters)?;
        pop.structure = CovarianceStructure::Diagonal;
        Ok(pop)
    }

    pub fn from_parameter_space(parameters: impl Into<ParameterSpace>) -> Result<Self> {
        let parameters = parameters.into();
        let n = parameters.len();

        if n == 0 {
            bail!("Cannot create population with zero parameters");
        }

        let mu = Col::from_fn(n, |i| {
            let param = &parameters.iter().nth(i).unwrap();
            let (lower, upper) = parameter_bounds(param);
            (lower + upper) / 2.0
        });

        let omega = Mat::from_fn(n, n, |i, j| {
            if i == j {
                let param = &parameters.iter().nth(i).unwrap();
                let (lower, upper) = parameter_bounds(param);
                let range = upper - lower;
                (range / 4.0).powi(2)
            } else {
                0.0
            }
        });

        Self::new(mu, omega, parameters)
    }

    pub fn mu(&self) -> &Col<f64> {
        &self.mu
    }

    pub fn mu_mut(&mut self) -> &mut Col<f64> {
        &mut self.mu
    }

    pub fn omega(&self) -> &Mat<f64> {
        &self.omega
    }

    pub fn omega_mut(&mut self) -> &mut Mat<f64> {
        &mut self.omega
    }

    pub fn parameters(&self) -> &ParameterSpace {
        &self.parameters
    }

    pub fn npar(&self) -> usize {
        self.mu.nrows()
    }

    pub fn structure(&self) -> &CovarianceStructure {
        &self.structure
    }

    pub fn set_structure(&mut self, structure: CovarianceStructure) {
        self.structure = structure;
    }

    pub fn param_names(&self) -> Vec<String> {
        self.parameters.names()
    }

    pub fn standard_deviations(&self) -> Col<f64> {
        Col::from_fn(self.npar(), |i| self.omega[(i, i)].sqrt())
    }

    pub fn correlation_matrix(&self) -> Mat<f64> {
        let n = self.npar();
        let sds = self.standard_deviations();

        Mat::from_fn(n, n, |i, j| {
            if sds[i] > 0.0 && sds[j] > 0.0 {
                self.omega[(i, j)] / (sds[i] * sds[j])
            } else if i == j {
                1.0
            } else {
                0.0
            }
        })
    }

    pub fn coefficient_of_variation(&self) -> Col<f64> {
        Col::from_fn(self.npar(), |i| {
            let omega_ii = self.omega[(i, i)];
            ((omega_ii.exp() - 1.0).sqrt()) * 100.0
        })
    }

    pub fn update_from_sufficient_stats(
        &mut self,
        stats: &crate::estimation::parametric::SufficientStats,
    ) {
        let n = stats.count() as f64;

        for i in 0..self.npar() {
            self.mu[i] = stats.s1()[i] / n;
        }

        for i in 0..self.npar() {
            for j in 0..self.npar() {
                self.omega[(i, j)] = stats.s2()[(i, j)] / n - self.mu[i] * self.mu[j];
            }
        }

        self.apply_structure_constraint();
    }

    fn apply_structure_constraint(&mut self) {
        match &self.structure {
            CovarianceStructure::Full => {}
            CovarianceStructure::Diagonal => {
                for i in 0..self.npar() {
                    for j in 0..self.npar() {
                        if i != j {
                            self.omega[(i, j)] = 0.0;
                        }
                    }
                }
            }
            CovarianceStructure::BlockDiagonal(blocks) => {
                let mut current_start = 0;
                let mut block_ranges: Vec<(usize, usize)> = Vec::new();

                for &block_size in blocks {
                    block_ranges.push((current_start, current_start + block_size));
                    current_start += block_size;
                }

                for i in 0..self.npar() {
                    for j in 0..self.npar() {
                        let in_same_block = block_ranges
                            .iter()
                            .any(|&(start, end)| i >= start && i < end && j >= start && j < end);

                        if !in_same_block {
                            self.omega[(i, j)] = 0.0;
                        }
                    }
                }
            }
        }
    }

    pub fn update_mu(&mut self, mu: Col<f64>) -> Result<()> {
        if mu.nrows() != self.npar() {
            bail!(
                "Mean vector length ({}) doesn't match population size ({})",
                mu.nrows(),
                self.npar()
            );
        }
        self.mu = mu;
        Ok(())
    }

    pub fn update_omega(&mut self, omega: Mat<f64>) -> Result<()> {
        let n = self.npar();
        if omega.nrows() != n || omega.ncols() != n {
            bail!(
                "Omega dimensions ({}x{}) don't match population size ({})",
                omega.nrows(),
                omega.ncols(),
                n
            );
        }
        self.omega = omega;
        self.apply_structure_constraint();
        Ok(())
    }

    pub fn mu_as_vec(&self) -> Vec<f64> {
        (0..self.npar()).map(|i| self.mu[i]).collect()
    }

    pub fn variances_as_vec(&self) -> Vec<f64> {
        (0..self.npar()).map(|i| self.omega[(i, i)]).collect()
    }
}

impl Default for Population {
    fn default() -> Self {
        Self {
            mu: Col::zeros(0),
            omega: Mat::zeros(0, 0),
            parameters: ParameterSpace::new(),
            structure: CovarianceStructure::Full,
        }
    }
}

pub(crate) fn ensure_positive_definite_covariance(omega: &Mat<f64>) -> Mat<f64> {
    let n = omega.nrows();
    let min_var = 1e-8;
    let mut result = omega.clone();

    for index in 0..n {
        if result[(index, index)] < min_var {
            result[(index, index)] = min_var;
        }
    }

    if result.llt(faer::Side::Lower).is_err() {
        let ridge = result
            .diagonal()
            .column_vector()
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            * 0.01
            + min_var;
        for index in 0..n {
            result[(index, index)] += ridge;
        }
        tracing::debug!("Added ridge {:.2e} to Omega diagonal to ensure PD", ridge);
    }

    result
}

fn parameter_bounds(parameter: &crate::model::ParameterSpec) -> (f64, f64) {
    match parameter.domain {
        ParameterDomain::Bounded { lower, upper } => (lower, upper),
        ParameterDomain::Positive { lower, upper } => {
            (lower.unwrap_or(0.0), upper.unwrap_or(1.0e6))
        }
        ParameterDomain::Unbounded { lower, upper } => {
            (lower.unwrap_or(-1.0e6), upper.unwrap_or(1.0e6))
        }
    }
}

impl Serialize for Population {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("Population", 4)?;

        let mu_vec: Vec<f64> = (0..self.mu.nrows()).map(|i| self.mu[i]).collect();
        state.serialize_field("mu", &mu_vec)?;

        let omega_vec: Vec<Vec<f64>> = (0..self.omega.nrows())
            .map(|i| {
                (0..self.omega.ncols())
                    .map(|j| self.omega[(i, j)])
                    .collect()
            })
            .collect();
        state.serialize_field("omega", &omega_vec)?;

        state.serialize_field("parameters", &self.parameters)?;
        state.serialize_field("structure", &self.structure)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_creation() {
        let params = ParameterSpace::new()
            .add(crate::model::ParameterSpec::bounded("CL", 0.1, 10.0))
            .add(crate::model::ParameterSpec::bounded("V", 1.0, 100.0));

        let mu = Col::from_fn(2, |i| if i == 0 { 5.0 } else { 50.0 });
        let omega = Mat::from_fn(2, 2, |i, j| if i == j { 0.1 } else { 0.0 });

        let pop = Population::new(mu, omega, params).unwrap();

        assert_eq!(pop.npar(), 2);
        assert_eq!(pop.mu()[0], 5.0);
        assert_eq!(pop.omega()[(0, 0)], 0.1);
    }

    #[test]
    fn test_from_parameter_space() {
        let params = ParameterSpace::new()
            .add(crate::model::ParameterSpec::bounded("CL", 0.0, 10.0))
            .add(crate::model::ParameterSpec::bounded("V", 0.0, 100.0));

        let pop = Population::from_parameter_space(params).unwrap();

        assert_eq!(pop.mu()[0], 5.0);
        assert_eq!(pop.mu()[1], 50.0);
    }

    #[test]
    fn test_diagonal_structure() {
        let params = ParameterSpace::new()
            .add(crate::model::ParameterSpec::bounded("CL", 0.1, 10.0))
            .add(crate::model::ParameterSpec::bounded("V", 1.0, 100.0));

        let mu = Col::from_fn(2, |_| 1.0);
        let variances = Col::from_fn(2, |_| 0.1);

        let pop = Population::new_diagonal(mu, variances, params).unwrap();

        assert_eq!(*pop.structure(), CovarianceStructure::Diagonal);
        assert_eq!(pop.omega()[(0, 1)], 0.0);
        assert_eq!(pop.omega()[(1, 0)], 0.0);
    }
}
