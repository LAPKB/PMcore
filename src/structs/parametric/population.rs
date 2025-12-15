//! Parametric population representation
//!
//! This module defines the [`Population`] struct which represents a parametric
//! population distribution using a mean vector (μ) and covariance matrix (Ω).
//!
//! # Mathematical Background
//!
//! In parametric mixed-effects models, the population distribution of random effects
//! is typically assumed to follow a multivariate normal distribution:
//!
//! ```text
//! η ~ N(μ, Ω)
//! ```
//!
//! where:
//! - `μ` (mu): Population mean vector of the random effects
//! - `Ω` (Omega): Between-subject variability covariance matrix
//!
//! # Covariance Structure
//!
//! The covariance matrix can have different structures depending on the model:
//! - **Full**: All elements estimated (most flexible, most parameters)
//! - **Diagonal**: Only variances estimated, covariances assumed zero
//! - **Block-diagonal**: Parameters grouped into correlated blocks

use anyhow::{bail, Result};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};

use crate::prelude::Parameters;

/// Represents the covariance structure of the population distribution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum CovarianceStructure {
    /// Full covariance matrix - all elements estimated
    #[default]
    Full,
    /// Diagonal covariance - only variances, no covariances
    Diagonal,
    /// Block diagonal - parameters grouped into correlated blocks
    /// The vector contains the sizes of each block
    BlockDiagonal(Vec<usize>),
}

/// Parametric population distribution parameters
///
/// Represents the population distribution as a multivariate normal distribution
/// parameterized by a mean vector (μ) and covariance matrix (Ω).
///
/// # Example
///
/// ```ignore
/// use pmcore::structs::parametric::Population;
/// use faer::{Col, Mat};
///
/// // Create a 2-parameter population with initial values
/// let mu = Col::from_fn(2, |i| if i == 0 { 1.0 } else { 0.5 });
/// let omega = Mat::from_fn(2, 2, |i, j| if i == j { 0.1 } else { 0.0 });
///
/// let pop = Population::new(mu, omega, parameters)?;
/// ```
#[derive(Debug, Clone)]
pub struct Population {
    /// Population mean vector (μ) - fixed effects or typical values
    mu: Col<f64>,
    /// Between-subject variability covariance matrix (Ω)
    omega: Mat<f64>,
    /// Parameter metadata (names, bounds, etc.)
    parameters: Parameters,
    /// Covariance structure constraint
    structure: CovarianceStructure,
}

impl Population {
    /// Create a new parametric population from mean and covariance
    ///
    /// # Arguments
    ///
    /// * `mu` - Population mean vector
    /// * `omega` - Covariance matrix (must be symmetric positive semi-definite)
    /// * `parameters` - Parameter definitions
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Dimensions are inconsistent
    /// - Omega is not square
    /// - Mu length doesn't match omega dimensions
    pub fn new(mu: Col<f64>, omega: Mat<f64>, parameters: Parameters) -> Result<Self> {
        // Validate dimensions
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

    /// Create a population with diagonal covariance structure
    pub fn new_diagonal(mu: Col<f64>, variances: Col<f64>, parameters: Parameters) -> Result<Self> {
        let n = mu.nrows();

        if variances.nrows() != n {
            bail!(
                "Variances length ({}) must match mean vector length ({})",
                variances.nrows(),
                n
            );
        }

        // Create diagonal covariance matrix
        let omega = Mat::from_fn(n, n, |i, j| {
            if i == j {
                variances[i]
            } else {
                0.0
            }
        });

        let mut pop = Self::new(mu, omega, parameters)?;
        pop.structure = CovarianceStructure::Diagonal;
        Ok(pop)
    }

    /// Create an initial population from parameter bounds
    ///
    /// Initializes μ at the midpoint of bounds and Ω with reasonable default variances
    /// based on the parameter ranges.
    pub fn from_parameters(parameters: Parameters) -> Result<Self> {
        let n = parameters.len();

        if n == 0 {
            bail!("Cannot create population with zero parameters");
        }

        // Initialize μ at midpoint of bounds
        let mu = Col::from_fn(n, |i| {
            let param = &parameters.iter().nth(i).unwrap();
            (param.lower + param.upper) / 2.0
        });

        // Initialize Ω with variances based on parameter ranges
        // Using (range/4)^2 as initial variance (assumes 95% of values within bounds)
        let omega = Mat::from_fn(n, n, |i, j| {
            if i == j {
                let param = &parameters.iter().nth(i).unwrap();
                let range = param.upper - param.lower;
                (range / 4.0).powi(2)
            } else {
                0.0
            }
        });

        Self::new(mu, omega, parameters)
    }

    // ========== Getters ==========

    /// Get the population mean vector (μ)
    pub fn mu(&self) -> &Col<f64> {
        &self.mu
    }

    /// Get a mutable reference to the mean vector
    pub fn mu_mut(&mut self) -> &mut Col<f64> {
        &mut self.mu
    }

    /// Get the covariance matrix (Ω)
    pub fn omega(&self) -> &Mat<f64> {
        &self.omega
    }

    /// Get a mutable reference to the covariance matrix
    pub fn omega_mut(&mut self) -> &mut Mat<f64> {
        &mut self.omega
    }

    /// Get the parameter definitions
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// Get the number of parameters
    pub fn npar(&self) -> usize {
        self.mu.nrows()
    }

    /// Get the covariance structure
    pub fn structure(&self) -> &CovarianceStructure {
        &self.structure
    }

    /// Set the covariance structure
    pub fn set_structure(&mut self, structure: CovarianceStructure) {
        self.structure = structure;
    }

    /// Get the parameter names
    pub fn param_names(&self) -> Vec<String> {
        self.parameters.names()
    }

    /// Get the standard deviations (square root of diagonal of Ω)
    pub fn standard_deviations(&self) -> Col<f64> {
        Col::from_fn(self.npar(), |i| self.omega[(i, i)].sqrt())
    }

    /// Get the correlation matrix derived from Ω
    pub fn correlation_matrix(&self) -> Mat<f64> {
        let n = self.npar();
        let sds = self.standard_deviations();

        Mat::from_fn(n, n, |i, j| {
            if sds[i] > 0.0 && sds[j] > 0.0 {
                self.omega[(i, j)] / (sds[i] * sds[j])
            } else {
                if i == j {
                    1.0
                } else {
                    0.0
                }
            }
        })
    }

    /// Get the coefficient of variation (CV%) for each parameter
    /// Assumes log-normal distribution: CV = sqrt(exp(omega_ii) - 1) * 100
    pub fn coefficient_of_variation(&self) -> Col<f64> {
        Col::from_fn(self.npar(), |i| {
            let omega_ii = self.omega[(i, i)];
            ((omega_ii.exp() - 1.0).sqrt()) * 100.0
        })
    }

    // ========== Update Methods ==========

    /// Update the population parameters from sufficient statistics
    ///
    /// This implements the M-step of the SAEM algorithm:
    /// - μ = S₁ / n
    /// - Ω = S₂ / n - μμᵀ
    pub fn update_from_sufficient_stats(&mut self, stats: &super::SufficientStats) {
        let n = stats.count() as f64;

        // Update μ = S₁ / n
        for i in 0..self.npar() {
            self.mu[i] = stats.s1()[i] / n;
        }

        // Update Ω = S₂ / n - μμᵀ
        for i in 0..self.npar() {
            for j in 0..self.npar() {
                self.omega[(i, j)] = stats.s2()[(i, j)] / n - self.mu[i] * self.mu[j];
            }
        }

        // Apply structure constraints if needed
        self.apply_structure_constraint();
    }

    /// Apply the covariance structure constraint to Ω
    fn apply_structure_constraint(&mut self) {
        match &self.structure {
            CovarianceStructure::Full => {
                // No constraint
            }
            CovarianceStructure::Diagonal => {
                // Zero out off-diagonal elements
                for i in 0..self.npar() {
                    for j in 0..self.npar() {
                        if i != j {
                            self.omega[(i, j)] = 0.0;
                        }
                    }
                }
            }
            CovarianceStructure::BlockDiagonal(blocks) => {
                // Zero out elements outside blocks
                let mut current_start = 0;
                let mut block_ranges: Vec<(usize, usize)> = Vec::new();

                for &block_size in blocks {
                    block_ranges.push((current_start, current_start + block_size));
                    current_start += block_size;
                }

                for i in 0..self.npar() {
                    for j in 0..self.npar() {
                        // Check if (i,j) is within any block
                        let in_same_block = block_ranges.iter().any(|&(start, end)| {
                            i >= start && i < end && j >= start && j < end
                        });

                        if !in_same_block {
                            self.omega[(i, j)] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

impl Default for Population {
    fn default() -> Self {
        Self {
            mu: Col::zeros(0),
            omega: Mat::zeros(0, 0),
            parameters: Parameters::new(),
            structure: CovarianceStructure::Full,
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

        // Serialize mu as Vec<f64>
        let mu_vec: Vec<f64> = (0..self.mu.nrows()).map(|i| self.mu[i]).collect();
        state.serialize_field("mu", &mu_vec)?;

        // Serialize omega as Vec<Vec<f64>>
        let omega_vec: Vec<Vec<f64>> = (0..self.omega.nrows())
            .map(|i| (0..self.omega.ncols()).map(|j| self.omega[(i, j)]).collect())
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
        let params = Parameters::new()
            .add("CL", 0.1, 10.0)
            .add("V", 1.0, 100.0);

        let mu = Col::from_fn(2, |i| if i == 0 { 5.0 } else { 50.0 });
        let omega = Mat::from_fn(2, 2, |i, j| if i == j { 0.1 } else { 0.0 });

        let pop = Population::new(mu, omega, params).unwrap();

        assert_eq!(pop.npar(), 2);
        assert_eq!(pop.mu()[0], 5.0);
        assert_eq!(pop.omega()[(0, 0)], 0.1);
    }

    #[test]
    fn test_from_parameters() {
        let params = Parameters::new()
            .add("CL", 0.0, 10.0)
            .add("V", 0.0, 100.0);

        let pop = Population::from_parameters(params).unwrap();

        // Check μ is at midpoint
        assert_eq!(pop.mu()[0], 5.0); // (0 + 10) / 2
        assert_eq!(pop.mu()[1], 50.0); // (0 + 100) / 2
    }

    #[test]
    fn test_diagonal_structure() {
        let params = Parameters::new()
            .add("CL", 0.1, 10.0)
            .add("V", 1.0, 100.0);

        let mu = Col::from_fn(2, |_| 1.0);
        let variances = Col::from_fn(2, |_| 0.1);

        let pop = Population::new_diagonal(mu, variances, params).unwrap();

        assert_eq!(*pop.structure(), CovarianceStructure::Diagonal);
        assert_eq!(pop.omega()[(0, 1)], 0.0);
        assert_eq!(pop.omega()[(1, 0)], 0.0);
    }
}
