//! Individual parameter estimates
//!
//! This module defines structures for individual-level parameter estimates,
//! commonly known as Empirical Bayes Estimates (EBEs) or conditional modes.
//!
//! # Mathematical Background
//!
//! In mixed-effects models, the individual parameter vector for subject i is:
//!
//! ```text
//! ψᵢ = g(θ, ηᵢ)
//! ```
//!
//! where:
//! - `θ`: Fixed effects (population parameters)
//! - `ηᵢ`: Random effects for subject i
//!
//! The individual estimates are typically obtained by:
//! - **MAP estimation**: Maximizing p(ηᵢ | yᵢ, θ) ∝ p(yᵢ | ηᵢ, θ) × p(ηᵢ | θ)
//! - **MCMC sampling**: Drawing samples from the conditional distribution p(ηᵢ | yᵢ, θ)

use anyhow::{bail, Result};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};

/// Individual parameter estimates for a single subject
///
/// Contains the estimated random effects (η) and optionally the
/// uncertainty around these estimates (e.g., from Laplacian approximation).
#[derive(Debug, Clone)]
pub struct Individual {
    /// Subject identifier
    subject_id: String,
    /// Estimated random effects (η̂)
    eta: Col<f64>,
    /// Individual parameter values (ψ = g(θ, η))
    /// These are the actual parameter values used in simulation
    psi: Col<f64>,
    /// Conditional variance-covariance of η (from Hessian or MCMC)
    /// This represents the uncertainty in the individual estimates
    conditional_variance: Option<Mat<f64>>,
    /// Individual objective function value: -2 × log(p(yᵢ | ψᵢ) × p(ηᵢ))
    objective_function: Option<f64>,
}

impl Individual {
    /// Create a new individual estimate
    ///
    /// # Arguments
    ///
    /// * `subject_id` - Unique identifier for the subject
    /// * `eta` - Estimated random effects
    /// * `psi` - Individual parameter values
    pub fn new(subject_id: impl Into<String>, eta: Col<f64>, psi: Col<f64>) -> Result<Self> {
        if eta.nrows() != psi.nrows() {
            bail!(
                "Random effects length ({}) must match parameter length ({})",
                eta.nrows(),
                psi.nrows()
            );
        }

        Ok(Self {
            subject_id: subject_id.into(),
            eta,
            psi,
            conditional_variance: None,
            objective_function: None,
        })
    }

    /// Create an individual estimate with variance information
    pub fn with_variance(
        subject_id: impl Into<String>,
        eta: Col<f64>,
        psi: Col<f64>,
        variance: Mat<f64>,
    ) -> Result<Self> {
        let n = eta.nrows();
        if variance.nrows() != n || variance.ncols() != n {
            bail!(
                "Variance matrix dimensions ({}x{}) must match parameter count ({})",
                variance.nrows(),
                variance.ncols(),
                n
            );
        }

        let mut individual = Self::new(subject_id, eta, psi)?;
        individual.conditional_variance = Some(variance);
        Ok(individual)
    }

    // ========== Getters ==========

    /// Get the subject identifier
    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    /// Get the estimated random effects (η̂)
    pub fn eta(&self) -> &Col<f64> {
        &self.eta
    }

    /// Get the individual parameter values (ψ)
    pub fn psi(&self) -> &Col<f64> {
        &self.psi
    }

    /// Get the conditional variance if available
    pub fn conditional_variance(&self) -> Option<&Mat<f64>> {
        self.conditional_variance.as_ref()
    }

    /// Get the individual objective function value
    pub fn objective_function(&self) -> Option<f64> {
        self.objective_function
    }

    /// Get the number of parameters
    pub fn npar(&self) -> usize {
        self.eta.nrows()
    }

    /// Get the standard errors of η estimates (sqrt of diagonal of variance)
    pub fn standard_errors(&self) -> Option<Col<f64>> {
        self.conditional_variance.as_ref().map(|var| {
            Col::from_fn(self.npar(), |i| var[(i, i)].sqrt())
        })
    }

    // ========== Setters ==========

    /// Set the conditional variance
    pub fn set_conditional_variance(&mut self, variance: Mat<f64>) -> Result<()> {
        let n = self.npar();
        if variance.nrows() != n || variance.ncols() != n {
            bail!(
                "Variance matrix dimensions ({}x{}) must match parameter count ({})",
                variance.nrows(),
                variance.ncols(),
                n
            );
        }
        self.conditional_variance = Some(variance);
        Ok(())
    }

    /// Set the objective function value
    pub fn set_objective_function(&mut self, objf: f64) {
        self.objective_function = Some(objf);
    }
}

/// Collection of individual estimates for all subjects
///
/// This structure holds the individual parameter estimates for an entire population,
/// along with aggregate statistics and diagnostics.
#[derive(Debug, Clone, Default)]
pub struct IndividualEstimates {
    /// Individual estimates indexed by subject
    estimates: Vec<Individual>,
}

impl IndividualEstimates {
    /// Create a new empty collection
    pub fn new() -> Self {
        Self {
            estimates: Vec::new(),
        }
    }

    /// Create from a vector of individual estimates
    pub fn from_vec(estimates: Vec<Individual>) -> Self {
        Self { estimates }
    }

    /// Add an individual estimate
    pub fn add(&mut self, individual: Individual) {
        self.estimates.push(individual);
    }

    /// Get the number of subjects
    pub fn nsubjects(&self) -> usize {
        self.estimates.len()
    }

    /// Get an individual estimate by index
    pub fn get(&self, index: usize) -> Option<&Individual> {
        self.estimates.get(index)
    }

    /// Get an individual estimate by subject ID
    pub fn get_by_id(&self, id: &str) -> Option<&Individual> {
        self.estimates.iter().find(|e| e.subject_id() == id)
    }

    /// Iterate over all individual estimates
    pub fn iter(&self) -> impl Iterator<Item = &Individual> {
        self.estimates.iter()
    }

    /// Get all η estimates as a matrix (subjects × parameters)
    pub fn eta_matrix(&self) -> Option<Mat<f64>> {
        if self.estimates.is_empty() {
            return None;
        }

        let n_subjects = self.estimates.len();
        let n_params = self.estimates[0].npar();

        let matrix = Mat::from_fn(n_subjects, n_params, |i, j| self.estimates[i].eta()[j]);

        Some(matrix)
    }

    /// Get all ψ estimates as a matrix (subjects × parameters)
    pub fn psi_matrix(&self) -> Option<Mat<f64>> {
        if self.estimates.is_empty() {
            return None;
        }

        let n_subjects = self.estimates.len();
        let n_params = self.estimates[0].npar();

        let matrix = Mat::from_fn(n_subjects, n_params, |i, j| self.estimates[i].psi()[j]);

        Some(matrix)
    }

    /// Calculate the empirical mean of η across subjects
    pub fn eta_mean(&self) -> Option<Col<f64>> {
        if self.estimates.is_empty() {
            return None;
        }

        let n_subjects = self.estimates.len() as f64;
        let n_params = self.estimates[0].npar();

        let mean = Col::from_fn(n_params, |j| {
            self.estimates.iter().map(|e| e.eta()[j]).sum::<f64>() / n_subjects
        });

        Some(mean)
    }

    /// Calculate the empirical covariance of η across subjects
    pub fn eta_covariance(&self) -> Option<Mat<f64>> {
        let mean = self.eta_mean()?;
        let n_subjects = self.estimates.len() as f64;
        let n_params = self.estimates[0].npar();

        let cov = Mat::from_fn(n_params, n_params, |i, j| {
            self.estimates
                .iter()
                .map(|e| (e.eta()[i] - mean[i]) * (e.eta()[j] - mean[j]))
                .sum::<f64>()
                / (n_subjects - 1.0) // Using n-1 for unbiased estimate
        });

        Some(cov)
    }

    /// Calculate shrinkage for each parameter
    ///
    /// Shrinkage measures how much the individual estimates are "shrunk" toward
    /// the population mean due to limited individual information.
    ///
    /// Shrinkage = 1 - var(η̂) / ω² where ω² is the population variance
    pub fn shrinkage(&self, population_variance: &Col<f64>) -> Option<Col<f64>> {
        let eta_cov = self.eta_covariance()?;
        let n_params = self.estimates[0].npar();

        let shrinkage = Col::from_fn(n_params, |i| {
            let eta_var = eta_cov[(i, i)];
            let pop_var = population_variance[i];
            if pop_var > 0.0 {
                1.0 - (eta_var / pop_var)
            } else {
                0.0
            }
        });

        Some(shrinkage)
    }
}

impl Serialize for Individual {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("Individual", 4)?;
        state.serialize_field("subject_id", &self.subject_id)?;

        // Serialize eta as Vec<f64>
        let eta_vec: Vec<f64> = (0..self.eta.nrows()).map(|i| self.eta[i]).collect();
        state.serialize_field("eta", &eta_vec)?;

        // Serialize psi as Vec<f64>
        let psi_vec: Vec<f64> = (0..self.psi.nrows()).map(|i| self.psi[i]).collect();
        state.serialize_field("psi", &psi_vec)?;

        state.serialize_field("objective_function", &self.objective_function)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for Individual {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct IndividualData {
            subject_id: String,
            eta: Vec<f64>,
            psi: Vec<f64>,
            objective_function: Option<f64>,
        }

        let data = IndividualData::deserialize(deserializer)?;

        let eta = Col::from_fn(data.eta.len(), |i| data.eta[i]);
        let psi = Col::from_fn(data.psi.len(), |i| data.psi[i]);

        let mut individual = Individual::new(data.subject_id, eta, psi)
            .map_err(serde::de::Error::custom)?;

        individual.objective_function = data.objective_function;

        Ok(individual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_creation() {
        let eta = Col::from_fn(2, |i| if i == 0 { 0.1 } else { -0.2 });
        let psi = Col::from_fn(2, |i| if i == 0 { 5.5 } else { 45.0 });

        let ind = Individual::new("SUBJ001", eta, psi).unwrap();

        assert_eq!(ind.subject_id(), "SUBJ001");
        assert_eq!(ind.npar(), 2);
        assert_eq!(ind.eta()[0], 0.1);
        assert_eq!(ind.psi()[1], 45.0);
    }

    #[test]
    fn test_individual_estimates_collection() {
        let mut estimates = IndividualEstimates::new();

        for i in 0..3 {
            let eta = Col::from_fn(2, |j| (i as f64) * 0.1 + (j as f64) * 0.05);
            let psi = Col::from_fn(2, |j| 5.0 + (i as f64) + (j as f64) * 10.0);
            let ind = Individual::new(format!("SUBJ{:03}", i), eta, psi).unwrap();
            estimates.add(ind);
        }

        assert_eq!(estimates.nsubjects(), 3);
        assert!(estimates.get_by_id("SUBJ001").is_some());
        assert!(estimates.eta_matrix().is_some());
    }
}
