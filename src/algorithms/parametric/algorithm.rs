//! Parametric algorithm trait definition
//!
//! This module defines the [`ParametricAlgorithm`] trait that all parametric
//! population estimation algorithms must implement.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use pharmsol::{Data, Equation};

use crate::routines::output::ParametricResult;
use crate::routines::settings::Settings;
use crate::structs::parametric::{IndividualEstimates, Population, SufficientStats};

use super::super::Status;

/// Configuration specific to parametric algorithms
#[derive(Debug, Clone)]
pub struct ParametricConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Number of burn-in iterations (for SAEM)
    pub burn_in: usize,
    /// Number of MCMC chains per subject
    pub n_chains: usize,
    /// Number of samples per chain per iteration
    pub n_samples: usize,
    /// Convergence tolerance for parameters
    pub parameter_tolerance: f64,
    /// Convergence tolerance for objective function
    pub objective_tolerance: f64,
    /// Whether to use simulated annealing in SAEM
    pub use_annealing: bool,
    /// Initial temperature for simulated annealing
    pub initial_temperature: f64,
}

impl Default for ParametricConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            burn_in: 200,
            n_chains: 1,
            n_samples: 1,
            parameter_tolerance: 1e-4,
            objective_tolerance: 1e-4,
            use_annealing: false,
            initial_temperature: 1.0,
        }
    }
}

/// Trait defining the interface for parametric population algorithms
///
/// This trait provides the common structure for algorithms that estimate
/// population parameters assuming a continuous (typically multivariate normal)
/// distribution for the random effects.
///
/// # Algorithm Workflow
///
/// 1. **Initialize**: Set up initial population parameters
/// 2. **E-step**: Compute or sample from conditional distribution p(η|y,θ)
/// 3. **M-step**: Update population parameters from E-step results
/// 4. **Evaluate**: Check convergence criteria
/// 5. **Repeat** until convergence or max iterations
///
/// # Type Parameters
///
/// * `E` - The equation type implementing pharmacokinetic/pharmacodynamic model
pub trait ParametricAlgorithm<E: Equation + Send + 'static>: Sync + Send {
    /// Create a new instance of the algorithm
    ///
    /// # Arguments
    ///
    /// * `settings` - Algorithm configuration and settings
    /// * `equation` - The pharmacokinetic/pharmacodynamic model
    /// * `data` - Population data
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>>
    where
        Self: Sized;

    // ========== Accessors ==========

    /// Get the algorithm settings
    fn settings(&self) -> &Settings;

    /// Get the equation/model
    fn equation(&self) -> &E;

    /// Get the data
    fn data(&self) -> &Data;

    /// Get the current population parameters
    fn population(&self) -> &Population;

    /// Get a mutable reference to population parameters
    fn population_mut(&mut self) -> &mut Population;

    /// Get the current individual estimates
    fn individual_estimates(&self) -> &IndividualEstimates;

    /// Get the current iteration number
    fn iteration(&self) -> usize;

    /// Increment the iteration counter and return new value
    fn increment_iteration(&mut self) -> usize;

    /// Get the current objective function value (-2LL)
    fn objective_function(&self) -> f64;

    /// Get the current algorithm status
    fn status(&self) -> &Status;

    /// Set the algorithm status
    fn set_status(&mut self, status: Status);

    // ========== Algorithm Steps ==========

    /// Initialize the algorithm
    ///
    /// Sets up initial population parameters, prepares data structures,
    /// and performs any pre-processing required before the main loop.
    fn initialize(&mut self) -> Result<()> {
        // Remove stop file if it exists
        if Path::new("stop").exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file("stop").context("Unable to remove previous stop file")?;
        }
        self.set_status(Status::Continue);
        Ok(())
    }

    /// Perform the E-step (Expectation step)
    ///
    /// This step computes or samples from the conditional distribution of
    /// individual parameters given the observations and current population parameters.
    ///
    /// - **FOCE/Laplacian**: Finds the MAP estimate (mode) and Hessian
    /// - **SAEM**: Samples from p(η|y,θ) using MCMC
    fn e_step(&mut self) -> Result<()>;

    /// Perform the M-step (Maximization step)
    ///
    /// Updates the population parameters (μ, Ω) based on the E-step results.
    ///
    /// - **FOCE/Laplacian**: Uses the modes and Hessians
    /// - **SAEM**: Uses sufficient statistics from MCMC samples
    fn m_step(&mut self) -> Result<()>;

    /// Evaluate convergence and update status
    ///
    /// Checks various convergence criteria and determines whether to continue
    /// or stop the algorithm.
    fn evaluate(&mut self) -> Result<Status>;

    /// Log the current iteration state
    fn log_iteration(&mut self);

    /// Perform a single iteration of the algorithm
    ///
    /// Default implementation calls E-step, M-step, logging, and evaluation.
    fn next_iteration(&mut self) -> Result<Status> {
        let iter = self.increment_iteration();

        let span = tracing::info_span!("", "{}", format!("Iteration {}", iter));
        let _enter = span.enter();

        self.e_step()?;
        self.m_step()?;
        self.log_iteration();
        self.evaluate()
    }

    /// Run the full estimation procedure
    ///
    /// Initializes the algorithm and iterates until convergence or stopping criteria.
    fn fit(&mut self) -> Result<ParametricResult<E>> {
        self.initialize()?;

        loop {
            match self.next_iteration()? {
                Status::Continue => continue,
                Status::Stop(_) => break,
            }
        }

        self.into_result()
    }

    /// Convert the algorithm state into a result object
    fn into_result(&self) -> Result<ParametricResult<E>>;

    // ========== Optional Methods ==========

    /// Get sufficient statistics (for SAEM-like algorithms)
    fn sufficient_stats(&self) -> Option<&SufficientStats> {
        None
    }

    /// Perform optimization of error model parameters
    ///
    /// Some algorithms may optimize error model parameters alongside population parameters.
    fn optimize_error_model(&mut self) -> Result<()> {
        // Default: no optimization
        Ok(())
    }

    /// Apply constraints to population parameters
    ///
    /// Ensures parameters stay within bounds and covariance matrix remains positive definite.
    fn apply_constraints(&mut self) -> Result<()> {
        // Default: no additional constraints
        Ok(())
    }
}

/// Dispatch function for parametric algorithms
///
/// Creates the appropriate algorithm instance based on settings.
pub fn dispatch_parametric_algorithm<E: Equation + Clone + Send + 'static>(
    settings: Settings,
    equation: E,
    data: Data,
) -> Result<Box<dyn ParametricAlgorithm<E>>> {
    use crate::algorithms::Algorithm;
    use super::saem::FSAEM;

    match settings.config().algorithm {
        Algorithm::SAEM => {
            // Create f-SAEM using the trait's new method
            let saem = FSAEM::new(settings, equation, data)?;
            Ok(saem as Box<dyn ParametricAlgorithm<E>>)
        }
        Algorithm::FOCE | Algorithm::FOCEI => {
            // TODO: Implement FOCE
            anyhow::bail!("FOCE algorithm not yet implemented")
        }
        Algorithm::FO => {
            // TODO: Implement FO
            anyhow::bail!("FO algorithm not yet implemented")
        }
        Algorithm::Laplacian => {
            // TODO: Implement Laplacian
            anyhow::bail!("Laplacian algorithm not yet implemented")
        }
        Algorithm::IT2B => {
            // TODO: Implement IT2B
            anyhow::bail!("IT2B algorithm not yet implemented")
        }
        _ => anyhow::bail!("Algorithm {:?} is not a parametric algorithm", settings.config().algorithm),
    }
}
