use std::fs;
use std::path::Path;

use crate::api::{EstimationProblem, Framework};
use crate::estimation::nonparametric::{NonParametricResult, Psi, Theta};
use crate::results::FitResult;

use anyhow::Context;
use anyhow::Result;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};

use pharmsol::prelude::{data::Data, simulator::Equation};

use pharmsol::{Predictions, Subject};
use serde::{Deserialize, Serialize};

/// Defines an algorithm capable of building an execution engine
pub trait Algorithm<E: Equation, F: crate::api::Framework> {
    /// The strictly-typed runner struct (e.g., `NPAG<E>`)
    type Runner;

    /// Consumes the configuration and the problem to fully hydrate the runner state.
    fn build_runner(self, problem: EstimationProblem<E, F>) -> Result<Self::Runner>;
}

/// A trait for runners that can be executed to produce a fit result
pub trait Fitter<E: Equation> {
    /// The specific result struct (e.g., NonParametricResult<E>)
    type Output: FitResult;

    /// Executes the optimization and returns the strictly-typed metrics.
    fn fit(self) -> anyhow::Result<Self::Output>;
}

// Module organization for algorithm types
pub mod nonparametric;
pub mod parametric;

impl<E: Equation, F: Framework> EstimationProblem<E, F> {
    /// The "Swap and Fit" API:
    /// Consumes the problem and the algorithm configuration, builds the engine,
    /// and runs it to completion automatically.
    pub fn fit_with<A>(self, algorithm: A) -> anyhow::Result<<A::Runner as Fitter<E>>::Output>
    where
        A: Algorithm<E, F>,
        A::Runner: Fitter<E>,
    {
        algorithm.build_runner(self)?.fit()
    }
    /// The "Extract Structure" API:
    /// Consumes the problem and configuration, returning the strictly-typed
    /// execution runner (e.g., `NPAG<E>`) for advanced inspection.
    pub fn runner<A>(self, algorithm: A) -> Result<A::Runner>
    where
        A: Algorithm<E, F>,
    {
        algorithm.build_runner(self)
    }
}

pub trait NonParametricAlgorithm<E: Equation + Send + 'static>: Sync + Send + 'static {
    fn validate_psi(&mut self) -> Result<()> {
        // Count problematic values in psi
        let mut nan_count = 0;
        let mut inf_count = 0;

        let psi = self.psi().to_ndarray();
        // First coerce all NaN and infinite in psi to 0.0
        for i in 0..psi.nrows() {
            for j in 0..self.psi().matrix().ncols() {
                let val = psi.get((i, j)).unwrap();
                if val.is_nan() {
                    nan_count += 1;
                    // *val = 0.0;
                } else if val.is_infinite() {
                    inf_count += 1;
                    // *val = 0.0;
                }
            }
        }

        if nan_count + inf_count > 0 {
            tracing::warn!(
                "Psi matrix contains {} NaN, {} Infinite values of {} total values",
                nan_count,
                inf_count,
                psi.ncols() * psi.nrows()
            );
        }

        let (_, col) = psi.dim();
        let ecol: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::ones(col);
        let plam = psi.dot(&ecol);
        let w = 1. / &plam;

        // Get the index of each element in `w` that is NaN or infinite
        let indices: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_nan() || x.is_infinite())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        if !indices.is_empty() {
            let subject: Vec<&Subject> = self.data().subjects();
            let zero_probability_subjects: Vec<&String> =
                indices.iter().map(|&i| subject[i].id()).collect();

            tracing::error!(
                "{}/{} subjects have zero probability given the model",
                indices.len(),
                psi.nrows()
            );

            // For each problematic subject
            for index in &indices {
                tracing::debug!("Subject with zero probability: {}", subject[*index].id());

                let error_model = self.error_models().clone();

                // Simulate all support points in parallel
                let spp_results: Vec<_> = self
                    .theta()
                    .matrix()
                    .row_iter()
                    .enumerate()
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|(i, spp)| {
                        let support_point: Vec<f64> = spp.iter().copied().collect();
                        let (pred, ll) = self
                            .equation()
                            .simulate_subject_dense(
                                subject[*index],
                                &support_point,
                                Some(&error_model),
                            )
                            .unwrap(); //TODO: Handle error
                        (i, support_point, pred.get_predictions(), ll)
                    })
                    .collect();

                // Count problematic likelihoods for this subject
                let mut nan_ll = 0;
                let mut inf_pos_ll = 0;
                let mut inf_neg_ll = 0;
                let mut zero_ll = 0;
                let mut valid_ll = 0;

                for (_, _, _, ll) in &spp_results {
                    match ll {
                        Some(ll_val) if ll_val.is_nan() => nan_ll += 1,
                        Some(ll_val) if ll_val.is_infinite() && ll_val.is_sign_positive() => {
                            inf_pos_ll += 1
                        }
                        Some(ll_val) if ll_val.is_infinite() && ll_val.is_sign_negative() => {
                            inf_neg_ll += 1
                        }
                        Some(ll_val) if *ll_val == 0.0 => zero_ll += 1,
                        Some(_) => valid_ll += 1,
                        None => nan_ll += 1,
                    }
                }

                tracing::debug!(
                    "\tLikelihood analysis for subject {} ({} support points):",
                    subject[*index].id(),
                    spp_results.len()
                );
                tracing::debug!(
                    "\tNaN likelihoods: {} ({:.1}%)",
                    nan_ll,
                    100.0 * nan_ll as f64 / spp_results.len() as f64
                );
                tracing::debug!(
                    "\t+Inf likelihoods: {} ({:.1}%)",
                    inf_pos_ll,
                    100.0 * inf_pos_ll as f64 / spp_results.len() as f64
                );
                tracing::debug!(
                    "\t-Inf likelihoods: {} ({:.1}%)",
                    inf_neg_ll,
                    100.0 * inf_neg_ll as f64 / spp_results.len() as f64
                );
                tracing::debug!(
                    "\tZero likelihoods: {} ({:.1}%)",
                    zero_ll,
                    100.0 * zero_ll as f64 / spp_results.len() as f64
                );
                tracing::debug!(
                    "\tValid likelihoods: {} ({:.1}%)",
                    valid_ll,
                    100.0 * valid_ll as f64 / spp_results.len() as f64
                );

                // Sort and show top 10 most likely support points
                let mut sorted_results = spp_results;
                sorted_results.sort_by(|a, b| {
                    b.3.unwrap_or(f64::NEG_INFINITY)
                        .partial_cmp(&a.3.unwrap_or(f64::NEG_INFINITY))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let take = 3;

                tracing::debug!("Top {} most likely support points:", take);
                for (i, support_point, preds, ll) in sorted_results.iter().take(take) {
                    tracing::debug!("\tSupport point #{}: {:?}", i, support_point);
                    tracing::debug!("\t\tLog-likelihood: {:?}", ll);

                    let times = preds.iter().map(|x| x.time()).collect::<Vec<f64>>();
                    let observations = preds
                        .iter()
                        .map(|x| x.observation())
                        .collect::<Vec<Option<f64>>>();
                    let predictions = preds.iter().map(|x| x.prediction()).collect::<Vec<f64>>();
                    let outeqs = preds.iter().map(|x| x.outeq()).collect::<Vec<usize>>();
                    let states = preds
                        .iter()
                        .map(|x| x.state().to_vec())
                        .collect::<Vec<Vec<f64>>>();

                    tracing::debug!("\t\tTimes: {:?}", times);
                    tracing::debug!("\t\tObservations: {:?}", observations);
                    tracing::debug!("\t\tPredictions: {:?}", predictions);
                    tracing::debug!("\t\tOuteqs: {:?}", outeqs);
                    tracing::debug!("\t\tStates: {:?}", states);
                }
                tracing::debug!("=====================");
            }

            return Err(anyhow::anyhow!(
                    "The probability of {}/{} subjects is zero given the model. Affected subjects: {:?}",
                    indices.len(),
                    self.psi().matrix().nrows(),
                    zero_probability_subjects
                ));
        }

        Ok(())
    }

    fn error_models(&self) -> &pharmsol::prelude::data::AssayErrorModels;
    /// Get the equation used in the algorithm
    fn equation(&self) -> &E;
    /// Get the data used in the algorithm
    fn data(&self) -> &Data;

    /// Increment the cycle counter and return the new value
    fn increment_cycle(&mut self) -> usize;
    /// Get the current cycle number
    fn cycle(&self) -> usize;
    /// Set the current [Theta]
    fn set_theta(&mut self, theta: Theta);
    /// Get the current [Theta]
    fn theta(&self) -> &Theta;
    /// Get the current [Psi]
    fn psi(&self) -> &Psi;
    /// Get the current likelihood
    fn likelihood(&self) -> f64;
    /// Get the current negative two log-likelihood
    fn n2ll(&self) -> f64 {
        -2.0 * self.likelihood()
    }
    /// Get the current [Status] of the algorithm
    fn status(&self) -> &Status;
    /// Set the current [Status] of the algorithm
    fn set_status(&mut self, status: Status);
    /// Evaluate convergence criteria and update status
    fn evaluation(&mut self) -> Result<Status>;

    /// Create and log a cycle state with the current algorithm state
    fn log_cycle_state(&mut self);

    /// Initialize the algorithm, setting up initial [Theta] and [Status]
    fn initialize(&mut self) -> Result<()> {
        // If a stop file exists in the current directory, remove it
        if Path::new("stop").exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file("stop").context("Unable to remove previous stop file")?;
        }
        self.set_status(Status::Continue);

        Ok(())
    }
    fn estimation(&mut self) -> Result<()>;
    /// Performs condensation of [Theta] and updates [Psi]
    ///
    /// This step reduces the number of support points in [Theta] based on the current weights,
    /// and updates the [Psi] matrix accordingly to reflect the new set of support points.
    /// It is typically performed after the estimation step in each cycle of the algorithm.
    fn condensation(&mut self) -> Result<()>;

    /// Performs optimizations on the current [AssayErrorModels] and updates [Psi] accordingly
    ///
    /// This step refines the error model parameters to better fit the data,
    /// and subsequently updates the [Psi] matrix to reflect these changes.
    fn optimizations(&mut self) -> Result<()>;

    /// Performs expansion of [Theta]
    ///
    /// This step increases the number of support points in [Theta] based on the current distribution,
    /// allowing for exploration of the parameter space.
    fn expansion(&mut self) -> Result<()>;

    /// Proceed to the next cycle of the algorithm
    ///
    /// This method increments the cycle counter, performs expansion if necessary,
    /// and then runs the estimation, condensation, optimization, logging, and evaluation steps
    /// in sequence. It returns the current [Status] of the algorithm after completing these steps.
    fn next_cycle(&mut self) -> Result<Status> {
        let cycle = self.increment_cycle();

        if cycle > 1 {
            self.expansion()?;
        }

        let span = tracing::info_span!("", "{}", format!("Cycle {}", self.cycle()));
        let _enter = span.enter();
        self.estimation()?;
        self.condensation()?;
        self.optimizations()?;
        self.evaluation()
    }

    /// Fit the model until convergence or stopping criteria are met
    ///
    /// This method runs the full fitting process, starting with initialization,
    /// followed by iterative cycles of estimation, condensation, optimization, and evaluation
    /// until the algorithm converges or meets a stopping criteria.
    fn fit(&mut self) -> Result<NonParametricResult<E>> {
        self.initialize().unwrap();
        while let Status::Continue = self.next_cycle()? {}
        self.into_result()
    }

    #[allow(clippy::wrong_self_convention)]
    fn into_result(&self) -> Result<NonParametricResult<E>>;
}

/// Represents the status/result of the algorithm
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    Continue,
    Stop(StopReason),
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Continue => write!(f, "Continue"),
            Status::Stop(s) => write!(f, "Stop: {:?}", s),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]

pub enum StopReason {
    Converged,
    MaxCycles,
    Stopped,
    Completed,
}
