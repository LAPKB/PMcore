use std::fs;
use std::path::Path;

use crate::routines::math::logsumexp_rows;
use crate::routines::output::NPResult;
use crate::routines::settings::Settings;
use crate::structs::psi::Psi;
use crate::structs::theta::Theta;
use anyhow::Context;
use anyhow::Result;
use faer_ext::IntoNdarray;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use npag::*;
use npod::NPOD;
use pharmsol::prelude::{data::Data, simulator::Equation};
use pharmsol::{Predictions, Subject};
use postprob::POSTPROB;
use serde::{Deserialize, Serialize};

pub mod npag;
pub mod npod;
pub mod postprob;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Algorithm {
    NPAG,
    NPOD,
    POSTPROB,
}

pub trait Algorithms<E: Equation + Send + 'static>: Sync + Send + 'static {
    fn new(config: Settings, equation: E, data: Data) -> Result<Box<Self>>
    where
        Self: Sized;
    fn validate_psi(&mut self) -> Result<()> {
        // Count problematic values in psi
        let mut nan_count = 0;
        let mut inf_count = 0;
        let is_log_space = match self.psi().space() {
            crate::structs::psi::Space::Linear => false,
            crate::structs::psi::Space::Log => true,
        };

        let psi = self.psi().matrix().as_ref().into_ndarray();
        // First coerce all NaN and infinite in psi to 0.0 (or NEG_INFINITY for log-space)
        for i in 0..psi.nrows() {
            for j in 0..self.psi().matrix().ncols() {
                let val = psi.get((i, j)).unwrap();
                if val.is_nan() {
                    nan_count += 1;
                } else if val.is_infinite() {
                    // In log-space, NEG_INFINITY is valid (represents zero probability)
                    // Only count positive infinity as problematic
                    if !is_log_space || val.is_sign_positive() {
                        inf_count += 1;
                    }
                }
            }
        }

        if nan_count + inf_count > 0 {
            tracing::warn!(
                "Psi matrix contains {} NaN, {} problematic Infinite values of {} total values",
                nan_count,
                inf_count,
                psi.ncols() * psi.nrows()
            );
        }

        // Calculate row sums: for regular space: sum; for log-space: logsumexp
        let plam: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = if is_log_space {
            // For log-space, use logsumexp for each row
            Array::from_vec(logsumexp_rows(psi.nrows(), psi.ncols(), |i, j| psi[(i, j)]))
        } else {
            // For regular space, sum each row
            let (_, col) = psi.dim();
            let ecol: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::ones(col);
            psi.dot(&ecol)
        };

        // Check for subjects with zero probability
        // In log-space: -inf means zero probability
        // In regular space: 0 means zero probability
        let w: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = if is_log_space {
            // For log-space, check if logsumexp result is -inf
            Array::from_shape_fn(plam.len(), |i| {
                if plam[i].is_infinite() && plam[i].is_sign_negative() {
                    f64::INFINITY // Will be flagged as problematic
                } else {
                    1.0 // Valid
                }
            })
        } else {
            1. / &plam
        };

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

                let error_model = self.settings().errormodels().clone();

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
                            .simulate_subject(subject[*index], &support_point, Some(&error_model))
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
                        .map(|x| x.state().clone())
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

    fn settings(&self) -> &Settings;
    /// Get the equation used in the algorithm
    fn equation(&self) -> &E;
    /// Get the data used in the algorithm
    fn data(&self) -> &Data;
    fn get_prior(&self) -> Theta;
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
        self.set_theta(self.get_prior());
        Ok(())
    }
    fn estimation(&mut self) -> Result<()>;
    /// Performs condensation of [Theta] and updates [Psi]
    ///
    /// This step reduces the number of support points in [Theta] based on the current weights,
    /// and updates the [Psi] matrix accordingly to reflect the new set of support points.
    /// It is typically performed after the estimation step in each cycle of the algorithm.
    fn condensation(&mut self) -> Result<()>;

    /// Performs optimizations on the current [ErrorModels] and updates [Psi] accordingly
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
    fn fit(&mut self) -> Result<NPResult<E>> {
        self.initialize().unwrap();
        loop {
            match self.next_cycle()? {
                Status::Continue => continue,
                Status::Stop(_) => break,
            }
        }
        Ok(self.into_npresult()?)
    }

    #[allow(clippy::wrong_self_convention)]
    fn into_npresult(&self) -> Result<NPResult<E>>;
}

pub fn dispatch_algorithm<E: Equation + Send + 'static>(
    settings: Settings,
    equation: E,
    data: Data,
) -> Result<Box<dyn Algorithms<E>>> {
    match settings.config().algorithm {
        Algorithm::NPAG => Ok(NPAG::new(settings, equation, data)?),
        Algorithm::NPOD => Ok(NPOD::new(settings, equation, data)?),
        Algorithm::POSTPROB => Ok(POSTPROB::new(settings, equation, data)?),
    }
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
