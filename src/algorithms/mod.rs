use std::fs;
use std::path::Path;

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

pub trait Algorithms<E: Equation>: Sync {
    fn new(config: Settings, equation: E, data: Data) -> Result<Box<Self>>
    where
        Self: Sized;
    fn validate_psi(&mut self) -> Result<()> {
        // Count problematic values in psi
        let mut nan_count = 0;
        let mut inf_count = 0;

        let psi = self.psi().matrix().as_ref().into_ndarray();
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
            let subject: Vec<&Subject> = self.get_data().subjects();
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

                let error_model = self.get_settings().errormodels().clone();

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
    fn get_settings(&self) -> &Settings;
    fn equation(&self) -> &E;
    fn get_data(&self) -> &Data;
    fn get_prior(&self) -> Theta;
    fn inc_cycle(&mut self) -> usize;
    fn get_cycle(&self) -> usize;
    fn set_theta(&mut self, theta: Theta);
    fn theta(&self) -> &Theta;
    fn psi(&self) -> &Psi;
    fn likelihood(&self) -> f64;
    fn n2ll(&self) -> f64 {
        -2.0 * self.likelihood()
    }
    fn status(&self) -> &Status;
    fn set_status(&mut self, status: Status);
    fn convergence_evaluation(&mut self);
    fn converged(&self) -> bool;
    fn initialize(&mut self) -> Result<()> {
        // If a stop file exists in the current directory, remove it
        if Path::new("stop").exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file("stop").context("Unable to remove previous stop file")?;
        }
        self.set_status(Status::InProgress);
        self.set_theta(self.get_prior());
        Ok(())
    }
    fn evaluation(&mut self) -> Result<()>;
    fn condensation(&mut self) -> Result<()>;
    fn optimizations(&mut self) -> Result<()>;
    fn logs(&self);
    fn expansion(&mut self) -> Result<()>;
    fn next_cycle(&mut self) -> Result<bool> {
        if self.inc_cycle() > 1 {
            self.expansion()?;
        }
        let span = tracing::info_span!("", "{}", format!("Cycle {}", self.get_cycle()));
        let _enter = span.enter();
        self.evaluation()?;
        self.condensation()?;
        self.optimizations()?;
        self.logs();
        self.convergence_evaluation();
        Ok(self.converged())
    }
    fn fit(&mut self) -> Result<NPResult<E>> {
        self.initialize().unwrap();
        while !self.next_cycle()? {}
        Ok(self.into_npresult())
    }

    #[allow(clippy::wrong_self_convention)]
    fn into_npresult(&self) -> NPResult<E>;
}

pub fn dispatch_algorithm<E: Equation>(
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

/// Represents the status of the algorithm
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    /// Algorithm is starting up
    Starting,
    /// Algorithm has converged to a solution
    Converged,
    /// Algorithm stopped due to reaching maximum cycles
    MaxCycles,
    /// Algorithm is currently running
    InProgress,
    /// Algorithm was manually stopped by user
    ManualStop,
    /// Other status with custom message
    Other(String),
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Starting => write!(f, "Starting"),
            Status::Converged => write!(f, "Converged"),
            Status::MaxCycles => write!(f, "Maximum cycles reached"),
            Status::InProgress => write!(f, "In progress"),
            Status::ManualStop => write!(f, "Manual stop requested"),
            Status::Other(msg) => write!(f, "{}", msg),
        }
    }
}
