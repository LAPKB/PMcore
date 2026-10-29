use anyhow::Result;
use ndarray::Array1;
use pharmsol::prelude::{data::AssayErrorModels, simulator::Equation};
use pharmsol::ParameterOptimizer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::error_optim::{ErrorModelOptimizer, ErrorOptimConfig};
use super::fit::{FitState, Runner};
use crate::algorithms::{Status, StopReason};

/// NPOD has converged when the log-likelihood changes by at most this much in a cycle.
const THETA_F: f64 = 1e-2;
/// Minimum distance between an optimized candidate and the existing support points.
const THETA_D: f64 = 1e-4;
/// Support points with at most this fraction of the largest weight are dropped.
const PRUNE_THRESHOLD: f64 = 1e-3;
/// Tolerance of the rank-revealing QR decomposition used during condensation.
const QR_TOLERANCE: f64 = 1e-8;

/// Configuration options for the Non-Parametric Optimal Design (NPOD) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpodConfig {
    /// Maximum number of cycles to run the algorithm for.
    pub max_cycles: usize,
    /// Configuration for the error-model factor (gamma/lambda) optimization.
    pub error_optim: ErrorOptimConfig,
    /// Whether to print progress information during the first cycle.
    pub progress: bool,
}

impl NpodConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_cycles(mut self, cycles: usize) -> Self {
        self.max_cycles = cycles;
        self
    }

    pub fn error_optim(mut self, config: ErrorOptimConfig) -> Self {
        self.error_optim = config;
        self
    }

    pub fn progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }
}

impl Default for NpodConfig {
    fn default() -> Self {
        Self {
            max_cycles: 100,
            error_optim: ErrorOptimConfig::default(),
            progress: true,
        }
    }
}

/// NPOD's own state between cycles: its configuration and the error-model optimizer.
#[derive(Debug)]
pub(crate) struct Npod {
    config: NpodConfig,
    error_optimizer: ErrorModelOptimizer,
}

impl Npod {
    pub(crate) fn new(config: NpodConfig, error_models: &AssayErrorModels) -> Self {
        Self {
            error_optimizer: ErrorModelOptimizer::new(config.error_optim.clone(), error_models),
            config,
        }
    }
}

impl<E: Equation> Runner<E> for Npod {
    fn step(&mut self, state: &mut FitState<E>) -> Result<Status> {
        if state.cycle > 1 {
            expand(state)?;
        }

        let lambda = state.estimate(self.config.progress && state.cycle == 1)?;
        state.condense(&lambda, PRUNE_THRESHOLD, QR_TOLERANCE)?;
        self.error_optimizer.optimize(state)?;
        state.log_cycle_summary();

        if state
            .objective_change()
            .is_some_and(|change| change <= THETA_F)
        {
            tracing::info!("Objective function convergence reached");
            return Ok(Status::Stop(StopReason::Converged));
        }
        Ok(state
            .limit_reached(self.config.max_cycles)
            .map_or(Status::Continue, Status::Stop))
    }
}

/// Move each support point towards a local maximum of the directional derivative of the
/// likelihood, and add the result as a new candidate point.
fn expand<E: Equation>(state: &mut FitState<E>) -> Result<()> {
    // The population likelihood of each subject, Ψ·w. `ParameterOptimizer` works with raw
    // likelihoods, so undo the per-subject scaling of Ψ.
    // TODO: Move NPOD and its parameter optimizer fully into log space.
    let pyl: Array1<f64> = (state.psi.matrix() * state.weights.weights())
        .iter()
        .zip(state.psi.row_log_scales())
        .map(|(pyl, row_log_scale)| pyl * row_log_scale.exp())
        .collect();

    let candidates: Vec<Array1<f64>> = state
        .theta
        .matrix()
        .row_iter()
        .map(|row| row.iter().copied().collect())
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|point| {
            ParameterOptimizer::new(&state.equation, &state.data, &state.error_models, &pyl)
                .optimize_point(point)
        })
        .collect::<Result<_, _>>()?;

    for candidate in candidates {
        state.theta.suggest_point(&candidate.to_vec(), THETA_D)?;
    }
    Ok(())
}
