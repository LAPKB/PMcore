use crate::{
    algorithms::{FitState, NonParametricRunner, Status, StopReason},
    estimation::nonparametric::{CycleLog, NPCycle, NonParametricResult, Psi, Theta, Weights},
};
use pharmsol::ParameterOptimizer;

use anyhow::Result;
use pharmsol::prelude::{data::Data, simulator::Equation};
use pharmsol::AssayErrorModels;

use ndarray::Array1;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use super::error_optim::{optimize_error_models, ErrorOptimConfig};
use super::stages::{self, CondensationOptions};

const THETA_D: f64 = 1e-4;

/// Configuration options for the Non-Parametric Optimal Design (NPOD) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpodConfig {
    /// Maximum number of cycles to run the algorithm for.
    pub max_cycles: usize,
    /// Configuration for the error-model factor (gamma/lambda) optimization.
    pub error_optim: ErrorOptimConfig,
    /// Whether to print progress information during the first cycle.
    pub progress: bool,
    /// Convergence tolerance on the change in the objective function.
    pub objective_tolerance: f64,
    /// Support points whose weight is below `max_weight × prune_threshold` are dropped.
    pub prune_threshold: f64,
    /// Minimum `|r_ii| / ‖r_i‖` ratio for a support point to survive the QR decomposition.
    pub qr_tolerance: f64,
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

    /// Set the convergence tolerance on the change in the objective function.
    pub fn objective_tolerance(mut self, tolerance: f64) -> Self {
        self.objective_tolerance = tolerance;
        self
    }

    /// Set the weight below which support points are dropped, relative to the
    /// largest weight.
    pub fn prune_threshold(mut self, threshold: f64) -> Self {
        self.prune_threshold = threshold;
        self
    }

    /// Set the minimum `|r_ii| / ‖r_i‖` ratio for a support point to survive the
    /// QR decomposition.
    pub fn qr_tolerance(mut self, tolerance: f64) -> Self {
        self.qr_tolerance = tolerance;
        self
    }
}

impl Default for NpodConfig {
    fn default() -> Self {
        Self {
            max_cycles: 100,
            error_optim: ErrorOptimConfig::default(),
            progress: true,
            objective_tolerance: 1e-2,
            prune_threshold: 1e-3,
            qr_tolerance: 1e-8,
        }
    }
}

#[derive(Debug)]
pub struct NPOD<E: Equation + Send + 'static> {
    equation: E,
    psi: Psi,
    prior: Theta,
    theta: Theta,
    lambda: Weights,
    w: Weights,
    last_log_likelihood: f64,
    log_likelihood: f64,
    cycle: usize,
    gamma_delta: Vec<f64>,
    error_models: AssayErrorModels,
    status: Status,
    cycle_log: CycleLog,
    data: Data,
    config: NpodConfig,
}

impl<E: Equation + Send + 'static> NPOD<E> {
    pub(crate) fn from_parts(
        equation: E,
        data: Data,
        error_models: AssayErrorModels,
        theta: Theta,
        config: NpodConfig,
    ) -> Result<Self> {
        let gamma_delta = vec![config.error_optim.step; error_models.len()];

        Ok(Self {
            equation,
            psi: Psi::new(),
            prior: theta.clone(),
            theta,
            lambda: Weights::default(),
            w: Weights::default(),
            last_log_likelihood: -1e30,
            log_likelihood: f64::NEG_INFINITY,
            cycle: 0,
            gamma_delta,
            error_models,
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            data,
            config,
        })
    }
}

impl<E: Equation + Send + 'static> FitState<E> for NPOD<E> {
    fn equation(&self) -> &E {
        &self.equation
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }

    fn theta(&self) -> &Theta {
        &self.theta
    }

    fn psi(&self) -> &Psi {
        &self.psi
    }

    fn weights(&self) -> &Weights {
        &self.w
    }

    fn cycle(&self) -> usize {
        self.cycle
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }
}

impl<E: Equation + Send + 'static> NonParametricRunner<E> for NPOD<E> {
    type Output = NonParametricResult<E>;

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn increment_cycle(&mut self) -> usize {
        self.cycle += 1;
        self.cycle
    }

    fn into_result(self: Box<Self>) -> Result<Self::Output> {
        let this = *self;
        let n2ll = this.n2ll();

        NonParametricResult::new(
            this.equation,
            this.data,
            this.error_models,
            this.prior,
            this.theta,
            this.psi,
            this.w,
            n2ll,
            this.cycle,
            this.status,
            this.cycle_log,
        )
    }

    fn push_cycle(&mut self, cycle: NPCycle) {
        self.cycle_log.push(cycle);
    }

    fn objective_delta(&mut self) -> f64 {
        let delta = (self.last_log_likelihood - self.log_likelihood).abs();
        self.last_log_likelihood = self.log_likelihood;
        delta
    }

    fn evaluation(&mut self) -> Result<Status> {
        stages::log_objective(
            self.log_likelihood,
            self.last_log_likelihood,
            &self.theta,
            &self.error_models,
        );

        if (self.last_log_likelihood - self.log_likelihood).abs() <= self.config.objective_tolerance
        {
            tracing::info!("Objective function convergence reached");
            return Ok(Status::Stop(StopReason::Converged));
        }

        if self.cycle >= self.config.max_cycles {
            tracing::warn!("Maximum number of cycles reached");
            return Ok(Status::Stop(StopReason::MaxCycles));
        }

        if crate::algorithms::stop::stop_file_present() {
            tracing::warn!("Stopfile detected - breaking");
            return Ok(Status::Stop(StopReason::StopFile));
        }

        Ok(Status::Continue)
    }

    fn estimation(&mut self) -> Result<()> {
        let (psi, lambda, log_likelihood) = stages::estimate(
            &self.equation,
            &self.data,
            &self.error_models,
            &self.theta,
            self.cycle == 1 && self.config.progress,
        )?;

        self.psi = psi;
        self.lambda = lambda;
        self.log_likelihood = log_likelihood;

        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        let (lambda, log_likelihood) = stages::condense(
            &self.equation,
            &self.data,
            &self.error_models,
            &mut self.theta,
            &mut self.psi,
            &self.lambda,
            &CondensationOptions {
                prune_threshold: self.config.prune_threshold,
                qr_tolerance: self.config.qr_tolerance,
                check_zero_probability: true,
            },
        )?;

        self.lambda = lambda;
        self.log_likelihood = log_likelihood;
        self.w = self.lambda.clone();

        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        optimize_error_models(
            &self.equation,
            &self.data,
            &self.theta,
            &mut self.error_models,
            &mut self.gamma_delta,
            &mut self.log_likelihood,
            &mut self.lambda,
            &mut self.psi,
            &self.config.error_optim,
        )
    }

    fn expansion(&mut self) -> Result<()> {
        let pyl_col = self.psi().matrix().as_ref() * self.w.weights().as_ref();
        let mut pyl: Array1<f64> = pyl_col.iter().copied().collect();

        // ParameterOptimizer currently evaluates raw likelihoods, so restore
        // pyl to the same scale before calculating directional derivatives.
        // TODO: Move NPOD and its parameter optimizer fully into log space.
        for (pyl_i, row_log_scale) in pyl.iter_mut().zip(self.psi().row_log_scales()) {
            *pyl_i *= row_log_scale.exp();
        }

        let error_model: AssayErrorModels = self.error_models.clone();

        let mut candididate_points: Vec<Array1<f64>> = Vec::default();
        for spp in self.theta.matrix().row_iter() {
            let candidate: Vec<f64> = spp.iter().cloned().collect();
            let spp = Array1::from(candidate);
            candididate_points.push(spp.to_owned());
        }
        candididate_points.par_iter_mut().for_each(|spp| {
            let optimizer = ParameterOptimizer::new(&self.equation, &self.data, &error_model, &pyl);
            let candidate_point = optimizer.optimize_point(spp.to_owned()).unwrap();
            *spp = candidate_point;
        });
        for cp in candididate_points {
            self.theta.suggest_point(cp.to_vec().as_slice(), THETA_D)?;
        }
        Ok(())
    }
}
