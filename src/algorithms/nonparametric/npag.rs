use crate::algorithms::{FitState, NonParametricRunner, Status, StopReason};
use crate::estimation::nonparametric::{
    CycleLog, NPCycle, NonParametricResult, Psi, Theta, Weights,
};

use anyhow::Result;
use pharmsol::prelude::{
    data::{AssayErrorModels, Data},
    simulator::Equation,
};

use crate::estimation::nonparametric::adaptative_grid;

use super::error_optim::{optimize_error_models, ErrorOptimConfig};
use super::stages::{self, CondensationOptions};

use serde::{Deserialize, Serialize};

/// Configuration options for the Non-Parametric Adaptive Grid (NPAG) algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NpagConfig {
    pub eps: f64,
    pub min_eps: f64,
    pub objective_tolerance: f64,
    pub pyl_tolerance: f64,
    pub prune_threshold: f64,
    pub qr_tolerance: f64,
    pub grid_tolerance: f64,
    pub error_optim: ErrorOptimConfig,
    pub max_cycles: usize,
    pub progress: bool,
}

impl Default for NpagConfig {
    fn default() -> Self {
        Self {
            eps: 0.2,
            min_eps: 1e-4,
            objective_tolerance: 1e-4,
            pyl_tolerance: 1e-2,
            prune_threshold: 1e-3,
            qr_tolerance: 1e-8,
            grid_tolerance: 1e-4,
            error_optim: ErrorOptimConfig::default(),
            max_cycles: 1000,
            progress: true,
        }
    }
}

impl NpagConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn min_eps(mut self, min_eps: f64) -> Self {
        self.min_eps = min_eps;
        self
    }

    pub fn objective_tolerance(mut self, tolerance: f64) -> Self {
        self.objective_tolerance = tolerance;
        self
    }

    pub fn pyl_tolerance(mut self, tolerance: f64) -> Self {
        self.pyl_tolerance = tolerance;
        self
    }

    pub fn prune_threshold(mut self, threshold: f64) -> Self {
        self.prune_threshold = threshold;
        self
    }

    pub fn qr_tolerance(mut self, tolerance: f64) -> Self {
        self.qr_tolerance = tolerance;
        self
    }

    pub fn grid_tolerance(mut self, tolerance: f64) -> Self {
        self.grid_tolerance = tolerance;
        self
    }

    pub fn error_optim(mut self, config: ErrorOptimConfig) -> Self {
        self.error_optim = config;
        self
    }

    pub fn max_cycles(mut self, cycles: usize) -> Self {
        self.max_cycles = cycles;
        self
    }

    pub fn progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }
}

#[derive(Debug)]
pub struct NPAG<E: Equation + Send + 'static> {
    equation: E,
    ranges: Vec<(f64, f64)>,
    psi: Psi,
    prior: Theta,
    theta: Theta,
    lambda: Weights,
    w: Weights,
    eps: f64,
    last_log_likelihood: f64,
    log_likelihood: f64,
    f0: f64,
    f1: f64,
    cycle: usize,
    gamma_delta: Vec<f64>,
    error_models: AssayErrorModels,
    status: Status,
    cycle_log: CycleLog,
    data: Data,
    config: NpagConfig,
}

impl<E: Equation + Send + 'static> NPAG<E> {
    /// Construct an `NPAG` instance from explicit parts.
    ///
    /// The `parameter_space` is used solely to derive the finite bounds for the
    /// adaptive grid. The initial support points come from `theta`, which is
    /// also kept as the prior.
    pub(crate) fn from_parts(
        equation: E,
        data: Data,
        error_models: AssayErrorModels,
        theta: Theta,
        config: NpagConfig,
    ) -> Result<Self> {
        let ranges = theta.parameters().finite_ranges();
        let gamma_delta = vec![config.error_optim.step; error_models.len()];
        let eps = config.eps;

        Ok(Self {
            equation,
            ranges,
            psi: Psi::new(),
            prior: theta.clone(),
            theta,
            lambda: Weights::default(),
            w: Weights::default(),
            eps,
            last_log_likelihood: -1e30,
            log_likelihood: f64::NEG_INFINITY,
            f0: -1e30,
            f1: f64::default(),
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

impl<E: Equation + Send + 'static> FitState<E> for NPAG<E> {
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

impl<E: Equation + Send + 'static> NonParametricRunner<E> for NPAG<E> {
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
        tracing::debug!("EPS = {:.4}", self.eps);

        let psi = self.psi.matrix();
        let w = &self.w;
        if (self.last_log_likelihood - self.log_likelihood).abs() <= self.config.objective_tolerance
            && self.eps > self.config.min_eps
        {
            self.eps /= 2.;
            if self.eps <= self.config.min_eps {
                let pyl = psi * w.weights();
                self.f1 = pyl.iter().map(|x| x.ln()).sum();
                if (self.f1 - self.f0).abs() <= self.config.pyl_tolerance {
                    tracing::info!("The model converged after {} cycles", self.cycle);
                    return Ok(Status::Stop(StopReason::Converged));
                } else {
                    self.f0 = self.f1;
                    self.eps = self.config.eps;
                }
            }
        }

        // Stop if we have reached maximum number of cycles
        if self.cycle >= self.config.max_cycles {
            tracing::warn!("Maximum number of cycles reached");
            return Ok(Status::Stop(StopReason::MaxCycles));
        }

        // Stop if stopfile exists
        if crate::algorithms::stop::stop_file_present() {
            tracing::warn!("Stopfile detected - breaking");
            return Ok(Status::Stop(StopReason::StopFile));
        }

        // Continue with normal operation
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
        adaptative_grid(
            &mut self.theta,
            self.eps,
            &self.ranges,
            self.config.grid_tolerance,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    use pharmsol::{fa, fetch_params, lag, Subject, SubjectBuilderExt};

    fn simple_equation() -> pharmsol::equation::ODE {
        pharmsol::equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            pharmsol::equation::metadata::new("npag_settings_test")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["0"])
                .route(pharmsol::equation::Route::bolus("0").to_state("central")),
        )
        .expect("metadata attachment should validate")
    }

    fn simple_data() -> Data {
        let subject = Subject::builder("1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .build();

        Data::new(vec![subject])
    }

    #[test]
    fn npag_runs_without_error() {
        let parameters = ParameterSpace::bounded()
            .add("ke", 0.001, 3.0)
            .add("v", 25.0, 250.0);
        let prior = Theta::sobol_default(&parameters).expect("Failed to build prior");
        let error_models = AssayErrorModels::new()
            .add(
                "0",
                AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
            )
            .expect("Failed to build error models");
        let problem =
            EstimationProblem::nonparametric(simple_equation(), simple_data(), prior, error_models)
                .expect("Failed to build problem");

        let result = problem.fit_with(NonParametricAlgorithm::npag());

        assert!(
            result.is_ok(),
            "NPAG algorithm should run without error, but got: {:?}",
            result.err()
        );
    }
}
