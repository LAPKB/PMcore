use crate::algorithms::{NonParametricRunner, Status, StopReason};
use crate::estimation::nonparametric::{
    calculate_psi, CycleLog, NPCycle, NonParametricResult, Psi, Theta, Weights,
};

pub(crate) use crate::estimation::nonparametric::ipm::burke;
pub(crate) use crate::estimation::nonparametric::qr;

use anyhow::bail;
use anyhow::Result;
use pharmsol::prelude::{data::Data, simulator::Equation};

use crate::{AssayErrorModel, AssayErrorModels};

use crate::estimation::nonparametric::adaptative_grid;

use super::error_optim::{optimize_error_models, ErrorOptimConfig};

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
    last_objf: f64,
    objf: f64,
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
    /// adaptive grid. Initial support points can be supplied separately via
    /// [`NonParametricRunner::set_theta`].
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
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
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

impl<E: Equation + Send + 'static> NonParametricRunner<E> for NPAG<E> {
    fn equation(&self) -> &E {
        &self.equation
    }

    fn into_result(&self) -> Result<NonParametricResult<E>> {
        NonParametricResult::new(
            self.equation.clone(),
            self.data.clone(),
            self.error_models.clone(),
            self.prior.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            -2. * self.objf,
            self.cycle,
            self.status.clone(),
            self.cycle_log.clone(),
        )
    }

    fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn increment_cycle(&mut self) -> usize {
        self.cycle += 1;
        self.cycle
    }

    fn cycle(&self) -> usize {
        self.cycle
    }

    fn set_theta(&mut self, theta: Theta) {
        self.theta = theta;
    }

    fn theta(&self) -> &Theta {
        &self.theta
    }

    fn psi(&self) -> &Psi {
        &self.psi
    }

    fn evaluation(&mut self) -> Result<Status> {
        tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
        tracing::debug!("Support points: {}", self.theta.nspp());

        self.error_models.iter().for_each(|(outeq, em)| {
            if AssayErrorModel::None == *em {
                return;
            }
            tracing::debug!(
                "Error model for outeq {}: {:.2}",
                outeq,
                em.factor().unwrap_or_default()
            );
        });

        tracing::debug!("EPS = {:.4}", self.eps);
        // Increasing objf signals instability or model misspecification.
        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective function decreased from {:.4} to {:.4} (delta = {})",
                -2.0 * self.last_objf,
                -2.0 * self.objf,
                -2.0 * self.last_objf - -2.0 * self.objf
            );
        }

        let psi = self.psi.matrix();
        let w = &self.w;
        if (self.last_objf - self.objf).abs() <= self.config.objective_tolerance
            && self.eps > self.config.min_eps
        {
            self.eps /= 2.;
            if self.eps <= self.config.min_eps {
                let pyl = psi * w.weights();
                self.f1 = pyl.iter().map(|x| x.ln()).sum();
                if (self.f1 - self.f0).abs() <= self.config.pyl_tolerance {
                    tracing::info!("The model converged after {} cycles", self.cycle,);
                    self.set_status(Status::Stop(StopReason::Converged));
                    self.log_cycle_state();
                    return Ok(self.status().clone());
                } else {
                    self.f0 = self.f1;
                    self.eps = self.config.eps;
                }
            }
        }

        // Stop if we have reached maximum number of cycles
        if self.cycle >= self.config.max_cycles {
            tracing::warn!("Maximum number of cycles reached");
            self.set_status(Status::Stop(StopReason::MaxCycles));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // Stop if stopfile exists
        if std::path::Path::new("stop").exists() {
            tracing::warn!("Stopfile detected - breaking");
            self.set_status(Status::Stop(StopReason::StopFile));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // Continue with normal operation
        self.set_status(Status::Continue);
        self.log_cycle_state();
        Ok(self.status().clone())
    }

    fn estimation(&mut self) -> Result<()> {
        self.psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &self.error_models,
            self.cycle == 1 && self.config.progress,
        )?;

        if let Err(err) = self.check_zero_probability_subjects() {
            bail!(err);
        }

        (self.lambda, _) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                bail!("Error in IPM during estimation: {:?}", err);
            }
        };
        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        // Filter out the support points with lambda < max(lambda)/1000

        let max_lambda = self
            .lambda
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        let mut keep = Vec::<usize>::new();
        for (index, lam) in self.lambda.iter().enumerate() {
            if lam > max_lambda * self.config.prune_threshold {
                keep.push(index);
            }
        }
        if self.psi.matrix().ncols() != keep.len() {
            tracing::debug!(
                "Lambda (max/1000) dropped {} support point(s)",
                self.psi.matrix().ncols() - keep.len(),
            );
        }

        self.theta.filter_indices(keep.as_slice());
        self.psi.filter_column_indices(keep.as_slice());

        //Rank-Revealing Factorization
        let (r, perm) = qr::qrd(&self.psi)?;

        let mut keep = Vec::<usize>::new();

        // The minimum between the number of subjects and the actual number of support points
        let keep_n = self.psi.matrix().ncols().min(self.psi.matrix().nrows());
        for i in 0..keep_n {
            let test = r.col(i).norm_l2();
            let r_diag_val = r.get(i, i);
            let ratio = r_diag_val / test;
            if ratio.abs() >= self.config.qr_tolerance {
                keep.push(*perm.get(i).unwrap());
            }
        }

        // If a support point is dropped, log it as a debug message
        if self.psi.matrix().ncols() != keep.len() {
            tracing::debug!(
                "QR decomposition dropped {} support point(s)",
                self.psi.matrix().ncols() - keep.len(),
            );
        }

        // Filter to keep only the support points (rows) that are in the `keep` vector
        self.theta.filter_indices(keep.as_slice());
        // Filter to keep only the support points (columns) that are in the `keep` vector
        self.psi.filter_column_indices(keep.as_slice());

        self.check_zero_probability_subjects()?;
        (self.lambda, self.objf) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "Error in IPM during condensation: {:?}",
                    err
                ));
            }
        };
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
            &mut self.objf,
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

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn log_cycle_state(&mut self) {
        let state = NPCycle::new(
            self.cycle,
            -2. * self.objf,
            self.error_models.clone(),
            self.theta.clone(),
            self.w.clone(),
            self.theta.nspp(),
            (self.last_objf - self.objf).abs(),
            self.status.clone(),
        );
        self.cycle_log.push(state);
        self.last_objf = self.objf;
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
