use crate::{
    algorithms::{NonParametricRunner, Status, StopReason},
    estimation::nonparametric::{
        calculate_psi, ipm::burke, qr, CycleLog, NPCycle, NonParametricResult, Psi, Theta, Weights,
    },
};
use pharmsol::ParameterOptimizer;

use anyhow::bail;
use anyhow::Result;
use pharmsol::prelude::{data::Data, simulator::Equation};
use pharmsol::{prelude::AssayErrorModel, AssayErrorModels};

use ndarray::Array1;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use super::error_optim::{optimize_error_models, ErrorOptimConfig};

const THETA_F: f64 = 1e-2;
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

pub struct NPOD<E: Equation + Send + 'static> {
    equation: E,
    psi: Psi,
    prior: Theta,
    theta: Theta,
    lambda: Weights,
    w: Weights,
    last_objf: f64,
    objf: f64,
    cycle: usize,
    gamma_delta: Vec<f64>,
    error_models: AssayErrorModels,
    converged: bool,
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
            theta: theta,
            lambda: Weights::default(),
            w: Weights::default(),
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            cycle: 0,
            gamma_delta,
            error_models,
            converged: false,
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            data,
            config,
        })
    }
}

impl<E: Equation + Send + 'static> NonParametricRunner<E> for NPOD<E> {
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

    fn equation(&self) -> &E {
        &self.equation
    }

    fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }

    fn data(&self) -> &Data {
        &self.data
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

    fn likelihood(&self) -> f64 {
        self.objf
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
            self.theta.nspp(),
            (self.last_objf - self.objf).abs(),
            self.status.clone(),
        );
        self.cycle_log.push(state);
        self.last_objf = self.objf;
    }

    fn evaluation(&mut self) -> Result<Status> {
        tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
        tracing::debug!("Support points: {}", self.theta.nspp());
        self.error_models.iter().for_each(|(outeq, em)| {
            if AssayErrorModel::None == *em {
                return;
            }
            tracing::debug!(
                "Error model for outeq {}: {:.16}",
                outeq,
                em.factor().unwrap_or_default()
            );
        });
        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective function decreased from {:.4} to {:.4} (delta = {})",
                -2.0 * self.last_objf,
                -2.0 * self.objf,
                -2.0 * self.last_objf - -2.0 * self.objf
            );
        }

        if (self.last_objf - self.objf).abs() <= THETA_F {
            tracing::info!("Objective function convergence reached");
            self.converged = true;
            self.set_status(Status::Stop(StopReason::Converged));
            self.log_cycle_state();
            return Ok(self.status.clone());
        }

        if self.cycle >= self.config.max_cycles {
            tracing::warn!("Maximum number of cycles reached");
            self.converged = true;
            self.set_status(Status::Stop(StopReason::MaxCycles));
            self.log_cycle_state();
            return Ok(self.status.clone());
        }

        if std::path::Path::new("stop").exists() {
            tracing::warn!("Stopfile detected - breaking");
            self.converged = true;
            self.set_status(Status::Stop(StopReason::Stopped));
            self.log_cycle_state();
            return Ok(self.status.clone());
        }

        self.status = Status::Continue;
        self.log_cycle_state();
        Ok(self.status.clone())
    }

    fn estimation(&mut self) -> Result<()> {
        let error_model: AssayErrorModels = self.error_models.clone();

        self.psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &error_model,
            self.cycle == 1 && self.config.progress,
        )?;

        if let Err(err) = self.check_zero_probability_subjects() {
            bail!(err);
        }

        (self.lambda, _) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                bail!(err);
            }
        };
        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        let max_lambda = self
            .lambda
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        let mut keep = Vec::<usize>::new();
        for (index, lam) in self.lambda.iter().enumerate() {
            if lam > max_lambda / 1000_f64 {
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

        let (r, perm) = qr::qrd(&self.psi)?;

        let mut keep = Vec::<usize>::new();
        let keep_n = self.psi.matrix().ncols().min(self.psi.matrix().nrows());
        for i in 0..keep_n {
            let test = r.col(i).norm_l2();
            let r_diag_val = r.get(i, i);
            let ratio = r_diag_val / test;
            if ratio.abs() >= 1e-8 {
                keep.push(*perm.get(i).unwrap());
            }
        }

        if self.psi.matrix().ncols() != keep.len() {
            tracing::debug!(
                "QR decomposition dropped {} support point(s)",
                self.psi.matrix().ncols() - keep.len(),
            );
        }

        self.theta.filter_indices(keep.as_slice());
        self.psi.filter_column_indices(keep.as_slice());

        (self.lambda, self.objf) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                return Err(anyhow::anyhow!("Error in IPM: {:?}", err));
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
        let pyl_col = self.psi().matrix().as_ref() * self.w.weights().as_ref();
        let pyl: Array1<f64> = pyl_col.iter().copied().collect();

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
