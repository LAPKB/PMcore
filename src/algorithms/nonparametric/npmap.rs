use crate::{
    algorithms::{Algorithm, Fitter, NonParametricAlgorithm, Status, StopReason},
    api::{EstimationProblem, NonParametric},
    estimation::nonparametric::{
        calculate_psi, CycleLog, NPCycle, NonParametricResult, Prior, Psi, Theta, Weights,
    },
    model::parameter_space::NonParametricParameters,
};

use anyhow::{Context, Result};
use pharmsol::prelude::{
    data::{AssayErrorModels, Data},
    simulator::Equation,
};

use crate::estimation::nonparametric::ipm::burke;
use serde::{Deserialize, Serialize};

/// Configuration options for the posterior probability reweighting algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NpmapConfig {
    pub prior: Prior,
}

impl Default for NpmapConfig {
    fn default() -> Self {
        Self {
            prior: Prior::default(),
        }
    }
}

impl NpmapConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn prior(mut self, prior: Prior) -> Self {
        self.prior = prior;
        self
    }
}

/// Posterior probability algorithm
/// Reweights the prior probabilities to the observed data and error model
pub struct NPMAP<E: Equation + Send + 'static> {
    equation: E,
    psi: Psi,
    theta: Theta,
    w: Weights,
    objf: f64,
    cycle: usize,
    status: Status,
    data: Data,
    config: NpmapConfig,
    cyclelog: CycleLog,
    error_models: AssayErrorModels,
}

impl<E: Equation + Send + 'static> NPMAP<E> {
    pub(crate) fn from_parts(
        equation: E,
        data: Data,
        error_models: AssayErrorModels,
        parameters: &NonParametricParameters,
        config: NpmapConfig,
    ) -> Result<Self> {
        // Generate or load the initial support points from the prior
        let theta = config.prior.theta(parameters)?;

        Ok(Self {
            equation,
            psi: Psi::new(),
            theta,
            w: Weights::default(),
            objf: f64::INFINITY,
            cycle: 0,
            status: Status::Continue,
            data,
            config,
            cyclelog: CycleLog::new(),
            error_models,
        })
    }
}

impl<E: Equation + Send + 'static> NonParametricAlgorithm<E> for NPMAP<E> {
    fn into_workspace(&self) -> Result<NonParametricResult<E>> {
        NonParametricResult::new(
            self.equation.clone(),
            self.data.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            self.objf,
            self.cycle,
            self.status.clone(),
            self.cyclelog.clone(),
        )
    }

    fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }

    fn equation(&self) -> &E {
        &self.equation
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Theta {
        unimplemented!("get_prior method is not implemented yet")
    }

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn increment_cycle(&mut self) -> usize {
        0
    }

    fn cycle(&self) -> usize {
        0
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

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn evaluation(&mut self) -> Result<Status> {
        self.status = Status::Stop(StopReason::Converged);
        Ok(self.status.clone())
    }

    fn estimation(&mut self) -> Result<()> {
        self.psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &self.error_models,
            false,
        )?;
        (self.w, self.objf) = burke(&self.psi).context("Error in IPM")?;
        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        Ok(())
    }

    fn expansion(&mut self) -> Result<()> {
        Ok(())
    }

    fn log_cycle_state(&mut self) {
        // Postprob doesn't track last_objf, so we use 0.0 as the delta
        let state = NPCycle::new(
            self.cycle,
            self.objf,
            self.error_models.clone(),
            self.theta.clone(),
            self.theta.nspp(),
            0.0,
            self.status.clone(),
        );
        self.cyclelog.push(state);
    }
}

// ==============================================================================
// STRATEGY / ENGINE PIPELINE
// ==============================================================================

impl<E: Equation + Send + 'static> Algorithm<E, NonParametric> for NpmapConfig {
    type Runner = NPMAP<E>;

    fn build_runner(self, problem: EstimationProblem<E, NonParametric>) -> Result<Self::Runner> {
        NPMAP::from_parts(
            problem.model.equation,
            problem.data,
            problem.error_models,
            &problem.parameters,
            self,
        )
    }
}

impl<E: Equation + Send + 'static> Fitter<E> for NPMAP<E> {
    type Output = NonParametricResult<E>;

    fn fit(mut self) -> Result<Self::Output> {
        // Since NPMAP is a single-pass algorithm, the execution loop is very simple:
        self.estimation()?;
        self.evaluation()?;
        self.log_cycle_state();

        // Return the strictly-typed NonParametricResult
        self.into_workspace()
    }
}
