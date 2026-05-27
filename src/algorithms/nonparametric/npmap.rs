use crate::{
    algorithms::{NonParametricAlgorithm, Status, StopReason},
    api::EstimationProblem,
    estimation::nonparametric::{
        calculate_psi, CycleLog, NPCycle, NonParametricResult, Psi, Theta, Weights,
    },
    prelude::algorithms::Algorithm,
};
use anyhow::{Context, Result};

use pharmsol::prelude::{
    data::{AssayErrorModels, Data},
    simulator::Equation,
};

use crate::estimation::nonparametric::ipm::burke;

use serde::{Deserialize, Serialize};

/// Configuration options for the posterior probability reweighting algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NpmapConfig;

impl NpmapConfig {
    pub fn new() -> Self {
        Self
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
            Algorithm::NPMAP(NpmapConfig::default()),
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

impl<E: Equation + Send + 'static> NPMAP<E> {
    pub(crate) fn from_input(input: EstimationProblem<E>) -> Result<Box<Self>> {
        let config = match input.algorithm.clone() {
            Algorithm::NPMAP(config) => config,
            other => unreachable!(
                "NPMAP::from_input requires an NPMAP algorithm, got {}",
                other.name()
            ),
        };
        let error_models = input.error_models.models().clone();

        Ok(Box::new(Self {
            equation: input.model.equation,
            psi: Psi::new(),
            theta: Theta::new(),
            w: Weights::default(),
            objf: f64::INFINITY,
            cycle: 0,
            status: Status::Continue,
            data: input.data,
            config,
            cyclelog: CycleLog::new(),
            error_models,
        }))
    }
}
