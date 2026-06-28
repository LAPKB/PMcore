use crate::{
    algorithms::{NonParametricRunner, Status, StopReason},
    estimation::nonparametric::{
        calculate_psi, CycleLog, NPCycle, NonParametricResult, Psi, Theta, Weights,
    },
};

use anyhow::{Context, Result};
use pharmsol::prelude::{
    data::{AssayErrorModels, Data},
    simulator::Equation,
};

use crate::estimation::nonparametric::ipm::burke;
use serde::{Deserialize, Serialize};

/// Configuration options for the non-parametric maximum a posteriori (NPMAP) algorithm
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NpmapConfig {}

impl NpmapConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Non-parametric maximum a posteriori (NPMAP) algorithm
///
/// This algorithm is a wrapper around the IPM algorithm that calculates the posterior probabilities of the support points
/// given a prior distribution and the likelihood of the data.
pub struct NPMAP<E: Equation + Send + 'static> {
    equation: E,
    psi: Psi,
    theta: Theta,
    w: Weights,
    objf: f64,
    cycle: usize,
    status: Status,
    data: Data,
    cyclelog: CycleLog,
    error_models: AssayErrorModels,
    prior: Theta,
}

impl<E: Equation + Send + 'static> NPMAP<E> {
    pub(crate) fn from_parts(
        equation: E,
        data: Data,
        error_models: AssayErrorModels,
        theta: Theta,
        _config: NpmapConfig,
    ) -> Result<Self> {
        Ok(Self {
            equation,
            psi: Psi::new(),
            theta: theta.clone(),
            w: Weights::default(),
            objf: f64::INFINITY,
            cycle: 0,
            status: Status::Continue,
            data,
            cyclelog: CycleLog::new(),
            error_models,
            prior: theta,
        })
    }
}

impl<E: Equation + Send + 'static> NonParametricRunner<E> for NPMAP<E> {
    fn into_result(&self) -> Result<NonParametricResult<E>> {
        NonParametricResult::new(
            self.equation.clone(),
            self.data.clone(),
            self.error_models.clone(),
            self.prior.clone(),
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

    /// POSTPROB is a single-pass reweighting: it evaluates the likelihood of the
    /// fixed prior support points once, rather than iterating cycles.
    fn fit(&mut self) -> Result<NonParametricResult<E>> {
        self.estimation()?;
        self.evaluation()?;
        self.log_cycle_state();

        self.into_result()
    }
}
