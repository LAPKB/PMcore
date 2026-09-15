use crate::{
    algorithms::{FitState, NonParametricRunner, Status, StopReason},
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
#[derive(Debug)]
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

impl<E: Equation + Send + 'static> FitState<E> for NPMAP<E> {
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
        self.objf
    }
}

impl<E: Equation + Send + 'static> NonParametricRunner<E> for NPMAP<E> {
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
        let objf = this.n2ll();

        NonParametricResult::new(
            this.equation,
            this.data,
            this.error_models,
            this.prior,
            this.theta,
            this.psi,
            this.w,
            objf,
            this.cycle,
            this.status,
            this.cyclelog,
        )
    }

    fn push_cycle(&mut self, cycle: NPCycle) {
        self.cyclelog.push(cycle);
    }

    /// NPMAP is a single-pass reweighting of the prior support points, so the
    /// only cycle stops as soon as it has been evaluated.
    fn evaluation(&mut self) -> Result<Status> {
        Ok(Status::Stop(StopReason::Converged))
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
}
