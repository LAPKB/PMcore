use crate::{
    algorithms::{NativeNonparametricConfig, NonparametricAlgorithmInput, Status, StopReason},
    estimation::nonparametric::{calculate_psi, CycleLog, NonparametricWorkspace, NPCycle, Psi, Theta, Weights},
    prelude::algorithms::Algorithms,
};
use anyhow::{Context, Result};

use pharmsol::prelude::{
    data::{AssayErrorModels, Data},
    simulator::Equation,
};

use crate::estimation::nonparametric::ipm::burke;
use crate::estimation::nonparametric::sample_space_for_parameters;

/// Posterior probability algorithm
/// Reweights the prior probabilities to the observed data and error model
pub struct POSTPROB<E: Equation + Send + 'static> {
    equation: E,
    psi: Psi,
    theta: Theta,
    w: Weights,
    objf: f64,
    cycle: usize,
    status: Status,
    data: Data,
    config: NativeNonparametricConfig,
    cyclelog: CycleLog,
    error_models: AssayErrorModels,
}

impl<E: Equation + Send + 'static> Algorithms<E> for POSTPROB<E> {
    fn into_workspace(&self) -> Result<NonparametricWorkspace<E>> {
        NonparametricWorkspace::new(
            self.equation.clone(),
            self.data.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            self.objf,
            self.cycle,
            self.status.clone(),
            self.config.run_configuration.clone(),
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
        sample_space_for_parameters(&self.config.parameter_space, &self.config.prior)
            .unwrap()
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

impl<E: Equation + Send + 'static> POSTPROB<E> {
    pub(crate) fn from_input(input: NonparametricAlgorithmInput<E>) -> Result<Box<Self>> {
        let config = input.native_config()?;
        let error_models = input.error_models().clone();
        let equation = input.equation;
        let data = input.data;

        Ok(Box::new(Self {
            equation,
            psi: Psi::new(),
            theta: Theta::new(),
            w: Weights::default(),
            objf: f64::INFINITY,
            cycle: 0,
            status: Status::Continue,
            data,
            config,
            cyclelog: CycleLog::new(),
            error_models,
        }))
    }
}
