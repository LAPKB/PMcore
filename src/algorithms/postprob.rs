use crate::{
    algorithms::{Status, StopReason},
    prelude::algorithms::Algorithms,
    structs::{
        psi::{calculate_psi_dispatch, Psi},
        theta::Theta,
        weights::Weights,
    },
};
use anyhow::{Context, Result};

use pharmsol::prelude::{
    data::{Data, ErrorModels},
    simulator::Equation,
};

use crate::routines::estimation::ipm::burke_ipm;
use crate::routines::initialization;
use crate::routines::output::{cycles::CycleLog, NPResult};
use crate::routines::settings::Settings;

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
    settings: Settings,
    cyclelog: CycleLog,
    error_models: ErrorModels,
}

impl<E: Equation + Send + 'static> Algorithms<E> for POSTPROB<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        Ok(Box::new(Self {
            equation,
            psi: Psi::new(),
            theta: Theta::new(),
            w: Weights::default(),
            objf: f64::INFINITY,
            cycle: 0,
            status: Status::Continue,
            error_models: settings.errormodels().clone(),
            settings,
            data,
            cyclelog: CycleLog::new(),
        }))
    }
    fn into_npresult(&self) -> Result<NPResult<E>> {
        NPResult::new(
            self.equation.clone(),
            self.data.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            self.objf,
            self.cycle,
            self.status.clone(),
            self.settings.clone(),
            self.cyclelog.clone(),
        )
    }
    fn settings(&self) -> &Settings {
        &self.settings
    }

    fn equation(&self) -> &E {
        &self.equation
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Theta {
        initialization::sample_space(&self.settings).unwrap()
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
        let use_log_space = self.settings.advanced().log_space;

        self.psi = calculate_psi_dispatch(
            &self.equation,
            &self.data,
            &self.theta,
            &self.error_models,
            false,
            false,
            use_log_space,
        )?;

        (self.w, self.objf) = burke_ipm(&self.psi).context("Error in IPM")?;
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
        let state = crate::routines::output::cycles::NPCycle::new(
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
