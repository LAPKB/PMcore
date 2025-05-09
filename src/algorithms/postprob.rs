use crate::{
    prelude::algorithms::Algorithms,
    structs::{
        psi::{calculate_psi, Psi},
        theta::Theta,
    },
};
use anyhow::Result;
use faer::Col;
use pharmsol::prelude::{
    data::{Data, ErrorModel},
    simulator::Equation,
};

use crate::routines::evaluation::ipm::burke;
use crate::routines::initialization;
use crate::routines::output::CycleLog;
use crate::routines::output::NPResult;
use crate::routines::settings::Settings;

/// Posterior probability algorithm
/// Reweights the prior probabilities to the observed data and error model
pub struct POSTPROB<E: Equation> {
    equation: E,
    psi: Psi,
    theta: Theta,
    w: Col<f64>,
    objf: f64,
    cycle: usize,
    converged: bool,
    gamma: f64,
    data: Data,
    settings: Settings,
    cyclelog: CycleLog,
    error_model: ErrorModel,
}

impl<E: Equation> Algorithms<E> for POSTPROB<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        Ok(Box::new(Self {
            equation,
            psi: Psi::new(),
            theta: Theta::new(),
            w: Col::zeros(0),
            objf: f64::INFINITY,
            cycle: 0,
            converged: false,
            error_model: settings.error().clone().into(),
            gamma: settings.error().value,
            settings,
            data,

            cyclelog: CycleLog::new(),
        }))
    }
    fn into_npresult(&self) -> NPResult<E> {
        NPResult::new(
            self.equation.clone(),
            self.data.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            self.objf,
            self.cycle,
            self.converged,
            self.settings.clone(),
            self.cyclelog.clone(),
        )
    }
    fn get_settings(&self) -> &Settings {
        &self.settings
    }

    fn equation(&self) -> &E {
        &self.equation
    }

    fn get_data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Theta {
        initialization::sample_space(&self.settings).unwrap()
    }

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn inc_cycle(&mut self) -> usize {
        0
    }

    fn get_cycle(&self) -> usize {
        0
    }

    fn set_theta(&mut self, theta: Theta) {
        self.theta = theta;
    }

    fn get_theta(&self) -> &Theta {
        &self.theta
    }

    fn psi(&self) -> &Psi {
        &self.psi
    }

    fn convergence_evaluation(&mut self) {}

    fn converged(&self) -> bool {
        true
    }

    fn evaluation(&mut self) -> Result<()> {
        self.psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &self.error_model,
            false,
            false,
        );
        (self.w, self.objf) = burke(&self.psi).expect("Error in IPM");
        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        Ok(())
    }
    fn optimizations(&mut self) -> Result<()> {
        Ok(())
    }

    fn logs(&self) {}

    fn expansion(&mut self) -> Result<()> {
        Ok(())
    }
}
