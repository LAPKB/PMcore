use crate::prelude::{algorithms::Algorithm, ipm::burke, output::NPResult, settings::Settings};
use anyhow::{Error, Result};
use pharmsol::prelude::{
    data::{Data, ErrorModel, ErrorType},
    simulator::{psi, Equation},
};

use ndarray::{Array1, Array2};

use super::{initialization, output::CycleLog};

/// Posterior probability algorithm
/// Reweights the prior probabilities to the observed data and error model
pub struct POSTPROB<E: Equation> {
    equation: E,
    psi: Array2<f64>,
    theta: Array2<f64>,
    w: Array1<f64>,
    objf: f64,
    cycle: usize,
    converged: bool,
    gamma: f64,
    error_type: ErrorType,
    data: Data,
    c: (f64, f64, f64, f64),
    #[allow(dead_code)]
    settings: Settings,
    cyclelog: CycleLog,
}

impl<E: Equation> Algorithm<E> for POSTPROB<E> {
    type Matrix = Array2<f64>;

    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        Ok(Box::new(Self {
            equation,
            psi: Array2::default((0, 0)),
            theta: Array2::default((0, 0)),
            w: Array1::default(0),
            objf: f64::INFINITY,
            cycle: 0,
            converged: false,
            gamma: settings.error.value,
            error_type: match settings.error.class.as_str() {
                "additive" => ErrorType::Add,
                "proportional" => ErrorType::Prop,
                _ => panic!("Error type not supported"),
            },
            c: settings.error.poly,
            settings,
            data,

            cyclelog: CycleLog::new(),
        }))
    }
    fn to_npresult(&self) -> NPResult {
        NPResult::new(
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

    fn get_data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Self::Matrix {
        initialization::sample_space(&self.settings, &self.data, &self.equation).unwrap()
    }

    fn inc_cycle(&mut self) -> usize {
        0
    }

    fn set_theta(&mut self, theta: Self::Matrix) {
        self.theta = theta;
    }

    fn convergence_evaluation(&mut self) {}

    fn converged(&self) -> bool {
        true
    }

    fn evaluation(&mut self) -> Result<(), (Error, NPResult)> {
        self.psi = psi(
            &self.equation,
            &self.data,
            &self.theta,
            &ErrorModel::new(self.c, self.gamma, &self.error_type),
            false,
            false,
        );
        (self.w, self.objf) = burke(&self.psi).expect("Error in IPM");
        Ok(())
    }

    fn condensation(&mut self) -> Result<(), (Error, NPResult)> {
        Ok(())
    }
    fn optimizations(&mut self) -> Result<(), (Error, NPResult)> {
        Ok(())
    }

    fn logs(&self) {}

    fn expansion(&mut self) -> Result<(), (Error, NPResult)> {
        Ok(())
    }
}
