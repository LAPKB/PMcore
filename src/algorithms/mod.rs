use std::fs;
use std::path::Path;

use crate::routines::output::NPResult;
use crate::routines::settings::Settings;
use crate::structs::psi::Psi;
use crate::structs::theta::Theta;
use anyhow::Context;
use anyhow::Result;
use npag::*;
use npod::NPOD;
use pharmsol::prelude::{data::Data, simulator::Equation};
use postprob::POSTPROB;
use serde::{Deserialize, Serialize};

// use self::{data::Subject, simulator::Equation};

pub mod npag;
pub mod npod;
pub mod postprob;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Algorithm {
    NPAG,
    NPOD,
    POSTPROB,
}

pub trait Algorithms<E: for<'a> Equation<'a>>: Sync {
    fn new(config: Settings, equation: E, data: Data) -> Result<Box<Self>>
    where
        Self: Sized;
    fn validate_psi(&mut self) -> Result<()> {
        Ok(())
    }
    fn get_settings(&self) -> &Settings;
    fn equation(&self) -> &E;
    fn get_data(&self) -> &Data;
    fn get_prior(&self) -> Theta;
    fn inc_cycle(&mut self) -> usize;
    fn get_cycle(&self) -> usize;
    fn set_theta(&mut self, theta: Theta);
    fn get_theta(&self) -> &Theta;
    fn psi(&self) -> &Psi;
    fn likelihood(&self) -> f64;
    fn n2ll(&self) -> f64 {
        -2.0 * self.likelihood()
    }
    fn convergence_evaluation(&mut self);
    fn converged(&self) -> bool;
    fn initialize(&mut self) -> Result<()> {
        // If a stop file exists in the current directory, remove it
        if Path::new("stop").exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file("stop").context("Unable to remove previous stop file")?;
        }
        self.set_theta(self.get_prior());
        Ok(())
    }
    fn evaluation(&mut self) -> Result<()>;
    fn condensation(&mut self) -> Result<()>;
    fn optimizations(&mut self) -> Result<()>;
    fn logs(&self);
    fn expansion(&mut self) -> Result<()>;
    fn next_cycle(&mut self) -> Result<bool> {
        if self.inc_cycle() > 1 {
            self.expansion()?;
        }
        let span = tracing::info_span!("", "{}", format!("Cycle {}", self.get_cycle()));
        let _enter = span.enter();
        self.evaluation()?;
        self.condensation()?;
        self.optimizations()?;
        self.logs();
        self.convergence_evaluation();
        Ok(self.converged())
    }
    fn fit(&mut self) -> Result<NPResult<E>> {
        self.initialize().unwrap();
        while !self.next_cycle()? {}
        Ok(self.into_npresult())
    }

    fn into_npresult(&self) -> NPResult<E>;
}

pub fn dispatch_algorithm<E: for<'a> Equation<'a>>(
    settings: Settings,
    equation: E,
    data: Data,
) -> Result<Box<dyn Algorithms<E>>> {
    match settings.config().algorithm {
        Algorithm::NPAG => Ok(NPAG::new(settings, equation, data)?),
        Algorithm::NPOD => Ok(NPOD::new(settings, equation, data)?),
        Algorithm::POSTPROB => Ok(POSTPROB::new(settings, equation, data)?),
    }
}
