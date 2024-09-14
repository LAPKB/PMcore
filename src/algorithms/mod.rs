use std::fs;
use std::path::Path;

use crate::prelude::{self, settings::Settings};

use anyhow::{bail, Result};
use anyhow::{Context, Error};
use ndarray::Array2;
use npag::*;
use npod::NPOD;
use output::NPResult;
use pharmsol::prelude::{data::Data, simulator::Equation};
use postprob::POSTPROB;
use prelude::*;
// use self::{data::Subject, simulator::Equation};

pub mod npag;
pub mod npod;
pub mod postprob;

pub trait Algorithm<E: Equation> {
    fn new(config: Settings, equation: E, data: Data) -> Result<Box<Self>, Error>
    where
        Self: Sized;
    fn get_settings(&self) -> &Settings;
    fn get_data(&self) -> &Data;
    fn get_prior(&self) -> Array2<f64>;
    fn inc_cycle(&mut self) -> usize;
    fn set_theta(&mut self, theta: Array2<f64>);
    fn get_theta(&self) -> &Array2<f64>;
    fn convergence_evaluation(&mut self);
    fn converged(&self) -> bool;
    fn initialize(&mut self) -> Result<(), Error> {
        // If a stop file exists in the current directory, remove it
        if Path::new("stop").exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file("stop").context("Unable to remove previous stop file")?;
        }
        self.set_theta(self.get_prior());
        Ok(())
    }
    fn evaluation(&mut self) -> Result<(), (Error, NPResult<E>)>;
    fn condensation(&mut self) -> Result<(), (Error, NPResult<E>)>;
    fn optimizations(&mut self) -> Result<(), (Error, NPResult<E>)>;
    fn logs(&self);
    fn expansion(&mut self) -> Result<(), (Error, NPResult<E>)>;
    fn next_cycle(&mut self) -> Result<bool, (Error, NPResult<E>)> {
        let span = tracing::info_span!("", Cycle = 1);
        let _enter = span.enter();
        if self.inc_cycle() > 1 {
            self.expansion()?;
        }
        self.evaluation()?;
        self.condensation()?;
        self.optimizations()?;
        self.logs();
        self.convergence_evaluation();
        Ok(self.converged())
    }
    fn fit(&mut self) -> Result<NPResult<E>, (Error, NPResult<E>)> {
        self.initialize().unwrap();
        while !self.next_cycle()? {}
        Ok(self.into_npresult())
    }
    fn into_npresult(&self) -> NPResult<E>;
}

pub fn dispatch_algorithm<E: Equation>(
    settings: Settings,
    equation: E,
    data: Data,
) -> Result<Box<dyn Algorithm<E>>, Error> {
    match settings.config.algorithm.as_str() {
        "NPAG" => Ok(NPAG::new(settings, equation, data)?),
        "NPOD" => Ok(NPOD::new(settings, equation, data)?),
        "POSTPROB" => Ok(POSTPROB::new(settings, equation, data)?),
        alg => bail!("Algorithm {} not implemented", alg),
    }
}
