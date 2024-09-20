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
    fn get_cycle(&mut self) -> usize;
    fn inc_cycle(&mut self);
    fn set_theta(&mut self, theta: Array2<f64>);
    fn get_theta(&self) -> &Array2<f64>;
    // Has the algorithm converged?
    fn convergence(&mut self) -> bool;
    /// Should the algorithm stop?
    fn stop(&mut self) -> bool {
        // Stop if we have reached maximum number of cycles
        if self.get_cycle() >= self.get_settings().config.cycles {
            tracing::warn!("Maximum number of cycles reached - program will sotp");
            return true;
        }

        // Stop if stopfile exists
        if std::path::Path::new("stop").exists() {
            tracing::warn!("Stopfile detected - program will stop");
            return true;
        } else {
            return false;
        }
    }
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
    fn logs(&mut self);
    fn expansion(&mut self) -> Result<(), (Error, NPResult<E>)>;
    fn next_cycle(&mut self) -> Result<bool, (Error, NPResult<E>)> {
        self.inc_cycle();
        let _ = tracing::info_span!("", Cycle = self.get_cycle()).enter();
        if self.get_cycle() > 1 {
            self.expansion()?;
        }
        self.evaluation()?;
        self.condensation()?;
        self.optimizations()?;
        self.logs();
        let converged = self.convergence();
        let stop = self.stop();
        Ok(converged || stop)
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
