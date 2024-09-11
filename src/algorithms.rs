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
use settings::Config;
use tokio::sync::mpsc::{self, UnboundedSender};
// use self::{data::Subject, simulator::Equation};

pub mod npag;
pub mod npod;
pub mod postprob;

pub trait Algorithm<E: Equation> {
    type Matrix;
    fn new(
        config: Settings,
        equation: E,
        data: Data,
        tx: Option<UnboundedSender<Comm>>,
    ) -> Result<Box<Self>, Error>
    where
        Self: Sized;
    fn get_settings(&self) -> &Settings;
    fn get_data(&self) -> &Data;
    fn get_prior(&self) -> Self::Matrix;
    fn get_cycle(&self) -> usize;
    fn set_theta(&mut self, theta: Self::Matrix);
    fn end_cycle(&mut self);
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
    fn evaluation(&mut self) -> Result<(), (Error, NPResult)>;
    fn filter(&mut self) -> Result<(), (Error, NPResult)>;
    fn optimizations(&mut self) -> Result<(), (Error, NPResult)>;
    fn logs(&self);
    fn expansion(&mut self) -> Result<(), (Error, NPResult)>;
    fn fit(&mut self) -> Result<NPResult, (Error, NPResult)> {
        self.initialize().unwrap();
        loop {
            self.evaluation()?;
            self.filter()?;
            self.optimizations()?;
            self.logs();

            self.end_cycle();
            if self.converged() {
                break;
            }
            self.expansion()?;
        }
        Ok(self.to_npresult())
    }
    fn to_npresult(&self) -> NPResult;
}

pub fn dispatch_algorithm<E: Equation, M>(
    settings: Settings,
    equation: E,
    data: Data,
    tx: Option<UnboundedSender<Comm>>,
) -> Result<Box<dyn Algorithm<E, Matrix = Array2<f64>>>, Error> {
    match settings.config.algorithm.as_str() {
        "NPAG" => Ok(NPAG::new(settings, equation, data, tx)?),
        "NPOD" => Ok(NPOD::new(settings, equation, data, tx)?),
        "POSTPROB" => Ok(POSTPROB::new(settings, equation, data, tx)?),
        alg => bail!("Algorithm {} not implemented", alg),
    }
}
