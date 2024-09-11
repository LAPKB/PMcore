use std::fs;
use std::path::Path;

use crate::prelude::{self, settings::Settings};

use anyhow::{bail, Result};
use anyhow::{Context, Error};
use output::NPResult;
use pharmsol::prelude::{data::Data, simulator::Equation};
use prelude::*;
use settings::Config;
use tokio::sync::mpsc;
// use self::{data::Subject, simulator::Equation};

pub mod npag;
pub mod npod;
pub mod postprob;

pub trait Algorithm: Sized {
    type Matrix;
    fn new(config: &Config) -> Result<Self>;
    fn get_settings(&self) -> &Settings;
    fn get_data(&self) -> &Data;
    fn get_prior(&self) -> Self::Matrix;
    fn set_theta(&mut self, theta: Self::Matrix);
    fn initialize(&mut self) -> Result<(), Error> {
        // If a stop file exists in the current directory, remove it
        if Path::new("stop").exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file("stop").context("Unable to remove previous stop file")?;
        }
        // Initialize the sample space
        self.set_theta(self.get_prior());
        // let theta = initialization::sample_space(
        //     self.get_settings(),
        //     self.get_data(),
        //     self.get_equation(),
        // )?;
        Ok(())
    }
    fn converge_criteria(&self) -> bool;
    fn evaluation(&mut self) -> Result<(), Error>;
    fn filter(&mut self) -> Result<(), Error>;
    fn expansion(&mut self) -> Result<(), Error>;
    fn fit(mut self) -> Result<NPResult, Error> {
        self.initialize()?;
        while !self.converge_criteria() {
            self.evaluation()?;
            self.filter()?;
            self.expansion()?;
        }
        Ok(self.to_npresult())
    }
    fn to_npresult(self) -> NPResult;
}

pub fn initialize_algorithm(
    equation: impl Equation,
    settings: Settings,
    data: Data,
    tx: Option<mpsc::UnboundedSender<Comm>>,
) -> Result<Box<dyn Algorithm>, Error> {
    //This should be a macro, so it can automatically expands as soon as we add a new option in the Type Enum
    let ranges = settings.random.ranges();
    match settings.config.algorithm.as_str() {
        "NPAG" => Ok(Box::new(npag::NPAG::new(
            equation,
            ranges,
            theta,
            data,
            settings.error.poly,
            tx,
            settings,
        ))),
        "NPOD" => Ok(Box::new(npod::NPOD::new(
            equation,
            ranges,
            theta,
            data,
            settings.error.poly,
            tx,
            settings,
        ))),
        "POSTPROB" => Ok(Box::new(postprob::POSTPROB::new(
            equation,
            theta,
            data,
            settings.error.poly,
            tx,
            settings,
        ))),
        alg => bail!("Algorithm {} not implemented", alg),
    }
}
