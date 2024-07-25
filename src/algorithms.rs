use std::fs;
use std::path::Path;

use crate::prelude::{self, settings::Settings};

use anyhow::{bail, Result};
use anyhow::{Context, Error};
use output::NPResult;
use pharmsol::prelude::{data::Data, simulator::Equation};
use prelude::*;
use tokio::sync::mpsc;
// use self::{data::Subject, simulator::Equation};

pub mod npag;
pub mod npod;
pub mod postprob;

pub trait Algorithm {
    fn fit(&mut self) -> Result<NPResult>;
    fn to_npresult(&self) -> NPResult;
}

pub fn initialize_algorithm(
    equation: Equation,
    settings: Settings,
    data: Data,
    tx: Option<mpsc::UnboundedSender<Comm>>,
) -> Result<Box<dyn Algorithm>, Error> {
    // If a stop file exists in the current directory, remove it
    if Path::new("stop").exists() {
        tracing::info!("Removing existing stop file prior to run");
        fs::remove_file("stop").context("Unable to remove previous stop file")?;
    }

    // Initialize the sample space
    let theta = initialization::sample_space(&settings, &data, &equation)?;

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
