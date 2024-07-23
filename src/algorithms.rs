use crate::prelude::{self, settings::Settings};

use anyhow::{bail, Result};
use output::NPResult;
use pharmsol::prelude::{data::Data, simulator::Equation};
use prelude::*;
use tokio::sync::mpsc;
use anyhow::Error;
// use self::{data::Subject, simulator::Equation};

mod npag;
mod npod;
mod postprob;

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
    if std::path::Path::new("stop").exists() {
        match std::fs::remove_file("stop") {
            Ok(_) => tracing::info!("Removed previous stop file"),
            Err(err) => panic!("Unable to remove previous stop file: {}", err),
        }
    }
    let ranges = settings.random.ranges();
    let theta = initialization::sample_space(&settings, &ranges, &data, &equation);

    //This should be a macro, so it can automatically expands as soon as we add a new option in the Type Enum
    match settings.config.engine.as_str() {
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
        alg => {
            bail!("Algorithm {} not implemented", alg);
        }
    }
}
