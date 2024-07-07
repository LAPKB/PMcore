use crate::prelude::{self, settings::Settings};

use anyhow::Result;
use output::NPResult;
use pharmsol::prelude::{data::Data, simulator::Equation};
use prelude::*;
use tokio::sync::mpsc;
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
) -> Box<dyn Algorithm> {
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
        "NPAG" => Box::new(npag::NPAG::new(
            equation,
            ranges,
            theta,
            data,
            settings.error.poly,
            tx,
            settings,
        )),
        "NPOD" => Box::new(npod::NPOD::new(
            equation,
            ranges,
            theta,
            data,
            settings.error.poly,
            tx,
            settings,
        )),
        "POSTPROB" => Box::new(postprob::POSTPROB::new(
            equation,
            theta,
            data,
            settings.error.poly,
            tx,
            settings,
        )),
        alg => {
            eprintln!("Error: Algorithm not recognized: {}", alg);
            std::process::exit(-1)
        }
    }
}
