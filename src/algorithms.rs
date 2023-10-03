use crate::prelude::{self, output::NPCycle, settings::run::Data};

use output::NPResult;
use prelude::*;
use simulation::predict::{Engine, Predict};
use tokio::sync::mpsc;

mod npag;
mod postprob;

pub enum Type {
    NPAG,
    POSTPROB,
}

pub trait Algorithm {
    fn fit(&mut self) -> NPResult;
    fn to_npresult(&self) -> NPResult;
}

pub fn initialize_algorithm<S>(
    engine: Engine<S>,
    settings: Data,
    tx: mpsc::UnboundedSender<NPCycle>,
) -> Box<dyn Algorithm>
where
    S: Predict + std::marker::Sync + Clone + 'static,
{
    if std::path::Path::new("stop").exists() {
        match std::fs::remove_file("stop") {
            Ok(_) => log::info!("Removed previous stop file"),
            Err(err) => panic!("Unable to remove previous stop file: {}", err),
        }
    }
    let ranges = settings.computed.random.ranges.clone();
    let theta = initialization::sample_space(&settings, &ranges);
    let mut scenarios = datafile::parse(&settings.parsed.paths.data).unwrap();
    if let Some(exclude) = &settings.parsed.config.exclude {
        for val in exclude {
            scenarios.remove(val.as_integer().unwrap() as usize);
        }
    }
    //This should be a macro, so it can automatically expands as soon as we add a new option in the Type Enum
    match settings.parsed.config.engine.as_str() {
        "NPAG" => Box::new(npag::NPAG::new(
            engine,
            ranges,
            theta,
            scenarios,
            settings.parsed.error.poly,
            tx,
            settings,
        )),
        "POSTPROB" => Box::new(postprob::POSTPROB::new(
            engine,
            theta,
            scenarios,
            settings.parsed.error.poly,
            tx,
            settings,
        )),
        alg => {
            eprintln!("Error: Algorithm not recognized: {}", alg);
            std::process::exit(-1)
        }
    }
}
