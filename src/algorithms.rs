use crate::prelude::{self, settings::settings::Settings};

use output::NPResult;
use prelude::{datafile::Scenario, *};
use simulation::predict::{Engine, Predict};
use tokio::sync::mpsc;

mod npag;
mod npod;
mod postprob;

pub trait Algorithm {
    fn fit(&mut self) -> NPResult;
    fn to_npresult(&self) -> NPResult;
}

pub fn initialize_algorithm<S>(
    engine: Engine<S>,
    settings: Settings,
    scenarios: Vec<Scenario>,
    tx: mpsc::UnboundedSender<Comm>,
) -> Box<dyn Algorithm>
where
    S: Predict<'static> + std::marker::Sync + Clone + 'static,
{
    if std::path::Path::new("stop").exists() {
        match std::fs::remove_file("stop") {
            Ok(_) => tracing::info!("Removed previous stop file"),
            Err(err) => panic!("Unable to remove previous stop file: {}", err),
        }
    }
    let ranges = settings.run.random.ranges.clone();
    let theta = initialization::sample_space(&settings, &ranges);

    //This should be a macro, so it can automatically expands as soon as we add a new option in the Type Enum
    match settings.run.config.engine.as_str() {
        "NPAG" => Box::new(npag::NPAG::new(
            engine,
            ranges,
            theta,
            scenarios,
            settings.run.error.poly,
            tx,
            settings,
        )),
        "NPOD" => Box::new(npod::NPOD::new(
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
