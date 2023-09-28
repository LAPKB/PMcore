use crate::prelude::{self, datafile::Scenario, output::NPCycle, settings::run::Data};
use ndarray::Array2;
use output::NPResult;
use prelude::*;
use simulation::predict::{Engine, Predict};
use tokio::sync::mpsc::UnboundedSender;

pub mod npag;

pub enum Type {
    NPAG,
}

pub trait Algorithm<S> {
    // fn initialize(
    //     self,
    //     sim_eng: Engine<S>,
    //     ranges: Vec<(f64, f64)>,
    //     theta: Array2<f64>,
    //     scenarios: Vec<Scenario>,
    //     c: (f64, f64, f64, f64),
    //     tx: UnboundedSender<NPCycle>,
    //     settings: Data,
    // ) -> Self
    // where
    //     S: Predict + std::marker::Sync;
    fn fit(&mut self) -> (Engine<S>, NPResult)
    where
        S: Predict + std::marker::Sync + Clone;
}

pub fn initialize_algorithm<S>(
    alg_type: Type,
    sim_eng: Engine<S>,
    ranges: Vec<(f64, f64)>,
    theta: Array2<f64>,
    scenarios: Vec<Scenario>,
    c: (f64, f64, f64, f64),
    tx: UnboundedSender<NPCycle>,
    settings: Data,
) -> Box<dyn Algorithm<S>>
where
    S: Predict + std::marker::Sync + 'static + Clone,
{
    match alg_type {
        Type::NPAG => Box::new(npag::NPAG::new(
            sim_eng, ranges, theta, scenarios, c, tx, settings,
        )),
    }
}
