use crate::prelude::{self, datafile::Scenario, output::NPCycle, settings::run::Data};
use ndarray::Array2;
use output::NPResult;
use prelude::*;
use simulation::predict::{Engine, Predict};
use tokio::sync::mpsc::UnboundedSender;

pub mod npag;

pub trait Algorithm<S, T = Self> {
    fn initialize(
        sim_eng: Engine<S>,
        ranges: Vec<(f64, f64)>,
        theta: Array2<f64>,
        scenarios: Vec<Scenario>,
        c: (f64, f64, f64, f64),
        tx: UnboundedSender<NPCycle>,
        settings: Data,
    ) -> T
    where
        S: Predict + std::marker::Sync;
    fn fit(self) -> (Engine<S>, NPResult)
    where
        S: Predict + std::marker::Sync;
}
