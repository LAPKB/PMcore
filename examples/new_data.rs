#![allow(dead_code)]
// Redesign of data formats

use std::collections::HashMap;
use serde::Deserialize;

/// An Event represents a row in the datafile. It can be a [Bolus], [Infusion], or [Observation]
#[derive(Debug)]
enum Event {
    Bolus(Bolus),
    Infusion(Infusion),
    Observation(Observation),
}

/// An instantaenous input of drug
#[derive(Debug)]
struct Bolus {
    time: f64,
    amount: f64,
    compartment: usize,
}

/// A continuous dose of drug
#[derive(Debug)]
struct Infusion {
    time: f64,
    amount: f64,
    compartment: usize,
    duration: f64,
}

/// A CovLine holds the linear interpolation required for covariates
/// If the covariate should be carried forward, the slope is set to 0
#[derive(Debug)]
pub struct CovLine {
    slope: f64,
    intercept: f64,
}

impl CovLine {
    pub fn interp(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

/// An observation of drug concentration or covariates
#[derive(Debug)]
struct Observation {
    time: f64,
    value: f64,
    outeq: usize,
    errpoly: Option<(f64, f64, f64, f64)>,
    covs: HashMap<String, CovLine>,
}

/// A Block is a collection of events that are related to each other
#[derive(Debug)]
struct Block {
    events: Vec<Event>,
    covs: HashMap<String, CovLine>,
}

impl Block {
    // Sort events by time
}

// A row represents a single row in the datafile
#[derive(Deserialize)]
#[derive(Debug)]
struct Row {
    pub id: String,
    pub evid: isize,
    pub time: f64,
    pub dur: Option<f64>,
    pub dose: Option<f64>,
    pub addl: Option<isize>,
    pub ii: Option<isize>,
    pub input: Option<usize>,
    pub out: Option<f64>,
    pub outeq: Option<usize>,
    pub c0: Option<f32>,
    pub c1: Option<f32>,
    pub c2: Option<f32>,
    pub c3: Option<f32>,
    #[serde(flatten)]
    pub covs: HashMap<String, Option<f64>>,
}

/// Data is a collection of blocks for one individual
#[derive(Debug)]
struct Data {
    id: String,
    blocks: Vec<Block>,
}

fn main() {

}
