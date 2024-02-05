#![allow(dead_code)]
// Redesign of data formats

use serde::Deserialize;
use std::fmt;
use std::{collections::HashMap, error::Error};

/// An Event represents a row in the datafile. It can be a [Bolus], [Infusion], or [Observation]
#[derive(Debug)]
pub enum Event {
    Bolus(Bolus),
    Infusion(Infusion),
    Observation(Observation),
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Event::Bolus(bolus) => write!(
                f,
                "Bolus at time {}: amount {} in compartment {}",
                bolus.time, bolus.amount, bolus.compartment
            ),
            Event::Infusion(infusion) => write!(
                f,
                "Infusion at time {}: amount {} in compartment {} with duration {}",
                infusion.time, infusion.amount, infusion.compartment, infusion.duration
            ),
            Event::Observation(observation) => write!(
                f,
                "Observation at time {}: value {} in outeq {}",
                observation.time, observation.value, observation.outeq
            ),
        }
    }
}

/// An instantaenous input of drug
#[derive(Debug)]
pub struct Bolus {
    time: f64,
    amount: f64,
    compartment: usize,
}

/// A continuous dose of drug
#[derive(Debug)]
pub struct Infusion {
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
pub struct Observation {
    time: f64,
    value: f64,
    outeq: usize,
    errpoly: Option<(f64, f64, f64, f64)>,
    covs: HashMap<String, CovLine>,
}

/// A Block is a collection of events that are related to each other
#[derive(Debug)]
pub struct Block {
    events: Vec<Event>,
    index: usize,
    covs: HashMap<String, CovLine>,
}

// Implement Display for Block
impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Block {}:", self.index)?;
        for event in &self.events {
            writeln!(f, "  {}", event)?;
        }
        Ok(())
    }
}

// A row represents a single row in the datafile
#[derive(Deserialize, Debug)]
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

impl Row {
    pub fn get_errorpoly(&self) -> Option<(f64, f64, f64, f64)> {
        match (self.c0, self.c1, self.c2, self.c3) {
            (Some(c0), Some(c1), Some(c2), Some(c3)) => {
                Some((c0 as f64, c1 as f64, c2 as f64, c3 as f64))
            }
            _ => None,
        }
    }
}

/// Scenario is a collection of blocks for one individual
#[derive(Debug)]
pub struct Scenario {
    id: String,
    blocks: Vec<Block>,
}

// Implement Display for Scenario
impl fmt::Display for Scenario {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Scenario for ID: {}", self.id)?;
        for block in &self.blocks {
            writeln!(f, "{}", block)?;
        }
        Ok(())
    }
}

pub fn read_datafile(path: &str) -> Result<Vec<Scenario>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .has_headers(true)
        .from_path(path)?;

    let mut scenarios: Vec<Scenario> = Vec::new();

    // Read the datafile into a hashmap of rows by ID
    let mut rows_map: HashMap<String, Vec<Row>> = HashMap::new();
    for result in rdr.deserialize() {
        let row: Row = result?;

        rows_map
            .entry(row.id.clone())
            .or_insert_with(Vec::new)
            .push(row);
    }

    // Sort the rows by time and EVID (descending for EVID)
    for (_, rows) in &mut rows_map {
        rows.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.evid.cmp(&a.evid))
        });
    }

    // Convert grouped rows to scenarios
    for (id, rows) in rows_map {
        let mut blocks: Vec<Block> = Vec::new();
        let mut current_block = Block {
            events: Vec::new(),
            index: 0,
            covs: HashMap::new(),
        };
        let mut block_index = 0;

        for row in rows {
            match row.evid {
                // Dose with or without reset
                1 | 4 => {
                    if row.evid == 4 {
                        // Finish current block and start a new one
                        if !current_block.events.is_empty() {
                            blocks.push(current_block);
                            block_index += 1;
                        }
                        current_block = Block {
                            events: Vec::new(),
                            index: block_index,
                            covs: HashMap::new(),
                        };
                    }

                    let event = if let Some(dur) = row.dur {
                        Event::Infusion(Infusion {
                            time: row.time,
                            amount: row.dose.expect("Infusion amount (DOSE) is missing"),
                            compartment: row
                                .input
                                .expect("Infusion compartment (INPUT) is missing"),
                            duration: dur,
                        })
                    } else {
                        Event::Bolus(Bolus {
                            time: row.time,
                            amount: row.dose.expect("Bolus amount (DOSE) is missing"),
                            compartment: row.input.expect("Bolus compartment (INPUT) is missing"),
                        })
                    };
                    current_block.events.push(event);
                }
                // Observation
                0 => {
                    let observation = Observation {
                        time: row.time,
                        value: row.out.expect("Observation OUT is missing"),
                        outeq: row.outeq.expect("Observation OUTEQ is missing"),
                        errpoly: row.get_errorpoly(),
                        covs: HashMap::new(), // TODO: Populate with covariate data
                    };
                    current_block.events.push(Event::Observation(observation));
                }
                _ => {
                    panic!(
                        "Unknown EVID: {} for ID {} at time {}",
                        row.evid, id, row.time
                    )
                }
            }
        }

        // Don't forget to add the last block for the current ID
        if !current_block.events.is_empty() {
            blocks.push(current_block);
        }

        let scenario = Scenario { id, blocks };
        scenarios.push(scenario);
    }

    Ok(scenarios)
}

fn main() {
    let scenarios = read_datafile("examples/data/bimodal_ke_blocks.csv").unwrap();
    for scenario in scenarios {
        println!("{}", scenario);
    }
}
