#![allow(dead_code)]
// Redesign of data formats

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;
use std::{collections::HashMap, error::Error};

/// An Event represents a row in the datafile. It can be a [Bolus], [Infusion], or [Observation]
#[derive(Debug, Clone)]
pub enum Event {
    Bolus(Bolus),
    Infusion(Infusion),
    Observation(Observation),
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Event::Bolus(bolus) => {
                write!(
                    f,
                    "Bolus at time {}: amount {} in compartment {}",
                    bolus.time, bolus.amount, bolus.compartment
                )?;
                // Display covariates
                if !bolus.covs.is_empty() {
                    write!(f, ", covariates: ")?;
                    for (key, cov) in &bolus.covs {
                        write!(f, "{}:{:?}", key, cov)?;
                    }
                }
            }
            Event::Infusion(infusion) => {
                write!(
                    f,
                    "Infusion at time {}: amount {} in compartment {} with duration {}",
                    infusion.time, infusion.amount, infusion.compartment, infusion.duration
                )?;
                // Display covariates
                if !infusion.covs.is_empty() {
                    write!(f, ", covariates: ")?;
                    for (key, cov) in &infusion.covs {
                        write!(f, "{}:{:?}", key, cov)?;
                    }
                }
            }
            Event::Observation(observation) => {
                write!(
                    f,
                    "Observation at time {}: value {} in outeq {}, ignore: {}",
                    observation.time, observation.value, observation.outeq, observation.ignore
                )?;
                // Display covariates
                if !observation.covs.is_empty() {
                    write!(f, ", covariates: ")?;
                    for (key, cov) in &observation.covs {
                        write!(f, "{}:{:?}", key, cov)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// An instantaenous input of drug
#[derive(Debug, Clone)]
pub struct Bolus {
    time: f64,
    amount: f64,
    compartment: usize,
    covs: HashMap<String, CovLine>,
}

/// A continuous dose of drug
#[derive(Debug, Clone)]
pub struct Infusion {
    time: f64,
    amount: f64,
    compartment: usize,
    duration: f64,
    covs: HashMap<String, CovLine>,
}

/// A CovLine holds the linear interpolation required for covariates
/// If the covariate should be carried forward, the slope is set to 0
#[derive(Debug, Clone)]
pub struct CovLine {
    slope: f64,
    intercept: f64,
    time: f64,
}

impl CovLine {
    /// Interpolate the covariate value at a given time from the last provided value
    pub fn interp(&self, t: f64) -> f64 {
        let delta_t = t - self.time;
        if delta_t < 0.0 {
            panic!("Interpolation time is before the last provided value")
        }
        self.slope * (t - self.time) + self.intercept
    }
}

/// An observation of drug concentration or covariates
#[derive(Debug, Clone)]
pub struct Observation {
    time: f64,
    value: f64,
    outeq: usize,
    errpoly: Option<(f64, f64, f64, f64)>,
    covs: HashMap<String, CovLine>,
    ignore: bool,
}

/// A Block is a collection of events that are related to each other
#[derive(Debug)]
pub struct Block {
    events: Vec<Event>,
    index: usize,
}

impl Block {
    /// Get times which will be stepped over in the simulation
    ///
    /// This function returns a Vec of times which are the start and end times of each event in the [Block]
    /// plus the start time of each [Observation]
    ///
    /// Additionally, the function takes an optional argument `idelta` which is the time step for the simulation
    /// If `idelta` is provided, it will additionally add times at `idelta` intervals between zero and the last time in the [Scenario]
    pub fn get_times(&self, idelta: f64) -> Vec<f64> {
        let mut times: Vec<f64> = Vec::new();
        for event in &self.events {
            match event {
                Event::Bolus(bolus) => {
                    times.push(bolus.time);
                }
                Event::Infusion(infusion) => {
                    times.push(infusion.time);
                    times.push(infusion.time + infusion.duration);
                }
                Event::Observation(observation) => {
                    times.push(observation.time);
                }
            }
        }

        // Add times at `idelta` intervals from zero to the last time in the scenario
        if let Some(last_time) = times.last() {
            let intervals = (last_time / idelta).ceil() as usize;
            for i in 1..intervals {
                times.push(i as f64 * idelta);
            }
        }
        // Sort and remove duplicates
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times.dedup();
        times
    }

    /// Calculate the covariate lines for the block
    pub fn calculate_covlines(&self) {
        !todo!("Calculate covariate lines for the block")
    }
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
    #[serde(deserialize_with = "deserialize_option_f64")]
    pub dur: Option<f64>,
    #[serde(deserialize_with = "deserialize_option_f64")]
    pub dose: Option<f64>,
    #[serde(deserialize_with = "deserialize_option_isize")]
    pub addl: Option<isize>,
    #[serde(deserialize_with = "deserialize_option_isize")]
    pub ii: Option<isize>,
    #[serde(deserialize_with = "deserialize_option_usize")]
    pub input: Option<usize>,
    #[serde(deserialize_with = "deserialize_option_f64")]
    pub out: Option<f64>,
    #[serde(deserialize_with = "deserialize_option_usize")]
    pub outeq: Option<usize>,
    #[serde(deserialize_with = "deserialize_option_f64")]
    pub c0: Option<f64>,
    #[serde(deserialize_with = "deserialize_option_f64")]
    pub c1: Option<f64>,
    #[serde(deserialize_with = "deserialize_option_f64")]
    pub c2: Option<f64>,
    #[serde(deserialize_with = "deserialize_option_f64")]
    pub c3: Option<f64>,
    #[serde(deserialize_with = "deserialize_covs", flatten)]
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

    pub fn get_covs(&self) -> HashMap<String, CovLine> {
        let mut covs = HashMap::new();
        for (key, value) in &self.covs {
            if let Some(v) = value {
                covs.insert(
                    key.clone(),
                    CovLine {
                        slope: 0.0,
                        intercept: *v,
                        time: self.time,
                    },
                );
            }
        }
        covs
    }
}

/// Scenario is a collection of blocks for one individual
#[derive(Debug)]
pub struct Scenario {
    id: String,
    blocks: Vec<Block>,
}

impl Scenario {
    /// Sort events by time, then by [Event] type so that [Bolus] and [Infusion] come before [Observation]
    pub fn sort(&mut self) {
        for block in &mut self.blocks {
            block.events.sort_by(|a, b| {
                // First, compare times using partial_cmp, then compare types if times are equal.
                let time_a = match a {
                    Event::Bolus(bolus) => bolus.time,
                    Event::Infusion(infusion) => infusion.time,
                    Event::Observation(observation) => observation.time,
                };
                let time_b = match b {
                    Event::Bolus(bolus) => bolus.time,
                    Event::Infusion(infusion) => infusion.time,
                    Event::Observation(observation) => observation.time,
                };

                match time_a.partial_cmp(&time_b) {
                    Some(std::cmp::Ordering::Equal) => {
                        // If times are equal, sort by event type.
                        let type_order_a = match a {
                            Event::Bolus(_) => 1,
                            Event::Infusion(_) => 2,
                            Event::Observation(_) => 3,
                        };
                        let type_order_b = match b {
                            Event::Bolus(_) => 1,
                            Event::Infusion(_) => 2,
                            Event::Observation(_) => 3,
                        };
                        type_order_a.cmp(&type_order_b)
                    }
                    other => other.unwrap_or(std::cmp::Ordering::Equal),
                }
            });
        }
    }

    /// Add lagtime to all [Bolus] events in a [Scenario]
    ///
    /// Lagtime is a HashMap with the compartment number as key and the lagtime (f64) as value
    pub fn add_lagtime(&mut self, lagtime: HashMap<usize, f64>) {
        for block in &mut self.blocks {
            for event in block.events.iter_mut() {
                if let Event::Bolus(bolus) = event {
                    if let Some(lag) = lagtime.get(&bolus.compartment) {
                        bolus.time += lag;
                    }
                }
            }
        }
        // Sort the events after adding lagtime
        self.sort();
    }
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

    // Convert grouped rows to scenarios
    for (id, rows) in rows_map {
        let mut blocks: Vec<Block> = Vec::new();
        let mut current_block = Block {
            events: Vec::new(),
            index: 0,
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
                        };
                    }

                    let event = match row.dur {
                        Some(dur) if dur > 0.0 => Event::Infusion(Infusion {
                            time: row.time,
                            amount: row.dose.expect("Infusion amount (DOSE) is missing"),
                            compartment: row
                                .input
                                .expect("Infusion compartment (INPUT) is missing"),
                            duration: dur,
                            covs: row.get_covs(),
                        }),
                        _ => Event::Bolus(Bolus {
                            time: row.time,
                            amount: row.dose.expect("Bolus amount (DOSE) is missing"),
                            compartment: row.input.expect("Bolus compartment (INPUT) is missing"),
                            covs: row.get_covs(),
                        }),
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
                        covs: row.get_covs(),
                        ignore: if row.out == Some(-99.0) { true } else { false },
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

fn deserialize_option_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    if s == "" || s == "." {
        Ok(None)
    } else {
        f64::from_str(&s)
            .map(Some)
            .map_err(serde::de::Error::custom)
    }
}

fn deserialize_option_isize<'de, D>(deserializer: D) -> Result<Option<isize>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    if s == "" || s == "." {
        Ok(None)
    } else {
        isize::from_str(&s)
            .map(Some)
            .map_err(serde::de::Error::custom)
    }
}

fn deserialize_option_usize<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    if s == "" || s == "." {
        Ok(None)
    } else {
        usize::from_str(&s)
            .map(Some)
            .map_err(serde::de::Error::custom)
    }
}

fn deserialize_covs<'de, D>(deserializer: D) -> Result<HashMap<String, Option<f64>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct CovsVisitor;

    impl<'de> Visitor<'de> for CovsVisitor {
        type Value = HashMap<String, Option<f64>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str(
                "a map of string keys to optionally floating-point numbers or placeholders",
            )
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut covs = HashMap::new();
            while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                let opt_value = match value {
                    serde_json::Value::String(s) => match s.as_str() {
                        "" => None,
                        "." => None,
                        _ => match s.parse::<f64>() {
                            Ok(val) => Some(val),
                            Err(_) => {
                                return Err(de::Error::custom(
                                    "expected a floating-point number or empty string",
                                ))
                            }
                        },
                    },
                    serde_json::Value::Number(n) => Some(n.as_f64().unwrap()),
                    _ => return Err(de::Error::custom("expected a string or number")),
                };
                covs.insert(key, opt_value);
            }
            Ok(covs)
        }
    }

    deserializer.deserialize_map(CovsVisitor)
}

fn main() {
    // Parse scenarios
    let scenarios = read_datafile("examples/data/bimodal_ke_blocks.csv").unwrap();

    for mut scenario in scenarios {
        // Apply lagtime adjustments if necessary
        let mut lagtimes = HashMap::new();
        lagtimes.insert(1, 1.0); // Example adjustment
        scenario.add_lagtime(lagtimes);

        // Calculate covariate lines for each block
        for block in &mut scenario.blocks {
            block.calculate_covlines();
        }

        // Now that covariate lines are calculated, print the scenario
        println!("{}", scenario);
    }
}
