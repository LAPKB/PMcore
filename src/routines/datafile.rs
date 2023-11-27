use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::process::exit;

type Record = HashMap<String, String>;

/// A Scenario is a collection of blocks that represent a single subject in the Datafile
#[derive(Debug, Clone)]
pub struct Scenario {
    pub id: String,
    pub blocks: Vec<Block>,
    pub obs: Vec<f64>,
    pub obs_times: Vec<f64>,
    pub times: Vec<f64>,
}

impl Scenario {
    pub fn new(events: Vec<Event>) -> Result<Self, Box<dyn Error>> {
        let mut scenario = Self::parse_events(events)?;
        scenario.inyect_covariates_regressions();
        Ok(scenario)
    }

    /// Adds "mock" events to a Scenario in order to generate predictions at those times
    /// The interval is mapped to the `idelta`-setting in the configuration time
    /// Time after dose (tad) will ensure that predictions are made until the last dose + tad
    pub fn add_event_interval(&self, interval: f64, tad: f64) -> Self {
        // Clone the underlying Event data instead of the references
        let all_events = self
            .clone()
            .blocks
            .iter()
            .flat_map(|block| block.events.iter().cloned())
            .collect::<Vec<_>>();

        // Determine the start and end times
        let start_time = all_events.first().unwrap().time;
        let mut end_time = all_events.last().unwrap().time;

        // Pad end time to accomodate time after dose
        if tad > 0.0 {
            let last_dose_time = all_events
                .iter()
                .filter(|event| event.evid == 1)
                .map(|event| event.time)
                .fold(std::f64::NEG_INFINITY, f64::max);

            if end_time < last_dose_time + tad {
                end_time = last_dose_time + tad
            }
        }

        // Determine the unique output equations in the events
        // TODO: This should be read from the model / engine
        let mut outeqs: Vec<_> = all_events.iter().filter_map(|event| event.outeq).collect();
        outeqs.sort_unstable();
        outeqs.dedup();

        // Generate dummy events
        let mut new_events = vec![];
        let mut current_time = start_time + interval; // Start from the first interval after the start time
        while current_time <= end_time {
            current_time = (current_time / interval).round() * interval; // Round to the nearest interval
            current_time = decimals(current_time, 4); // Round to 4 decimal places
            for outeq in &outeqs {
                new_events.push(Event {
                    id: self.id.clone(),
                    evid: 0,
                    time: current_time,
                    dur: None,
                    dose: None,
                    _addl: None,
                    _ii: None,
                    input: None,
                    out: Some(-99.0),
                    outeq: Some(*outeq),
                    _c0: None,
                    _c1: None,
                    _c2: None,
                    _c3: None,
                    covs: HashMap::new(),
                });
            }
            current_time += interval;
        }

        // Combine all_events with new_events
        let mut combined_events = all_events;
        combined_events.extend(new_events.iter().cloned());

        // Sort the events by time
        combined_events.sort_by(|a, b| a.cmp_by_id_then_time(b));
        // Remove duplicate events based on time and outeq
        // In essence, no need to have two predictions at the same time for the same outeq
        let time_tolerance = 1e-4;
        combined_events.sort_by(|a, b| a.cmp_by_id_then_time(b));
        combined_events.dedup_by(|a, b| {
            (a.time - b.time).abs() < time_tolerance && a.outeq == b.outeq && a.evid == b.evid
        });

        Scenario::new(combined_events).unwrap()
    }

    pub fn reorder_with_lag(&self, lag_inputs: Vec<(f64, usize)>) -> Self {
        if lag_inputs.is_empty() {
            return self.clone();
        }
        let mut events = Vec::new();
        for block in &self.blocks {
            for mut event in block.events.clone() {
                if event.evid == 1 {
                    for lag_term in &lag_inputs {
                        if event.input.unwrap() == lag_term.1 {
                            event.time += lag_term.0;
                        }
                    }
                }
                events.push(event);
            }
        }
        events.sort_by(|a, b| a.cmp_by_id_then_time(b));

        let mut scenario = Self::parse_events(events).unwrap();
        scenario.inyect_covariates_regressions();
        scenario
    }

    fn parse_events(events: Vec<Event>) -> Result<Self, Box<dyn Error>> {
        let id = events.first().unwrap().id.clone();
        let mut blocks: Vec<Block> = vec![];
        let mut block: Block = Block {
            events: vec![],
            covs: HashMap::new(),
        };
        let mut obs: Vec<f64> = vec![];
        let mut times: Vec<f64> = vec![];
        let mut obs_times: Vec<f64> = vec![];

        for mut event in events {
            times.push(event.time);
            //Covariate forward filling
            for (key, val) in &mut event.covs {
                if val.is_none() {
                    //This will fail if for example the block is empty, Doses always must have the cov values
                    *val = *block.events.last().unwrap().covs.get(key).unwrap();
                }
            }
            if event.evid == 1 {
                if event.dur.unwrap_or(0.0) > 0.0 {
                    check_infusion(&event)?;
                } else {
                    check_dose(&event)?;
                }

                if !block.events.is_empty() {
                    blocks.push(block);
                }
                block = Block {
                    events: vec![],
                    covs: HashMap::new(),
                };
                // clone the covs from the dose event and put them in the block
            } else if event.evid == 0 {
                check_obs(&event)?;
                obs_times.push(event.time);
                obs.push(event.out.unwrap());
            } else {
                tracing::error!("Error: Unsupported evid: {evid}", evid = event.evid);
                exit(-1);
            }
            block.events.push(event);
        }
        if !block.events.is_empty() {
            blocks.push(block);
        }
        Ok(Scenario {
            id,
            blocks,
            obs,
            obs_times,
            times,
        })
    }

    fn inyect_covariates_regressions(&mut self) {
        let mut b_it = self.blocks.iter_mut().peekable();
        while let Some(block) = b_it.next() {
            let mut block_covs: HashMap<String, CovLine> = HashMap::new();
            if let Some(next_block) = b_it.peek() {
                for (key, p_v) in &block.events.first().unwrap().covs {
                    let p_v = p_v.unwrap();
                    let p_t = block.events.first().unwrap().time;
                    let f_v = next_block
                        .events
                        .first()
                        .unwrap()
                        .covs
                        .get(key)
                        .unwrap()
                        .unwrap();
                    let f_t = next_block.events.first().unwrap().time;
                    let slope = (f_v - p_v) / (f_t - p_t);
                    let intercept = p_v - slope * p_t;
                    block_covs.insert(key.clone(), CovLine { intercept, slope });
                }
            } else {
                for (key, p_v) in &block.events.first().unwrap().covs {
                    let p_v = p_v.unwrap();
                    block_covs.insert(
                        key.clone(),
                        CovLine {
                            intercept: p_v,
                            slope: 0.0,
                        },
                    );
                }
            }
            block.covs = block_covs;
        }
    }
}
#[derive(Debug, Clone)]
pub struct Infusion {
    pub time: f64,
    pub dur: f64,
    pub amount: f64,
    pub compartment: usize,
}
#[derive(Debug, Clone)]
pub struct Dose {
    pub time: f64,
    pub amount: f64,
    pub compartment: usize,
}
#[derive(Debug, Clone)]
pub struct CovLine {
    slope: f64,
    intercept: f64,
}

impl CovLine {
    pub fn interp(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

// type Block = Vec<Event>;
#[derive(Debug, Clone)]
pub struct Block {
    pub events: Vec<Event>,
    pub covs: HashMap<String, CovLine>,
}

/// A Event represent a single row in the Datafile
#[derive(Debug, Clone)]
pub struct Event {
    pub id: String,
    pub evid: isize,
    pub time: f64,
    pub dur: Option<f64>,
    pub dose: Option<f64>,
    pub _addl: Option<isize>,
    pub _ii: Option<isize>,
    pub input: Option<usize>,
    pub out: Option<f64>,
    pub outeq: Option<usize>,
    pub _c0: Option<f32>,
    pub _c1: Option<f32>,
    pub _c2: Option<f32>,
    pub _c3: Option<f32>,
    pub covs: HashMap<String, Option<f64>>,
}

impl Event {
    pub fn cmp_by_id_then_time(&self, other: &Self) -> Ordering {
        match self.id.cmp(&other.id) {
            Ordering::Equal => self.time.partial_cmp(&other.time).unwrap(),
            other => other,
        }
    }
}

pub fn parse(path: &String) -> Result<Vec<Scenario>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        // .delimiter(b',')
        // .escape(Some(b'\\'))
        .comment(Some(b'#'))
        .from_path(path)
        .unwrap();
    let mut events: Vec<Event> = vec![];

    for result in rdr.deserialize() {
        let mut record: Record = result?;
        events.push(Event {
            id: record.remove("ID").unwrap(),
            evid: record.remove("EVID").unwrap().parse::<isize>().unwrap(),
            time: record.remove("TIME").unwrap().parse::<f64>().unwrap(),
            dur: record.remove("DUR").unwrap().parse::<f64>().ok(),
            dose: record.remove("DOSE").unwrap().parse::<f64>().ok(),
            _addl: record.remove("ADDL").unwrap().parse::<isize>().ok(), //TODO: To Be Implemented
            _ii: record.remove("II").unwrap().parse::<isize>().ok(),     //TODO: To Be Implemented
            input: record.remove("INPUT").unwrap().parse::<usize>().ok(),
            out: record.remove("OUT").unwrap().parse::<f64>().ok(),
            outeq: record.remove("OUTEQ").unwrap().parse::<usize>().ok(),
            _c0: record.remove("C0").unwrap().parse::<f32>().ok(), //TODO: To Be Implemented
            _c1: record.remove("C1").unwrap().parse::<f32>().ok(), //TODO: To Be Implemented
            _c2: record.remove("C2").unwrap().parse::<f32>().ok(), //TODO: To Be Implemented
            _c3: record.remove("C3").unwrap().parse::<f32>().ok(), //TODO: To Be Implemented
            covs: record
                .into_iter()
                .map(|(key, value)| {
                    let val = value.parse::<f64>().ok();
                    (key, val)
                })
                .collect(),
        });
    }

    let mut event_groups: HashMap<String, Vec<Event>> = HashMap::new();
    events.into_iter().for_each(|event| {
        event_groups
            .entry(event.id.clone())
            .or_insert_with(Vec::new)
            .push(event);
    });

    let mut scenarios: Vec<Scenario> = vec![];

    for (_id, s_events) in event_groups {
        let scenario = Scenario::new(s_events)?;
        scenarios.push(scenario);
    }

    scenarios.sort_by(|a, b| a.id.cmp(&b.id));

    Ok(scenarios)
}

fn check_dose(event: &Event) -> Result<(), Box<dyn Error>> {
    if event.dose.is_none() {
        tracing::error!("Error: Dose event without dose");
        //return Err("Error: Dose event without dose".into());
        exit(-1);
    }
    if event.input.is_none() {
        tracing::error!("Error: Dose event without input");
        //return Err("Error: Dose event without input".into());
        exit(-1);
    }
    Ok(())
}
fn check_infusion(event: &Event) -> Result<(), Box<dyn Error>> {
    if event.dose.is_none() {
        tracing::error!("Error: Infusion event without dose");
        //return Err("Error: Infusion event without dose".into());
        exit(-1);
    }
    if event.dur.is_none() {
        tracing::error!("Error: Infusion event without duration");
        //return Err("Error: Infusion event without duration".into());
        exit(-1);
    }
    if event.input.is_none() {
        tracing::error!("Error: Infusion event without input");
        //return Err("Error: Infusion event without input".into());
        exit(-1);
    }
    Ok(())
}
fn check_obs(event: &Event) -> Result<(), Box<dyn Error>> {
    if event.out.is_none() {
        tracing::error!("Error: Obs event without out");
        //return Err("Error: Obs event without out".into());
        exit(-1);
    }
    if event.outeq.is_none() {
        tracing::error!("Error: Obs event without outeq");
        //return Err("Error: Obs event without outeq".into());
        exit(-1);
    }
    Ok(())
}

fn decimals(value: f64, places: u32) -> f64 {
    let multiplier = 10f64.powi(places as i32);
    (value * multiplier).round() / multiplier
}
