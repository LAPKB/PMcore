#![allow(dead_code)]
use std::collections::HashMap;
use std::error::Error;

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

// impl Scenario {
//     pub fn new(events: Vec<Event>) -> Self {}
// }
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
    _addl: Option<isize>,
    _ii: Option<isize>,
    pub input: Option<usize>,
    out: Option<f64>,
    pub outeq: Option<usize>,
    _c0: Option<f32>,
    _c1: Option<f32>,
    _c2: Option<f32>,
    _c3: Option<f32>,
    pub covs: HashMap<String, Option<f64>>,
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

    let mut scenarios: Vec<Scenario> = vec![];

    let mut event_groups: HashMap<String, Vec<Event>> = HashMap::new();
    events.into_iter().for_each(|event| {
        event_groups
            .entry(event.id.clone())
            .or_insert_with(Vec::new)
            .push(event);
    });

    for (id, s_events) in event_groups {
        let mut blocks: Vec<Block> = vec![];
        let mut block: Block = Block {
            events: vec![],
            covs: HashMap::new(),
        };
        let mut obs: Vec<f64> = vec![];
        let mut times: Vec<f64> = vec![];
        let mut obs_times: Vec<f64> = vec![];

        for mut event in s_events {
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
                return Err("Error: Unsupported evid".into());
            }
            block.events.push(event);
        }
        if !block.events.is_empty() {
            blocks.push(block);
        }
        scenarios.push(Scenario {
            id,
            blocks,
            obs,
            obs_times,
            times,
        });
    }
    dbg!(&scenarios);
    // Prepare the linear interpolation of the covariates
    for scenario in &mut scenarios {
        let mut b_it = scenario.blocks.iter_mut().peekable();
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
    dbg!(scenarios);
    // hard stop execution
    std::process::exit(1);

    Ok(scenarios)
}

fn check_dose(event: &Event) -> Result<(), Box<dyn Error>> {
    if event.dose.is_none() {
        return Err("Error: Dose event without dose".into());
    }
    if event.input.is_none() {
        return Err("Error: Dose event without input".into());
    }
    Ok(())
}
fn check_infusion(event: &Event) -> Result<(), Box<dyn Error>> {
    if event.dose.is_none() {
        return Err("Error: Infusion event without dose".into());
    }
    if event.dur.is_none() {
        return Err("Error: Infusion event without duration".into());
    }
    if event.input.is_none() {
        return Err("Error: Infusion event without input".into());
    }
    Ok(())
}
fn check_obs(event: &Event) -> Result<(), Box<dyn Error>> {
    if event.out.is_none() {
        return Err("Error: Obs event without out".into());
    }
    if event.outeq.is_none() {
        return Err("Error: Obs event without outeq".into());
    }
    Ok(())
}
