use std::collections::HashMap;
use std::error::Error;

use interp::interp;

type Record = HashMap<String, String>;

/// A Scenario is a collection of blocks that represent a single subject in the Datafile
#[derive(Debug)]
pub struct Scenario {
    pub id: String,
    pub blocks: Vec<Block>,
    pub obs: Vec<f64>,
    pub obs_times: Vec<f64>,
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
/// A Block is a simulation unit, this means that one simulation is made for each block
type Block = Vec<Event>;

/// A Event represent a single row in the Datafile
#[derive(Debug, Clone)]
pub struct Event {
    id: String,
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
    _covs: HashMap<String, Option<f64>>,
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
            _covs: record
                .into_iter()
                .map(|(key, value)| (key, value.parse::<f64>().ok()))
                .collect(),
        });
    }
    let mut scenarios: Vec<Scenario> = vec![];
    let mut blocks: Vec<Block> = vec![];
    let mut block: Block = vec![];
    let mut obs: Vec<f64> = vec![];
    let mut times: Vec<f64> = vec![];
    let mut obs_times: Vec<f64> = vec![];
    let mut id = events[0].id.clone();
    for event in events {
        //Check if the id changed
        if event.id != id {
            if !block.is_empty() {
                blocks.push(block);
            }
            scenarios.push(Scenario {
                id,
                blocks,
                obs,
                obs_times,
            });
            block = vec![event.clone()];
            obs = vec![];
            blocks = vec![];
            obs_times = vec![];
            id = event.id.clone();
        }
        times.push(event.time);
        //Event validation logic
        if event.evid == 1 {
            if event.dur.unwrap_or(0.0) > 0.0 {
                check_infusion(&event)?;
                block.push(event);
            } else {
                check_dose(&event)?;
                if !block.is_empty() {
                    blocks.push(block);
                }
                block = vec![event];
            }
        } else if event.evid == 0 {
            check_obs(&event)?;
            obs_times.push(event.time);
            obs.push(event.out.unwrap());
            block.push(event);
        } else {
            return Err("Error: Unsupported evid".into());
        }
    }
    if !block.is_empty() {
        blocks.push(block);
    }
    scenarios.push(Scenario {
        id,
        blocks,
        obs,
        obs_times,
    });

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

#[derive(Debug)]
pub struct Cov {
    name: String,
    times: Vec<f64>,
    pub values: Vec<f64>,
}

//Covariates
type Covariates = Vec<Cov>;

pub fn get_mut_cov(covs: &mut Covariates, key: String) -> Option<&mut Cov> {
    for (i, cov) in covs.iter().enumerate() {
        if cov.name == key {
            return covs.get_mut(i);
        }
    }
    None
}

pub fn get_cov(covs: &Covariates, key: String) -> Option<&Cov> {
    for (i, cov) in covs.iter().enumerate() {
        if cov.name == key {
            return covs.get(i);
        }
    }
    None
}

impl Cov {
    pub fn interpolate(&self, t: f64) -> f64 {
        interp(&self.times, &self.values, t)
    }
}
