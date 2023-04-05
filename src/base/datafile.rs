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
struct CovLine {
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
    covs: HashMap<String, CovLine>,
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
    let mut blocks: Vec<Block> = vec![];
    let mut block: Block = Block {
        events: vec![],
        covs: HashMap::new(),
    };
    let mut obs: Vec<f64> = vec![];
    let mut times: Vec<f64> = vec![];
    let mut obs_times: Vec<f64> = vec![];
    let mut id = events[0].id.clone();

    for mut event in events {
        //First event of a scenario
        if event.id != id {
            for (key, val) in &event.covs {
                if val.is_none() {
                    return Err(format!("Error: Covariate {} does not have a value on the first event of subject {}.", key, event.id).into());
                }
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
            block = Block {
                events: vec![],
                covs: HashMap::new(),
            };
            obs = vec![];
            blocks = vec![];
            obs_times = vec![];
            times = vec![];
            id = event.id.clone();
        }
        times.push(event.time);
        //Covariate forward filling
        for (key, val) in &mut event.covs {
            if val.is_none() {
                *val = *block.events.last().unwrap().covs.get(key).unwrap();
            }
        }
        //Event validation logic
        if event.evid == 1 {
            if event.dur.unwrap_or(0.0) > 0.0 {
                check_infusion(&event)?;
            } else {
                check_dose(&event)?;
            }

            if !block.events.is_empty() {
                blocks.push(block);
            }
            // clone the covs from the dose event and put them in the block
            block = Block {
                events: vec![],
                covs: HashMap::new(),
            };
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
    // Prepare the linear interpolation of the covariates
    let scenarios_c = scenarios.clone();
    for (si, scenario) in scenarios.iter_mut().enumerate() {
        let scenario_c = scenarios_c.get(si).unwrap();
        for (bi, block) in scenario.blocks.iter_mut().enumerate() {
            if let Some(next_block) = scenario_c.blocks.get(bi + 1) {
                for (key, reg) in &mut block.covs {
                    let p_v = block
                        .events
                        .first()
                        .unwrap()
                        .covs
                        .get(key)
                        .unwrap()
                        .unwrap();
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
                    *reg = CovLine { intercept, slope };
                }
            }
        }
    }

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

// #[derive(Debug)]
// pub struct Cov {
//     name: String,
//     times: Vec<f64>,
//     pub values: Vec<f64>,
// }

//Covariates
// type Covariates = Vec<Cov>;

// pub fn get_mut_cov(covs: &mut Covariates, key: String) -> Option<&mut Cov> {
//     for (i, cov) in covs.iter().enumerate() {
//         if cov.name == key {
//             return covs.get_mut(i);
//         }
//     }
//     None
// }

// pub fn get_cov(covs: &Covariates, key: String) -> Option<&Cov> {
//     for (i, cov) in covs.iter().enumerate() {
//         if cov.name == key {
//             return covs.get(i);
//         }
//     }
//     None
// }

// impl Cov {
//     pub fn interpolate(&self, t: f64) -> f64 {
//         interp(&self.times, &self.values, t)
//     }
// }
