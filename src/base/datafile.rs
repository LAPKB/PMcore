use std::collections::HashMap;
use std::error::Error;

use interp::interp;

type Record = HashMap<String, String>;

//This structure represents a single row in the CSV file
#[derive(Debug)]
struct Event {
    id: String,
    evid: isize,
    time: f64,
    dur: Option<f64>,
    dose: Option<f64>,
    _addl: Option<isize>,
    _ii: Option<isize>,
    input: Option<usize>,
    out: Option<f64>,
    outeq: Option<usize>,
    _c0: Option<f32>,
    _c1: Option<f32>,
    _c2: Option<f32>,
    _c3: Option<f32>,
    covs: HashMap<String, Option<f64>>,
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
                .map(|(key, value)| (key, value.parse::<f64>().ok()))
                .collect(),
        });
    }
    let mut scenarios: Vec<Scenario> = vec![];
    let ev_iter = events.group_by_mut(|a, b| a.id == b.id);
    for group in ev_iter {
        scenarios.push(parse_events_to_scenario(group));
    }
    Ok(scenarios)
}

#[derive(Debug)]
pub struct Dose {
    pub time: f64,
    pub dose: f64,
    pub compartment: usize,
}

#[derive(Debug)]
pub struct Infusion {
    pub time: f64,
    pub dur: f64,
    pub amount: f64,
    pub compartment: usize,
}
//This structure represents a full set of dosing events for a single ID
//TODO: I should transform the ADDL and II elements into the right dose events
#[derive(Debug)]
pub struct Scenario {
    pub id: String,     //id of the Scenario
    pub time: Vec<f64>, //ALL times
    pub infusions: Vec<Infusion>,
    pub doses: Vec<Dose>,
    pub time_obs: Vec<Vec<f64>>, //obs times
    pub obs: Vec<Vec<f64>>,      // obs @ time_obs
    pub time_flat: Vec<f64>,
    pub obs_flat: Vec<f64>,
    pub covariates: Covariates,
}
// Current Limitations:
// This version does not handle
// *  EVID!= 1 or 2
// *  ADDL & II
// *  C0, C1, C2, C3
//TODO: time needs to be expanded with the times relevant to ADDL and II
//TODO: Also dose must be expanded because of the same reason
// cov: , //this should be a matrix (or function ), with values for each cov and time
fn parse_events_to_scenario(events: &[Event]) -> Scenario {
    let mut time: Vec<f64> = vec![];
    let mut doses: Vec<Dose> = vec![];
    let mut infusions: Vec<Infusion> = vec![];
    let mut raw_time_obs: Vec<f64> = vec![];
    let mut raw_obs: Vec<f64> = vec![];
    let mut raw_outeq: Vec<usize> = vec![];
    let mut covariates: Covariates = Default::default();
    for key in events.get(0).unwrap().covs.keys() {
        covariates.push(Cov {
            name: key.clone(),
            times: vec![],
            values: vec![],
        });
    }
    for event in events {
        time.push(event.time);

        if event.evid == 1 {
            //dose event
            if event.dur.unwrap_or(0.0) > 0.0 {
                infusions.push(Infusion {
                    time: event.time,
                    dur: event.dur.unwrap(),
                    amount: event.dose.unwrap(),
                    compartment: event.input.unwrap() - 1,
                });
            } else {
                doses.push(Dose {
                    time: event.time,
                    dose: event.dose.unwrap(),
                    compartment: event.input.unwrap() - 1,
                });
            }
        } else if event.evid == 0 {
            //obs event
            raw_obs.push(event.out.unwrap());
            raw_time_obs.push(event.time);
            raw_outeq.push(event.outeq.unwrap());
        }
        for (key, op_val) in &event.covs {
            if let Some(val) = op_val {
                let cov = get_mut_cov(&mut covariates, key.to_string()).unwrap();
                cov.times.push(event.time);
                cov.values.push(*val);
            }
        }
    }

    let max_outeq = raw_outeq.iter().max().unwrap();
    let mut time_obs: Vec<Vec<f64>> = vec![];
    let mut obs: Vec<Vec<f64>> = vec![];
    for _ in 0..*max_outeq {
        time_obs.push(vec![]);
        obs.push(vec![]);
    }
    for ((t, o), eq) in raw_time_obs
        .iter()
        .zip(raw_obs.iter())
        .zip(raw_outeq.iter())
    {
        time_obs.get_mut(eq - 1).unwrap().push(*t);
        obs.get_mut(eq - 1).unwrap().push(*o);
    }
    let time_flat = time_obs.clone().into_iter().flatten().collect::<Vec<f64>>();
    let obs_flat = obs.clone().into_iter().flatten().collect::<Vec<f64>>();
    Scenario {
        id: events[0].id.clone(),
        time,
        doses,
        infusions,
        time_obs,
        obs,
        time_flat,
        obs_flat,
        covariates,
    }
}

#[derive(Debug)]
pub struct Cov {
    name: String,
    times: Vec<f64>,
    values: Vec<f64>,
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
