use std::collections::HashMap;
use std::error::Error;

use interp::interp;

type Record = HashMap<String, String>;

enum TypeEvent {
    Infusion,
    Dose,
    Observation,
}

pub trait Event {
    fn get_type(&self) -> TypeEvent;
}

#[derive(Debug)]
pub struct Dose {
    pub time: f64,
    pub dose: f64,
    pub compartment: usize,
}
impl Event for Dose {
    fn get_type(&self) -> TypeEvent {
        TypeEvent::Dose
    }
}

#[derive(Debug)]
pub struct Infusion {
    pub time: f64,
    pub dur: f64,
    pub amount: f64,
    pub compartment: usize,
}

impl Event for Infusion {
    fn get_type(&self) -> TypeEvent {
        TypeEvent::Infusion
    }
}
#[derive(Debug)]
pub struct Observation {
    pub time: f64,
    pub obs: f64,
    pub outeq: usize,
}
impl Event for Observation {
    fn get_type(&self) -> TypeEvent {
        TypeEvent::Observation
    }
}

//This structure represents a single row in the CSV file
#[derive(Debug)]
struct RawEvent {
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
pub fn parse<E>(path: &String) -> Result<Vec<Box<dyn Event>>, Box<dyn Error>>
where
    E: Event,
{
    let mut rdr = csv::ReaderBuilder::new()
        // .delimiter(b',')
        // .escape(Some(b'\\'))
        .comment(Some(b'#'))
        .from_path(path)
        .unwrap();
    let mut raw_events: Vec<RawEvent> = vec![];

    for result in rdr.deserialize() {
        let mut record: Record = result?;
        raw_events.push(RawEvent {
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
    let mut events: Vec<Box<dyn Event>>;
    for event in raw_events {
        if event.evid == 1 {
            //dose event
            if event.dur.unwrap_or(0.0) > 0.0 {
                events.push(Box::new(Infusion {
                    time: event.time,
                    dur: event.dur.unwrap(),
                    amount: event.dose.unwrap(),
                    compartment: event.input.unwrap() - 1,
                }));
            } else {
                events.push(Box::new(Dose {
                    time: event.time,
                    dose: event.dose.unwrap(),
                    compartment: event.input.unwrap() - 1,
                }));
            }
        } else if event.evid == 0 {
            //obs event
            events.push(Box::new(Observation {
                time: event.time,
                obs: event.out.unwrap(),
                outeq: event.outeq.unwrap(),
            }));
        }
    }

    // let ev_iter = events.group_by_mut(|a, b| a.id == b.id);
    Ok(events)
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
