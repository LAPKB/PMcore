#![allow(dead_code)]
use std::cmp::Ordering;
use std::collections::HashMap;

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

impl Event {
    pub fn cmp_by_id_then_time(&self, other: &Self) -> Ordering {
        match self.id.cmp(&other.id) {
            Ordering::Equal => self.time.partial_cmp(&other.time).unwrap(),
            other => other,
        }
    }
}

fn main() {
    let mut events: Vec<Event> = vec![
        Event {
            id: "2".to_string(),
            evid: 1,
            time: 0.3,
            dur: None,
            dose: None,
            _addl: None,
            _ii: None,
            input: None,
            out: None,
            outeq: None,
            _c0: None,
            _c1: None,
            _c2: None,
            _c3: None,
            covs: HashMap::new(),
        },
        Event {
            id: "1".to_string(),
            evid: 1,
            time: 0.0,
            dur: None,
            dose: None,
            _addl: None,
            _ii: None,
            input: None,
            out: None,
            outeq: None,
            _c0: None,
            _c1: None,
            _c2: None,
            _c3: None,
            covs: HashMap::new(),
        },
        Event {
            id: "1".to_string(),
            evid: 1,
            time: 0.3,
            dur: None,
            dose: None,
            _addl: None,
            _ii: None,
            input: None,
            out: None,
            outeq: None,
            _c0: None,
            _c1: None,
            _c2: None,
            _c3: None,
            covs: HashMap::new(),
        },
        Event {
            id: "2".to_string(),
            evid: 1,
            time: 0.0,
            dur: None,
            dose: None,
            _addl: None,
            _ii: None,
            input: None,
            out: None,
            outeq: None,
            _c0: None,
            _c1: None,
            _c2: None,
            _c3: None,
            covs: HashMap::new(),
        },
        Event {
            id: "1".to_string(),
            evid: 1,
            time: 0.2,
            dur: None,
            dose: None,
            _addl: None,
            _ii: None,
            input: None,
            out: None,
            outeq: None,
            _c0: None,
            _c1: None,
            _c2: None,
            _c3: None,
            covs: HashMap::new(),
        },
    ];

    events.sort_by(|a, b| a.cmp_by_id_then_time(b));

    for event in &events {
        println!("id: {}, time: {}", event.id, event.time);
    }
}
