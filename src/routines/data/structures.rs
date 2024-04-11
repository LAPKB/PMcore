// Redesign of data formats
use crate::routines::data::traits::*;
use std::collections::HashMap;

/// An Event can be a Bolus, Infusion, or Observation
#[derive(Debug, Clone)]
pub enum Event {
    Bolus(Bolus),
    Infusion(Infusion),
    Observation(Observation),
}

/// An instantaenous input of drug
#[derive(Debug, Clone)]
pub struct Bolus {
    pub time: f64,
    pub amount: f64,
    pub compartment: usize,
}

/// A continuous dose of drug
#[derive(Debug, Clone)]
pub struct Infusion {
    pub time: f64,
    pub amount: f64,
    pub compartment: usize,
    pub duration: f64,
}

/// An observation of drug concentration or covariates
#[derive(Debug, Clone)]
pub struct Observation {
    pub time: f64,
    pub value: f64,
    pub outeq: usize,
    pub errorpoly: Option<(f64, f64, f64, f64)>,
    pub ignore: bool,
}

/// An [Occasion] is a collection of events that are related to each other
#[derive(Debug)]
pub struct Occasion {
    events: Vec<Event>,
    covariates: Covariates,
    index: usize,
}

impl Occasion {
    // Constructor
    pub fn new(events: Vec<Event>, covariates: Covariates, index: usize) -> Self {
        Occasion {
            events,
            covariates,
            index,
        }
    }

    pub fn add_lagtime(&mut self, lagtime: Option<HashMap<usize, f64>>) {
        if let Some(lag) = lagtime {
            for event in self.events.iter_mut() {
                if let Event::Bolus(bolus) = event {
                    if let Some(l) = lag.get(&bolus.compartment) {
                        bolus.time += l;
                    }
                }
            }
        }
        self.sort();
    }
    pub fn add_bioavailability(&mut self, bioavailability: Option<HashMap<usize, f64>>) {
        // If lagtime is empty, return early
        if let Some(fmap) = bioavailability {
            for event in self.events.iter_mut() {
                if let Event::Bolus(bolus) = event {
                    if let Some(f) = fmap.get(&bolus.compartment) {
                        bolus.time *= f;
                    }
                }
            }
        }
        self.sort();
    }

    /// Sort events by time, then by [Event] type so that [Bolus] and [Infusion] come before [Observation]
    pub fn sort(&mut self) {
        self.events.sort_by(|a, b| {
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

impl OccasionTrait for Occasion {
    fn get_events(
        &mut self,
        lagtime: Option<HashMap<usize, f64>>,
        bioavailability: Option<HashMap<usize, f64>>,
    ) -> Vec<&Event> {
        self.add_bioavailability(bioavailability);
        self.add_lagtime(lagtime);
        self.events.iter().collect()
    }
}

/// [Subject] is a collection of blocks for one individual
#[derive(Debug)]
pub struct Subject {
    pub id: String,
    pub occasions: Vec<Occasion>,
}

impl Subject {
    // Constructor
    pub fn new(id: String, occasions: Vec<Occasion>) -> Self {
        Subject { id, occasions }
    }
}

impl SubjectTrait for Subject {
    fn get_occasions(&self) -> Vec<&impl OccasionTrait> {
        self.occasions.iter().collect()
    }
    fn get_id(&self) -> &String {
        &self.id
    }
}

pub struct Data {
    subjects: Vec<Subject>,
}

impl Data {
    pub fn new(subjects: Vec<Subject>) -> Self {
        Data { subjects }
    }
}

impl DataTrait for Data {
    fn get_subjects(&self) -> Vec<&impl SubjectTrait> {
        self.subjects.iter().collect()
    }

    fn nsubjects(&self) -> usize {
        self.subjects.len()
    }
}

/// A [CovariateSegment] represents an interpolatable segment of a covariate
#[derive(Debug, Clone)]
pub enum CovariateSegment {
    /// Linear interpolation between two points
    LinearInterpolation {
        from: f64,
        to: f64,
        slope: f64,
        intercept: f64,
    },
    /// Carry forward of a value between two points
    CarryForward { from: f64, to: f64, value: f64 },
}

impl CovariateInterpolator for CovariateSegment {
    fn interpolate(&self, time: f64) -> Option<f64> {
        match self {
            CovariateSegment::LinearInterpolation {
                from,
                to,
                slope,
                intercept,
            } => {
                if time >= *from && time <= *to {
                    Some(slope * time + intercept)
                } else {
                    None
                }
            }
            CovariateSegment::CarryForward { from, to, value } => {
                if time >= *from && time <= *to {
                    Some(*value)
                } else {
                    None
                }
            }
        }
    }

    fn in_interval(&self, time: f64) -> bool {
        match self {
            CovariateSegment::LinearInterpolation { from, to, .. } => time >= *from && time <= *to,
            CovariateSegment::CarryForward { from, to, .. } => time >= *from && time <= *to,
        }
    }

    fn description(&self) -> String {
        match self {
            CovariateSegment::LinearInterpolation {
                from,
                to,
                slope,
                intercept,
            } => {
                format!(
                    "Linear interpolation from {:.2} to {:.2} with slope {:.2} and intercept {:.2}",
                    from, to, slope, intercept
                )
            }
            CovariateSegment::CarryForward { from, to, value } => {
                format!(
                    "Forward carry from {:.2} to {:.2} of value {:.2}",
                    from, to, value
                )
            }
        }
    }
}

/// [Covariates] are a collection of [CovariateSegment]s, each with a unique name
/// Covariates are used to model time-varying parameters in a model
#[derive(Clone, Debug)]
pub struct Covariates {
    // Mapping from covariate name to its segments
    segments: HashMap<String, Vec<CovariateSegment>>,
}

impl Covariates {
    pub fn new() -> Self {
        Covariates {
            segments: HashMap::new(),
        }
    }

    // Adds a segment to a specific covariate
    pub fn add_segment(&mut self, name: String, segment_type: CovariateSegment) {
        let segment: CovariateSegment = match segment_type {
            CovariateSegment::LinearInterpolation {
                from,
                to,
                slope,
                intercept,
            } => CovariateSegment::LinearInterpolation {
                from,
                to,
                slope,
                intercept,
            },
            CovariateSegment::CarryForward { from, to, value } => {
                CovariateSegment::CarryForward { from, to, value }
            }
        };
        self.segments
            .entry(name)
            .or_insert_with(Vec::new)
            .push(segment);
    }

    // Interpolates the value of a specific covariate at time x
    pub fn interpolate(&self, name: &str, x: f64) -> Option<f64> {
        self.segments
            .get(name)?
            .iter()
            .find(|segment| segment.in_interval(x))
            .map(|segment| segment.interpolate(x))
            .flatten()
    }
}
