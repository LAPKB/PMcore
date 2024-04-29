use serde::Deserialize;
use std::{collections::HashMap, fmt};
pub mod parse_pmetrics;

pub trait DataTrait {
    fn get_subjects(&self) -> Vec<&Subject>;
    /// Returns the number of subjects in the dataset
    fn nsubjects(&self) -> usize;
    /// Returns the number of observations in the dataset
    fn nobs(&self) -> usize;
}

/// [Subject] is a trait that represents a single individual in a dataset
pub trait SubjectTrait {
    fn get_occasions(&self) -> Vec<&Occasion>;
    fn get_id(&self) -> &String;
}

/// Each [Subject] can have multiple occasions
pub trait OccasionTrait {
    fn get_events(
        &self,
        lagtime: Option<HashMap<usize, f64>>,
        bioavailability: Option<HashMap<usize, f64>>,
        ignore: bool,
    ) -> Vec<Event>;
    fn get_covariates(&self) -> Option<&Covariates>;
    fn get_infusions(&self) -> Vec<Infusion>;
}

pub trait CovariatesTrait {
    fn get_covariate(&self, name: &str) -> Option<&Covariate>;
    fn get_covariate_names(&self) -> Vec<String>;
}

pub trait CovariateTrait {
    fn interpolate(&self, time: f64) -> Option<f64>;
}

pub trait CovariateInterpolator {
    fn interpolate(&self, time: f64) -> Option<f64>;
    fn in_interval(&self, time: f64) -> bool;
}

// Redesign of data formats

/// An Event can be a Bolus, Infusion, or Observation
#[derive(Debug, Clone, Deserialize)]
pub enum Event {
    Bolus(Bolus),
    Infusion(Infusion),
    Observation(Observation),
}

impl Event {
    pub fn get_time(&self) -> f64 {
        match self {
            Event::Bolus(bolus) => bolus.time,
            Event::Infusion(infusion) => infusion.time,
            Event::Observation(observation) => observation.time,
        }
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Event::Bolus(bolus) => write!(
                f,
                "Bolus at time {:.2} with amount {:.2} in compartment {}",
                bolus.time, bolus.amount, bolus.compartment
            ),
            Event::Infusion(infusion) => write!(
                f,
                "Infusion starting at {:.2} with amount {:.2} over {:.2} hours in compartment {}",
                infusion.time, infusion.amount, infusion.duration, infusion.compartment
            ),
            Event::Observation(observation) => {
                let errpoly_desc = match observation.errorpoly {
                    Some((c0, c1, c2, c3)) => {
                        format!("with error poly =  ({}, {}, {}, {})", c0, c1, c2, c3)
                    }
                    None => "".to_string(),
                };
                write!(
                    f,
                    "Observation at time {:.2}: {} (outeq {}) {}",
                    observation.time, observation.value, observation.outeq, errpoly_desc
                )
            }
        }
    }
}

/// An instantaenous input of drug
#[derive(Debug, Clone, Deserialize)]
pub struct Bolus {
    pub time: f64,
    pub amount: f64,
    pub compartment: usize,
}

/// A continuous dose of drug
#[derive(Debug, Clone, Deserialize)]
pub struct Infusion {
    pub time: f64,
    pub amount: f64,
    pub compartment: usize,
    pub duration: f64,
}

/// An observation of drug concentration or covariates
#[derive(Debug, Clone, Deserialize)]
pub struct Observation {
    pub time: f64,
    pub value: f64,
    pub outeq: usize,
    pub errorpoly: Option<(f64, f64, f64, f64)>,
    pub ignore: bool,
}

/// An [Occasion] is a collection of events, for a given [Subject], that are from a specific occasion
#[derive(Debug, Deserialize, Clone)]
pub struct Occasion {
    pub events: Vec<Event>,
    pub covariates: Covariates,
    pub index: usize,
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

impl fmt::Display for Occasion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Occasion {}:", self.index)?;
        for event in &self.events {
            writeln!(f, "  {}", event)?;
        }

        writeln!(f, "  Covariates:\n{}", self.covariates)?;
        Ok(())
    }
}

impl OccasionTrait for Occasion {
    // TODO: This clones the occasion, which is not ideal
    fn get_events(
        &self,
        lagtime: Option<HashMap<usize, f64>>,
        bioavailability: Option<HashMap<usize, f64>>,
        ignore: bool,
    ) -> Vec<Event> {
        let mut occ = self.clone();
        occ.add_bioavailability(bioavailability);
        occ.add_lagtime(lagtime);

        // Filter out events that are marked as ignore
        if ignore {
            occ.events
                .iter()
                .filter(|event| match event {
                    Event::Observation(observation) => !observation.ignore,
                    _ => true,
                })
                .cloned()
                .collect()
        } else {
            occ.events.clone()
        } 
    }
    fn get_covariates(&self) -> Option<&Covariates> {
        Some(&self.covariates)
    }
    fn get_infusions(&self) -> Vec<Infusion> {
        self.events
            .iter()
            .filter_map(|event| {
                if let Event::Infusion(infusion) = event {
                    Some(infusion.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// [Subject] is a collection of blocks for one individual
#[derive(Debug, Deserialize, Clone)]
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

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Subject ID: {}", self.id)?;
        for occasion in &self.occasions {
            writeln!(f, "{}", occasion)?;
        }
        Ok(())
    }
}

impl SubjectTrait for Subject {
    fn get_occasions(&self) -> Vec<&Occasion> {
        self.occasions.iter().collect()
    }
    fn get_id(&self) -> &String {
        &self.id
    }
}

/// [Data] is a collection of [Subject]s, which are collections of [Occasion]s, which are collections of [Event]s
///
/// This is the main data structure used to store the data, and is used to pass data to the model
/// [Data] implements the [DataTrait], which provides methods to access the data
#[derive(Debug, Clone)]
pub struct Data {
    subjects: Vec<Subject>,
}

impl Data {
    pub fn new(subjects: Vec<Subject>) -> Self {
        Data { subjects }
    }
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Data Overview: {} subjects", self.subjects.len())?;
        for subject in &self.subjects {
            writeln!(f, "{}", subject)?;
        }
        Ok(())
    }
}

impl DataTrait for Data {
    fn get_subjects(&self) -> Vec<&Subject> {
        self.subjects.iter().collect()
    }

    /// Returns the number of subjects in the data
    fn nsubjects(&self) -> usize {
        self.subjects.len()
    }

    fn nobs(&self) -> usize {
        // Count the number of the event type Observation in the data
        self.subjects
            .iter()
            .map(|subject| {
                subject
                    .occasions
                    .iter()
                    .map(|occasion| {
                        occasion
                            .events
                            .iter()
                            .filter(|event| matches!(event, Event::Observation(_)))
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum InterpolationMethod {
    Linear { slope: f64, intercept: f64 },
    CarryForward { value: f64 },
}

/// A [CovariateSegment] is a segment of the piece-wise interpolation of a [Covariate]
#[derive(Clone, Debug, Deserialize)]
pub struct CovariateSegment {
    pub from: f64,
    pub to: f64,
    pub method: InterpolationMethod,
}

impl CovariateInterpolator for CovariateSegment {
    fn interpolate(&self, time: f64) -> Option<f64> {
        if self.in_interval(time) == false {
            return None;
        }

        match self.method {
            InterpolationMethod::Linear { slope, intercept } => Some(slope * time + intercept),
            InterpolationMethod::CarryForward { value } => Some(value),
        }
    }

    fn in_interval(&self, time: f64) -> bool {
        self.from <= time && time <= self.to
    }
}

/// A [Covariate] is a collection of [CovariateSegment]s, which allows for interpolation of covariate values
#[derive(Clone, Debug, Deserialize)]
pub struct Covariate {
    name: String,
    segments: Vec<CovariateSegment>,
}

impl Covariate {
    pub fn new(name: String, segments: Vec<CovariateSegment>) -> Self {
        Covariate { name, segments }
    }
    pub fn add_segment(&mut self, segment: CovariateSegment) {
        self.segments.push(segment);
    }
    // Check that no segments are overlapping
    pub fn check(&self) -> bool {
        let mut sorted = self.segments.clone();
        sorted.sort_by(|a, b| a.from.partial_cmp(&b.from).unwrap());
        for i in 0..sorted.len() - 1 {
            if sorted[i].to > sorted[i + 1].from {
                return false;
            }
        }
        true
    }
}

impl CovariateTrait for Covariate {
    fn interpolate(&self, time: f64) -> Option<f64> {
        self.segments
            .iter()
            .find(|&segment| segment.in_interval(time))
            .and_then(|segment| segment.interpolate(time))
    }
}

impl fmt::Display for Covariate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Covariate '{}':\n", self.name)?;
        for (index, segment) in self.segments.iter().enumerate() {
            write!(
                f,
                "  Segment {}: from {:.2} to {:.2}, ",
                index + 1,
                segment.from,
                segment.to
            )?;
            match &segment.method {
                InterpolationMethod::Linear { slope, intercept } => {
                    writeln!(
                        f,
                        "Linear, Slope: {:.2}, Intercept: {:.2}",
                        slope, intercept
                    )
                }
                InterpolationMethod::CarryForward { value } => {
                    writeln!(f, "Carry Forward, Value: {:.2}", value)
                }
            }?;
        }
        Ok(())
    }
}

/// [Covariates] is a collection of [Covariate]
#[derive(Clone, Debug, Deserialize)]
pub struct Covariates {
    // Mapping from covariate name to its segments
    covariates: HashMap<String, Covariate>,
}

impl Covariates {
    pub fn new() -> Self {
        Covariates {
            covariates: HashMap::new(),
        }
    }

    pub fn add_covariate(&mut self, name: String, covariate: Covariate) {
        self.covariates.insert(name, covariate);
    }

    fn get_covariate(&self, name: &str) -> Option<&Covariate> {
        self.covariates.get(name)
    }
    fn get_covariate_names(&self) -> Vec<String> {
        self.covariates.keys().cloned().collect()
    }
}

impl CovariatesTrait for Covariates {
    fn get_covariate(&self, name: &str) -> Option<&Covariate> {
        self.get_covariate(name)
    }
    fn get_covariate_names(&self) -> Vec<String> {
        self.get_covariate_names()
    }
}

impl fmt::Display for Covariates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Covariates:")?;
        for (_, covariate) in &self.covariates {
            writeln!(f, "{}", covariate)?;
        }
        Ok(())
    }
}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_covariate_linear_interpolation() {
        let segment = CovariateSegment {
            from: 0.0,
            to: 10.0,
            method: InterpolationMethod::Linear {
                slope: 1.0,
                intercept: 0.0,
            },
        };

        assert_eq!(segment.interpolate(0.0), Some(0.0));
        assert_eq!(segment.interpolate(5.0), Some(5.0));
        assert_eq!(segment.interpolate(10.0), Some(10.0));
        assert_eq!(segment.interpolate(15.0), None);
    }

    #[test]
    fn test_covariate_carry_forward() {
        let segment = CovariateSegment {
            from: 0.0,
            to: 10.0,
            method: InterpolationMethod::CarryForward { value: 5.0 },
        };

        assert_eq!(segment.interpolate(0.0), Some(5.0));
        assert_eq!(segment.interpolate(5.0), Some(5.0));
        assert_eq!(segment.interpolate(10.0), Some(5.0));
        assert_eq!(segment.interpolate(15.0), None);
    }

    #[test]
    fn test_covariates() {
        let mut covariates = Covariates {
            covariates: HashMap::new(),
        };
        covariates.covariates.insert(
            "covariate1".to_string(),
            Covariate::new(
                "covariate1".to_string(),
                vec![
                    CovariateSegment {
                        from: 0.0,
                        to: 10.0,
                        method: InterpolationMethod::Linear {
                            slope: 1.0,
                            intercept: 0.0,
                        },
                    },
                    CovariateSegment {
                        from: 10.0,
                        to: 20.0,
                        method: InterpolationMethod::CarryForward { value: 10.0 },
                    },
                ],
            ),
        );

        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(0.0),
            Some(0.0)
        );
        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(5.0),
            Some(5.0)
        );
        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(10.0),
            Some(10.0)
        );
        assert_eq!(
            covariates
                .get_covariate("covariate1")
                .unwrap()
                .interpolate(15.0),
            Some(10.0)
        );
    }
}
