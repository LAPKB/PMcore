use serde::Deserialize;

// Redesign of data formats
use crate::routines::data::traits::*;
use std::collections::HashMap;

/// An Event can be a Bolus, Infusion, or Observation
#[derive(Debug, Clone, Deserialize)]
pub enum Event {
    Bolus(Bolus),
    Infusion(Infusion),
    Observation(Observation),
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
#[derive(Debug, Deserialize)]
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
    fn get_covariates(&self) -> Option<&impl CovariatesTrait> {
        Some(&self.covariates)
    }
}

/// [Subject] is a collection of blocks for one individual
#[derive(Debug, Deserialize)]
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

/// [Data] is a collection of [Subject]s, which are collections of [Occasion]s, which are collections of [Event]s
///
/// This is the main data structure used to store the data, and is used to pass data to the model
/// [Data] implements the [DataTrait], which provides methods to access the data
#[derive(Debug)]
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

/// Structures which implement the [CovariateInterpolator] trait
#[derive(Clone, Debug, Deserialize)]
pub enum CovariateSegment {
    LinearInterpolation(LinearInterpolation),
    CarryForward(CarryForward),
}

impl CovariateInterpolator for CovariateSegment {
    fn interpolate(&self, time: f64) -> Option<f64> {
        match self {
            CovariateSegment::LinearInterpolation(segment) => segment.interpolate(time),
            CovariateSegment::CarryForward(segment) => segment.interpolate(time),
        }
    }

    fn in_interval(&self, time: f64) -> bool {
        match self {
            CovariateSegment::LinearInterpolation(segment) => segment.in_interval(time),
            CovariateSegment::CarryForward(segment) => segment.in_interval(time),
        }
    }

    fn description(&self) -> String {
        match self {
            CovariateSegment::LinearInterpolation(segment) => segment.description(),
            CovariateSegment::CarryForward(segment) => segment.description(),
        }
    }
}

/// Linear interpolation between two points
#[derive(Clone, Debug, Deserialize)]
pub struct LinearInterpolation {
    pub from: f64,
    pub to: f64,
    pub slope: f64,
    pub intercept: f64,
}

impl LinearInterpolation {
    pub fn new(from: f64, to: f64, slope: f64, intercept: f64) -> Self {
        LinearInterpolation {
            from,
            to,
            slope,
            intercept,
        }
    }
}

impl CovariateInterpolator for LinearInterpolation {
    fn interpolate(&self, time: f64) -> Option<f64> {
        if time < self.from || time > self.to {
            return None;
        }
        Some(self.slope * time + self.intercept)
    }

    fn in_interval(&self, time: f64) -> bool {
        time >= self.from && time <= self.to
    }

    fn description(&self) -> String {
        format!(
            "Linear interpolation from {} to {} with slope {} and intercept {}",
            self.from, self.to, self.slope, self.intercept
        )
    }
}

/// Carry forward of a value between two points
#[derive(Clone, Debug, Deserialize)]
pub struct CarryForward {
    pub from: f64,
    pub to: f64,
    pub value: f64,
}

impl CarryForward {
    pub fn new(from: f64, to: f64, value: f64) -> Self {
        CarryForward { from, to, value }
    }
}

impl CovariateInterpolator for CarryForward {
    fn interpolate(&self, time: f64) -> Option<f64> {
        if time < self.from || time > self.to {
            return None;
        }
        Some(self.value)
    }

    fn in_interval(&self, time: f64) -> bool {
        time >= self.from && time <= self.to
    }

    fn description(&self) -> String {
        format!(
            "Carry forward from {} to {} with value {}",
            self.from, self.to, self.value
        )
    }
}

/// [Covariates] is a collection of vectors of [CovariateInterpolator]s
/// Covariates are used to model time-varying parameters in a model
#[derive(Clone, Debug, Deserialize)]
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
    pub fn add_segment(&mut self, name: String, segment: CovariateSegment) {
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

impl CovariatesTrait for Covariates {
    fn get_covariate(&self, name: &str) -> Option<&impl CovariateInterpolator> {
        self.segments.get(name).map(|segments| {
            segments
                .iter()
                .find(|segment| segment.in_interval(0.0))
                .unwrap_or_else(|| &segments[0])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariate_linear_interpolation() {
        let segment = CovariateSegment::LinearInterpolation(LinearInterpolation {
            from: 0.0,
            to: 10.0,
            slope: 1.0,
            intercept: 0.0,
        });

        assert_eq!(segment.interpolate(0.0), Some(0.0));
        assert_eq!(segment.interpolate(5.0), Some(5.0));
        assert_eq!(segment.interpolate(10.0), Some(10.0));
        assert_eq!(segment.interpolate(15.0), None);
    }

    #[test]
    fn test_covariate_carry_forward() {
        let segment = CovariateSegment::CarryForward(CarryForward {
            from: 0.0,
            to: 10.0,
            value: 5.0,
        });

        assert_eq!(segment.interpolate(0.0), Some(5.0));
        assert_eq!(segment.interpolate(5.0), Some(5.0));
        assert_eq!(segment.interpolate(10.0), Some(5.0));
        assert_eq!(segment.interpolate(15.0), None);
    }

    #[test]
    fn test_covariates() {
        let mut covariates = Covariates::new();
        covariates.add_segment(
            "covariate1".to_string(),
            CovariateSegment::LinearInterpolation(LinearInterpolation {
                from: 0.0,
                to: 10.0,
                slope: 1.0,
                intercept: 0.0,
            }),
        );
        covariates.add_segment(
            "covariate1".to_string(),
            CovariateSegment::CarryForward(CarryForward {
                from: 10.0,
                to: 20.0,
                value: 5.0,
            }),
        );

        assert_eq!(covariates.interpolate("covariate1", 0.0), Some(0.0));
        assert_eq!(covariates.interpolate("covariate1", 5.0), Some(5.0));
        assert_eq!(covariates.interpolate("covariate1", 10.0), Some(10.0));
        assert_eq!(covariates.interpolate("covariate1", 15.0), Some(5.0));
    }
}
