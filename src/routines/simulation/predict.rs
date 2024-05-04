use crate::routines::data::Subject;
use crate::routines::datafile::CovLine;
use crate::routines::datafile::Infusion;
use crate::routines::datafile::Scenario;
use crate::routines::evaluation::sigma::ErrorPoly;
use crate::simulator::likelihood::PopulationPredictions;
use crate::simulator::likelihood::Prediction;
use crate::simulator::Equation;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use lazy_static::lazy_static;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::{Array, Array2, Axis};
use std::collections::HashMap;
use std::error;
use std::hash::{Hash, Hasher};

/// Number of support points to cache for each scenario
const CACHE_SIZE: usize = 1000;

#[derive(Debug, Clone)]
pub struct Model {
    params: HashMap<String, f64>,
    _scenario: Scenario,
    _infusions: Vec<Infusion>,
    _cov: Option<HashMap<String, CovLine>>,
}
impl Model {
    pub fn get_param(&self, str: &str) -> f64 {
        *self.params.get(str).unwrap()
    }
}

/// return the predicted values for the given scenario and parameters
/// where the second element of the tuple is the predicted values
/// one per observation time in scenario and in the same order
/// it is not relevant the outeq of the specific event.
pub trait Predict<'a> {
    type Model: 'a + Clone;
    type State;
    fn initial_system(&self, params: &Vec<f64>, scenario: Scenario) -> (Self::Model, Scenario);
    fn initial_state(&self) -> Self::State;
    fn add_covs(&self, system: &mut Self::Model, cov: Option<HashMap<String, CovLine>>);
    fn add_infusion(&self, system: &mut Self::Model, infusion: Infusion);
    fn add_dose(&self, state: &mut Self::State, dose: f64, compartment: usize);
    fn get_output(&self, time: f64, state: &Self::State, system: &Self::Model, outeq: usize)
        -> f64;
    fn state_step(&self, state: &mut Self::State, system: &Self::Model, time: f64, next_time: f64);
}

#[derive(Clone, Debug)]
pub struct Engine<S>
where
    S: Predict<'static> + Clone,
{
    ode: S,
}

impl<S> Engine<S>
where
    S: Predict<'static> + Clone,
{
    pub fn new(ode: S) -> Self {
        Self { ode }
    }
    pub fn pred(&self, scenario: Scenario, params: Vec<f64>) -> Vec<f64> {
        let (mut system, scenario) = self.ode.initial_system(&params, scenario.clone());
        let mut yout = vec![];
        let mut x = self.ode.initial_state();
        let mut index: usize = 0;
        for block in scenario.blocks {
            self.ode.add_covs(&mut system, Some(block.covs)); //TODO: this should not be Some
            for event in &block.events {
                if event.evid == 1 {
                    if event.dur.unwrap_or(0.0) > 0.0 {
                        //infusion
                        self.ode.add_infusion(
                            &mut system,
                            Infusion {
                                time: event.time,
                                dur: event.dur.unwrap(),
                                amount: event.dose.unwrap(),
                                compartment: event.input.unwrap() - 1,
                            },
                        );
                    } else {
                        //     //dose
                        self.ode
                            .add_dose(&mut x, event.dose.unwrap(), event.input.unwrap() - 1);
                    }
                } else if event.evid == 0 {
                    //obs
                    yout.push(
                        self.ode
                            .get_output(event.time, &x, &system, event.outeq.unwrap()),
                    )
                }
                if let Some(next_time) = scenario.times.get(index + 1) {
                    // TODO: use the last dx as the initial one for the next simulation.
                    self.ode.state_step(&mut x, &system, event.time, *next_time);
                }
                index += 1;
            }
        }
        yout
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CacheKey {
    i: usize,
    support_point: Vec<f64>,
}

impl Eq for CacheKey {}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.i.hash(state);
        for value in &self.support_point {
            value.to_bits().hash(state);
        }
    }
}

lazy_static! {
    static ref YPRED_CACHE: DashMap<CacheKey, Array1<f64>> =
        DashMap::with_capacity(CACHE_SIZE); // Adjust cache size as needed
}

fn get_ypred<S: Predict<'static> + Sync + Clone>(
    sim_eng: &Engine<S>,
    scenario: Scenario,
    support_point: Vec<f64>,
    i: usize,
    cache: bool,
) -> Array1<f64> {
    let key = CacheKey {
        i,
        support_point: support_point.clone(),
    };
    if cache {
        match YPRED_CACHE.entry(key) {
            Entry::Occupied(entry) => entry.get().clone(), // Clone the cached value
            Entry::Vacant(entry) => {
                let new_value = Array::from(sim_eng.pred(scenario, support_point));
                entry.insert(new_value.clone());
                new_value
            }
        }
    } else {
        Array::from(sim_eng.pred(scenario, support_point))
    }
}

/// Simulate observations for multiple scenarios and support points.
///
/// This function performs simulation of observations for multiple scenarios and support points
/// using the provided simulation engine `sim_eng`. It returns a 2D Array where each element
/// represents the simulated observations for a specific scenario and support point.
///
/// # Arguments
///
/// * `sim_eng` - A reference to the simulation engine implementing the `Predict` trait.
///
/// * `scenarios` - A reference to a `Vec<Scenario>` containing information about different scenarios.
///
/// * `support_points` - A 2D Array `(Array2<f64>)` representing the support points. Each row
///                     corresponds to a different support point scenario.
///
/// * `cache` - A boolean flag indicating whether to cache predicted values during simulation.
///
/// # Returns
///
/// A 2D Array `(Array2<Array1<f64>>)` where each element is an `Array1<f64>` representing the
/// simulated observations for a specific scenario and support point.
///
/// # Example
///
///
/// In this example, `observations` will contain the simulated observations for multiple scenarios
/// and support points.
///
/// Note: This function allows for optional caching of predicted values, which can improve
/// performance when simulating observations for multiple scenarios.
///
pub fn get_population_predictions(
    equation: &Equation,
    subjects: &Vec<Subject>,
    support_points: &Array2<f64>,
    cache: bool,
) -> PopulationPredictions {
    let mut pred = Array2::default((subjects.len(), support_points.nrows()).f());
    pred.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(j, mut element)| {
                    let subject = subjects.get(i).unwrap();
                    let ypred =
                        equation.simulate_subject(subject, support_points.row(j).to_vec().as_ref());
                    element.fill(ypred);
                });
        });
    pred.into()
}

fn simple_sim<S>(sim_eng: &Engine<S>, scenario: Scenario, support_point: &Array1<f64>) -> Vec<f64>
where
    S: Predict<'static> + Sync + Clone,
{
    sim_eng.pred(scenario, support_point.to_vec())
}

fn post_predictions<S>(
    sim_engine: &Engine<S>,
    post: Array2<f64>,
    scenarios: &Vec<Scenario>,
) -> Result<Array1<Vec<f64>>, Box<dyn error::Error>>
where
    S: Predict<'static> + Sync + Clone,
{
    if post.nrows() != scenarios.len() {
        return Err("Error calculating the posterior predictions, size mismatch.".into());
    }
    let mut predictions: Array1<Vec<f64>> = Array1::default(post.nrows());

    predictions
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut pred)| {
            let scenario = scenarios.get(i).unwrap();
            let support_point = post.row(i).to_owned();
            pred.fill(simple_sim(sim_engine, scenario.clone(), &support_point))
        });

    Ok(predictions)
}
