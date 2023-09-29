use crate::routines::datafile::Scenario;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use lazy_static::lazy_static;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::{Array, Array2, Axis};
use std::error;
use std::hash::{Hash, Hasher};

const CACHE_SIZE: usize = 1000000;

/// return the predicted values for the given scenario and parameters
/// where the second element of the tuple is the predicted values
/// one per observation time in scenario and in the same order
/// it is not relevant the outeq of the specific event.
pub trait Predict {
    fn predict(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64>;
}

#[derive(Clone, Debug)]
pub struct Engine<S>
where
    S: Predict + Clone,
{
    ode: S,
}

impl<S> Engine<S>
where
    S: Predict + Clone,
{
    pub fn new(ode: S) -> Self {
        Self { ode }
    }
    pub fn pred(&self, scenario: &Scenario, params: Vec<f64>) -> Vec<f64> {
        self.ode.predict(params, scenario)
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

pub fn get_ypred<S: Predict + Sync + Clone>(
    sim_eng: &Engine<S>,
    scenario: &Scenario,
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

pub fn sim_obs<S>(
    sim_eng: &Engine<S>,
    scenarios: &Vec<Scenario>,
    support_points: &Array2<f64>,
    cache: bool,
) -> Array2<Array1<f64>>
where
    S: Predict + Sync + Clone,
{
    let mut pred: Array2<Array1<f64>> =
        Array2::default((scenarios.len(), support_points.nrows()).f());
    pred.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(j, mut element)| {
                    let scenario = scenarios.get(i).unwrap();
                    let ypred =
                        get_ypred(sim_eng, scenario, support_points.row(j).to_vec(), i, cache);
                    element.fill(ypred);
                });
        });
    pred
}

pub fn simple_sim<S>(
    sim_eng: &Engine<S>,
    scenario: &Scenario,
    support_point: &Array1<f64>,
) -> Vec<f64>
where
    S: Predict + Sync + Clone,
{
    sim_eng.pred(scenario, support_point.to_vec())
}

pub fn post_predictions<S>(
    sim_engine: &Engine<S>,
    post: Array2<f64>,
    scenarios: &Vec<Scenario>,
) -> Result<Array1<Vec<f64>>, Box<dyn error::Error>>
where
    S: Predict + Sync + Clone,
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
            pred.fill(simple_sim(sim_engine, scenario, &support_point))
        });

    Ok(predictions)
}
