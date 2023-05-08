use crate::base::datafile::Scenario;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use lazy_static::lazy_static;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::{Array, Array2, Axis};
use std::hash::{Hash, Hasher};

///
/// return the predicted values for the given scenario and parameters
/// where the second element of the tuple is the predicted values
/// one per observation time in scenario and in the same order
/// it is not relevant the outeq of the specific event.
pub trait Simulate {
    fn simulate(&self, params: Vec<f64>, scenario: &Scenario) -> Vec<f64>;
}

pub struct Engine<S>
where
    S: Simulate,
{
    sim: S,
}

impl<S> Engine<S>
where
    S: Simulate,
{
    pub fn new(sim: S) -> Self {
        Self { sim }
    }
    pub fn pred(&self, scenario: &Scenario, params: Vec<f64>) -> Vec<f64> {
        self.sim.simulate(params, scenario)
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
        DashMap::with_capacity(1000000); // Adjust cache size as needed
}

pub fn get_ypred<S: Simulate + Sync>(
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
    S: Simulate + Sync,
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
    S: Simulate + Sync,
{
    sim_eng.pred(scenario, support_point.to_vec())
}
