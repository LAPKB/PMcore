pub mod algebraic {
    pub mod one_compartment;
}
pub mod engine;

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use lazy_static::lazy_static;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Array1;
use std::error;

use ndarray::Array;
/// Number of support points to cache for each scenario
const CACHE_SIZE: usize = 1000;

use self::engine::Engine;
use self::engine::Predict;

use super::datafile::Scenario;

use std::hash::{Hash, Hasher};

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
pub fn sim_obs<M>(
    sim_eng: &Engine<M>,
    scenarios: &Vec<Scenario>,
    support_points: &Array2<f64>,
    cache: bool,
) -> Array2<Array1<f64>>
where
    M: Predict<'static> + Sync + Clone,
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
                    let ypred = cache_ypred(
                        sim_eng,
                        scenario.clone(),
                        support_points.row(j).to_vec(),
                        i,
                        cache,
                    );
                    element.fill(ypred);
                });
        });
    pred
}

// fn simple_sim<M>(sim_eng: &Engine<M>, scenario: Scenario, support_point: &Array1<f64>) -> Vec<f64>
// where
//     M: Predict<'static> + Sync + Clone,
// {
//     sim_eng.pred(scenario, support_point.to_vec())
// }

pub fn post_predictions<M>(
    sim_engine: &Engine<M>,
    post: Array2<f64>,
    scenarios: &Vec<Scenario>,
) -> Result<Array1<Vec<f64>>, Box<dyn error::Error>>
where
    M: Predict<'static> + Sync + Clone,
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
            pred.fill(sim_engine.pred(scenario.clone(), support_point.to_vec()))
        });

    Ok(predictions)
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

pub fn cache_ypred<M: Predict<'static> + Sync + Clone>(
    sim_eng: &Engine<M>,
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
