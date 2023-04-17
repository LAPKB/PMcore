use crate::prelude::{Engine, Scenario, Simulate};
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use lazy_static::lazy_static;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray::OwnedRepr;
use std::hash::{Hash, Hasher};

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
    static ref YPRED_CACHE: DashMap<CacheKey, ArrayBase<OwnedRepr<f64>, Ix1>> =
        DashMap::with_capacity(1000000); // Adjust cache size as needed
}

fn get_ypred<S: Simulate + Sync>(
    sim_eng: &Engine<S>,
    scenario: &Scenario,
    support_point: Vec<f64>,
    i: usize,
    cache: bool,
) -> ArrayBase<OwnedRepr<f64>, Ix1> {
    let key = CacheKey {
        i,
        support_point: support_point.clone(),
    };
    if cache {
        match YPRED_CACHE.entry(key.clone()) {
            Entry::Occupied(entry) => entry.get().clone(), // Clone the cached value
            Entry::Vacant(entry) => {
                let new_value = Array::from(sim_eng.pred(scenario, support_point.clone()));
                entry.insert(new_value.clone());
                new_value
            }
        }
    } else {
        Array::from(sim_eng.pred(scenario, support_point.clone()))
    }
}

const FRAC_1_SQRT_2PI: f64 =
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

//TODO: I might need to implement that cache manually
//Example: https://github.com/jaemk/cached/issues/16
pub fn prob<S>(
    sim_eng: &Engine<S>,
    scenarios: &Vec<Scenario>,
    support_points: &Array2<f64>,
    c: (f64, f64, f64, f64),
    cache: bool,
) -> Array2<f64>
//(Array2<f64>,Array2<Vec<f64>>)
where
    S: Simulate + Sync,
{
    // let pred:Arc<Mutex<Array2<Vec<f64>>>> = Arc::new(Mutex::new(Array2::default((scenarios.len(), support_points.nrows()).f())));
    let mut prob = Array2::<f64>::zeros((scenarios.len(), support_points.nrows()).f());
    prob.axis_iter_mut(Axis(0))
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
                    let yobs = Array::from(scenario.obs_flat.clone());
                    let sigma = c.0
                        + c.1 * &yobs
                        + c.2 * &yobs.mapv(|x| x.powi(2))
                        + c.3 * &yobs.mapv(|x| x.powi(3));
                    let diff = (yobs - ypred).mapv(|x| x.powi(2));
                    let two_sigma_sq = (2.0 * &sigma).mapv(|x| x.powi(2));
                    let aux_vec =
                        FRAC_1_SQRT_2PI * (-&diff / &two_sigma_sq).mapv(|x| x.exp()) / &sigma;
                    // if i == 0 && j == 0 {
                    //     log::info!("PSI[1,1]={:?}",&aux_vec.product());
                    // }
                    // log::info!("Sigma[{}]={:?}", i, &sigma);
                    element.fill(aux_vec.product());
                });
        });
    // let pred= Arc::try_unwrap(pred).unwrap().into_inner().unwrap();
    // (prob,pred)
    prob
}

pub fn sim_obs<S>(
    sim_eng: &Engine<S>,
    scenarios: &Vec<Scenario>,
    support_points: &Array2<f64>,
) -> Array2<Vec<f64>>
where
    S: Simulate + Sync,
{
    let mut pred: Array2<Vec<f64>> = Array2::default((scenarios.len(), support_points.nrows()).f());
    pred.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(j, mut element)| {
                    let scenario = scenarios.get(i).unwrap();
                    let ypred = sim_eng.pred(scenario, support_points.row(j).to_vec());
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
