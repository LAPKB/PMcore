// use std::fmt::Display;

use crate::prelude::{Engine, Scenario, Simulate};
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Array;

// #[derive(Default, Clone)]
// pub struct Observations(pub Vec<f64>);
// impl Display for Observations {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         self.0.iter().fold(Ok(()), |result, value| {
//             result.and_then(|_| write!(f, "{},", value))
//         })
//     }
// }

const FRAC_1_SQRT_2PI: f64 =
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

//TODO: I might need to implement that cache manually
//Example: https://github.com/jaemk/cached/issues/16
pub fn prob<S>(
    sim_eng: &Engine<S>,
    scenarios: &Vec<Scenario>,
    support_points: &Array2<f64>,
    c: (f64, f64, f64, f64),
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
                    let ypred = Array::from(sim_eng.pred(scenario, support_points.row(j).to_vec()));
                    let yobs = Array::from(scenario.obs.clone());
                    // let mut lock = pred.lock().unwrap();
                    // let predij = lock.get_mut((i,j)).unwrap();
                    // predij.append(&mut scenario.obs_flat.clone());
                    // log::info!("Yobs[{}]={:?}", i, &yobs);
                    // log::info!("Ypred[{}]={:?}", i, &ypred);
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
