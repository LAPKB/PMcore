use crate::prelude::{Scenario, Simulate, Engine};
use ndarray::{prelude::*};
use ndarray::{Array, ArrayBase, OwnedRepr};
use ndarray::parallel::prelude::*;

const FRAC_1_SQRT_2PI: f64 = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

pub fn prob<S>(sim_eng: &Engine<S>, scenarios: &Vec<Scenario>, support_points: &ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>, c: (f64,f64,f64,f64)) -> ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>
where
    S: Simulate + std::marker::Sync
{
    let mut prob = Array2::<f64>::zeros((scenarios.len(), support_points.shape()[0]).f());
    prob.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut row)|{
        row.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(j, mut element)|{
            let scenario = scenarios.get(i).unwrap();
            let spp = support_points.row(j);
            let ypred = Array::from(sim_eng.pred(scenario, spp.to_vec()));
            let yobs = Array::from(scenario.obs_flat.clone());
            let sigma = c.0 + c.1* &yobs + c.2 * &yobs.mapv(|x| x.powi(2))+ c.3 * &yobs.mapv(|x| x.powi(3));
            let diff = (yobs-ypred).mapv(|x| x.powi(2));
            let two_sigma_sq = (2.0*&sigma).mapv(|x| x.powi(2));
            let aux_vec = FRAC_1_SQRT_2PI * &sigma * (-&diff/&two_sigma_sq).mapv(|x| x.exp());
            element.fill(aux_vec.product());
        });
    });
    // for (i, scenario) in scenarios.iter().enumerate(){
    //     for (j, spp) in support_points.axis_iter(Axis(0)).enumerate(){  
    //        let ypred = Array::from(sim_eng.pred(scenario, spp.to_vec()));
    //        let yobs = Array::from(scenario.obs_flat.clone());
    //        //TODO: esto se puede mover a datafile::read
    //        let sigma = c.0 + c.1* &yobs + c.2 * &yobs.mapv(|x| x.powi(2))+ c.3 * &yobs.mapv(|x| x.powi(3));
    //        let diff = (yobs-ypred).mapv(|x| x.powi(2));
    //        let two_sigma_sq = (2.0*&sigma).mapv(|x| x.powi(2));
    //        let aux_vec = FRAC_1_SQRT_2PI * &sigma * (-&diff/&two_sigma_sq).mapv(|x| x.exp());
    //        let value = aux_vec.product();
    //        prob.slice_mut(s![i,j]).fill(value);
    //     }
    // }
    prob
}

//TODO: I might need to implement that cache manually
//Example: https://github.com/jaemk/cached/issues/16
