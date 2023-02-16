use crate::prelude::{Scenario, Simulate, Engine};
use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, OwnedRepr};

///una matrix del tamaño #scenarios x #support points con la prob
///inputs
/// Scenarios (los scenarios contienen las observaciones)
/// Support points
//0.3989422804 
const FRAC_1_SQRT_2PI: f64 = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;



pub fn prob<S>(sim_eng: &Engine<S>, scenarios: &Vec<Scenario>, support_points: &ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>, c: (f64,f64,f64,f64)) -> ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>
where
    S: Simulate
{
    let mut prob = Array::<f64, _>::zeros((scenarios.len(), support_points.shape()[0]).f());
    for (i, scenario) in scenarios.iter().enumerate(){
        for (j, spp) in support_points.axis_iter(Axis(0)).enumerate(){
           
           let ypred = Array::from(sim_eng.pred(&scenario, spp.to_vec()));
           let yobs = Array::from(scenario.obs.clone());
           //TODO: esto se puede mover a datafile::read
           // 0.020000,0.050000,-0.000200,0.000000
           let sigma = c.0 + c.1* &yobs + c.2 * &yobs.mapv(|x| x.powi(2))+ c.3 * &yobs.mapv(|x| x.powi(3));
           let diff = (yobs-ypred).mapv(|x| x.powi(2));
           let two_sigma_sq = (2.0*&sigma).mapv(|x| x.powi(2));
           
           let aux_vec = FRAC_1_SQRT_2PI * &sigma * (-&diff/&two_sigma_sq).mapv(|x| x.exp());
           let value = aux_vec.product();

           prob.slice_mut(s![i,j]).fill(value);

        }
    }
    // for ((i,j), prob) in prob.indexed_iter_mut(){

    // }
    prob
}
