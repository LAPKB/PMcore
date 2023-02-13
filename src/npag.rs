use ndarray::{stack, Axis, ArrayBase, ViewRepr, Dim};
use ndarray_stats::QuantileExt;

use crate::prelude::*;

const THETA_E: f64 = 1e-4; //convergence Criteria

pub fn npag<S>(sim_eng: Engine<S>, ranges: Vec<(f64,f64)>, settings_path: String, seed: u32)
where
    S: Simulate
{
    let settings = settings::read(settings_path);
    let theta = lds::sobol(settings.config.init_points, ranges, seed);
    let scenarios = datafile::parse(settings.paths.data).unwrap();

    let mut eps  = 0.2;

    // while eps > THETA_E {
        // psi n_sub rows, nspp columns
        let mut psi = prob(&sim_eng, &scenarios, &theta);
        let (lambda, objf) = match ipm::burke(&mut psi){
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) =>{
                //todo: write out report
                panic!("Error in IPM: {:?}",err);
            }
        };
        dbg!(&theta);

        let mut rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let max_lam = lambda.max().unwrap();
        for (index,lam) in lambda.iter().enumerate(){
            if lam > &1e-8 && lam > &(max_lam/1000 as f64){
                let aux = theta.row(index);
                rows.push(aux);
            }
            
        }
        

        let theta = stack(Axis(0),&rows).unwrap();
        dbg!(theta);
        
        // dbg!(lambda);
        // dbg!(objf);
        
    // }
}