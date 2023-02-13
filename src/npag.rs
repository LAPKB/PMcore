use ndarray::{stack, Axis, ArrayBase, ViewRepr, Dim, Array, OwnedRepr};
use ndarray_stats::QuantileExt;

use crate::prelude::*;

const THETA_E: f64 = 1e-4; //convergence Criteria
const THETA_G: f64 = 1e-4; //objf stop criteria
const THETA_F: f64 = 1e-2;

pub fn npag<S>(sim_eng: Engine<S>, ranges: Vec<(f64,f64)>, settings_path: String, seed: u32)
where
    S: Simulate
{
    let settings = settings::read(settings_path);
    let mut theta = lds::sobol(settings.config.init_points, ranges, seed);
    let scenarios = datafile::parse(settings.paths.data).unwrap();
    let mut psi: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;
    let mut lambda: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>;

    let mut eps  = 0.2;
    let mut last_objf = -1e30;
    let mut objf: f64;
    let mut f0 = -1e30;
    let mut f1:f64;
    let mut cycle = 1;

    while eps > THETA_E {
        println!("Cycle: {}", cycle);
        println!("Spp: {}", theta.shape()[0]);
        // psi n_sub rows, nspp columns
        psi = prob(&sim_eng, &scenarios, &theta);
        lambda = match ipm::burke(&psi){
            Ok(lambda) => lambda,
            Err(err) =>{
                //todo: write out report
                panic!("Error in IPM: {:?}",err);
            }
        };
        {
            let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
            let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
            let mut lambda_tmp: Vec<f64> = vec![];
            for (index,lam) in lambda.iter().enumerate(){
                if lam > &1e-8 && lam > &(lambda.max().unwrap()/1000 as f64){
                    theta_rows.push(theta.row(index));
                    psi_columns.push(psi.column(index));
                    lambda_tmp.push(lam.clone());
                }
            }
            theta = stack(Axis(0),&theta_rows).unwrap();
            psi = stack(Axis(1),&psi_columns).unwrap();
            lambda = Array::from(lambda_tmp);
            objf = psi.dot(&lambda).mapv(|x| x.ln()).sum();
        }

        if (last_objf-objf).abs() <= THETA_G && eps>THETA_E{
            eps = eps/2.;
            if eps <= THETA_E{
                f1 = objf;
                if (f1- f0).abs() <= THETA_F{
                    break;
                } else {
                    f0 = f1;
                    eps = 0.2;
                }
            }
        }

        if cycle >= settings.config.cycles{
            break;
        }
        // theta = adaptative_grid(theta, eps)
        cycle = cycle+1; 
        last_objf = objf;
    }
}