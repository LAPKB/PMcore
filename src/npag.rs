use ndarray::{stack, Axis, ArrayBase, ViewRepr, Dim, Array};
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
    let theta = lds::sobol(settings.config.init_points, ranges, seed);
    let scenarios = datafile::parse(settings.paths.data).unwrap();

    let mut eps  = 0.2;
    let mut last_objf = -1e30;
    let mut f0 = -1e30;
    let mut f1 = -2e30;
    let mut cycle = 1;

    while eps > THETA_E {
        // psi n_sub rows, nspp columns
        let psi = prob(&sim_eng, &scenarios, &theta);
        let (lambda, objf) = match ipm::burke(&psi){
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) =>{
                //todo: write out report
                panic!("Error in IPM: {:?}",err);
            }
        };
        // dbg!(&theta.shape());
        // dbg!(&psi.shape());
        // dbg!(&lambda);
        // dbg!(&objf);

        let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut lambda_tmp: Vec<f64> = vec![];
        let max_lam = lambda.max().unwrap();
        for (index,lam) in lambda.iter().enumerate(){
            if lam > &1e-8 && lam > &(max_lam/1000 as f64){
                theta_rows.push(theta.row(index));
                psi_columns.push(psi.column(index));
                lambda_tmp.push(lam.clone());
            }
        }
        let theta = stack(Axis(0),&theta_rows).unwrap();
        let psi = stack(Axis(1),&psi_columns).unwrap();
        let lambda = Array::from(lambda_tmp);


        let objf = psi.dot(&lambda).mapv(|x| x.ln()).sum();


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


        cycle = cycle +1; 
        last_objf = objf;
    }
}