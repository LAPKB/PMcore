

use std::fs::File;

use csv::WriterBuilder;
use linfa_linalg::qr::QR;
use ndarray::{stack, Axis, ArrayBase, ViewRepr, Dim, Array, OwnedRepr, Array1, s, Array2};
use ndarray_csv::Array2Writer;
use ndarray_stats::QuantileExt;

use ndarray_stats::DeviationExt;


use tokio::sync::mpsc::UnboundedSender;
use ndarray::parallel::prelude::*;
use crate::prelude::*;

use crate::base::array_extra::*;

use crate::tui::state::AppState;

const THETA_E: f64 = 1e-4; //convergence Criteria
const THETA_G: f64 = 1e-4; //objf stop criteria
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-3;

pub fn npag<S>(
    sim_eng: Engine<S>,
    ranges: Vec<(f64,f64)>,
    mut theta: Array2<f64>,
    scenarios: Vec<Scenario>,
    c: (f64,f64,f64,f64),
    tx: UnboundedSender<AppState>,
    max_cycles: usize,
    posterior_path: Option<String>
)
where
    S: Simulate + std::marker::Sync
{
    
    let mut psi: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;
    let mut lambda: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>;

    let mut eps  = 0.2;
    let mut last_objf = -1e30;
    let mut objf: f64;
    let mut f0 = -1e30;
    let mut f1:f64;
    let mut cycle = 1;

    while eps > THETA_E {
        log::info!("Cycle: {}", cycle);
        
        // psi n_sub rows, nspp columns
        psi = prob(&sim_eng, &scenarios, &theta, c);
        // dbg!(&psi);
        (lambda,_) = match ipm::burke(&psi){
            Ok((lambda,objf)) => (lambda, objf),
            Err(err) =>{
                //todo: write out report
                panic!("Error in IPM: {:?}",err);
            }
        };
        let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        for (index,lam) in lambda.iter().enumerate(){
            if lam > &1e-8 && lam > &(lambda.max().unwrap()/1000_f64){
                theta_rows.push(theta.row(index));
                psi_columns.push(psi.column(index));
            }
        }
        theta = stack(Axis(0),&theta_rows).unwrap();
        psi = stack(Axis(1),&psi_columns).unwrap();

        
        
        
        // Normalize the rows of Psi
        let mut n_psi = psi.clone();
        n_psi.axis_iter_mut(Axis(0)).into_par_iter().for_each(
            |mut row| row /= row.sum()
        );
        // permutate the columns of Psi
        let perm = n_psi.sort_axis_by(Axis(1), |i, j| n_psi.column(i).sum() > n_psi.column(j).sum());
        n_psi = n_psi.permute_axis(Axis(1), &perm);
        // QR decomposition
        match n_psi.qr() {
            Ok(qr) => {
                let r = qr.into_r();
                // Keep the valuable spp
                let mut keep = 0;
                //The minimum between the number of subjects and the actual number of support points
                let lim_loop = n_psi.nrows().min(n_psi.ncols());

                let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
                let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
                for i in 0..lim_loop{
                    let test = norm_zero(&r.column(i).to_owned());
                    if r.get((i,i)).unwrap()/test >= 1e-8{
                        theta_rows.push(theta.row(perm.indices[keep]));
                        psi_columns.push(psi.column(perm.indices[keep]));
                        keep +=1;
                    }
                }
                theta = stack(Axis(0),&theta_rows).unwrap();
                psi = stack(Axis(1),&psi_columns).unwrap();
            },
            Err(_) => {
                log::error!("# support points was {}", psi.ncols());
                let nsub = psi.nrows();
                // let perm = psi.sort_axis_by(Axis(1), |i, j| psi.column(i).sum() > psi.column(j).sum());
                psi = psi.permute_axis(Axis(1), &perm);
                theta = theta.permute_axis(Axis(0), &perm);
                psi = psi.slice(s![..,..nsub]).to_owned();
                theta = theta.slice(s![..nsub,..]).to_owned();
                log::error!("Pushed down to {}", psi.ncols());

            }
        }
        

        
        (lambda,objf) = match ipm::burke(&psi){
            Ok((lambda,objf)) => (lambda, objf),
            Err(err) =>{
                //todo: write out report
                panic!("Error in IPM: {:?}",err);
            }
        };
        
        let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut lambda_tmp: Vec<f64> = vec![];
        for (index,lam) in lambda.iter().enumerate(){
            if lam > &(lambda.max().unwrap()/1000_f64){
                theta_rows.push(theta.row(index));
                psi_columns.push(psi.column(index));
                lambda_tmp.push(*lam);
            }
        }
        theta = stack(Axis(0),&theta_rows).unwrap();
        let psi2 = stack(Axis(1),&psi_columns).unwrap();
        let w = Array::from(lambda_tmp);

        let pyl = psi2.dot(&w);
        log::info!("Spp: {}", theta.shape()[0]);
        log::info!("{:?}",&theta);
        log::info!("{:?}",&w);
        log::info!("Objf: {}", -2.*objf);
        // if last_objf > objf{
        //     log::error!("Objf decreased");
        //     break;
        // }
        let state = AppState{
            cycle,
            objf: -2.*objf,
            theta: theta.clone()
        };
        tx.send(state).unwrap();

        if (last_objf-objf).abs() <= THETA_G && eps>THETA_E{
            eps /= 2.;
            if eps <= THETA_E{
                f1 = pyl.mapv(|x| x.ln()).sum();
                if (f1- f0).abs() <= THETA_F{
                    log::info!("Likelihood criteria convergence");
                    break;
                } else {
                    f0 = f1;
                    eps = 0.2;
                }
            }
        }

        if cycle >= max_cycles{
            break;
        }
        theta = adaptative_grid(&mut theta, eps, &ranges);
        // dbg!(&theta);
        cycle += 1; 
        last_objf = objf;
    }
    if let Some(theta_path) =  posterior_path {
        let file = File::create(theta_path).unwrap();
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        writer.serialize_array2(&theta).unwrap();
    } 
       
}

fn adaptative_grid(theta: &mut ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, eps: f64, ranges: &[(f64,f64)]) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let old_theta = theta.clone();
    for spp in old_theta.rows(){
        for (j, val) in spp.into_iter().enumerate(){
            let l = eps * (ranges[j].1 - ranges[j].0);//abs?
            if val + l < ranges[j].1{
                let mut plus = Array::zeros(spp.len());
                plus[j] = l;
                plus = plus + spp;
                evaluate_spp(theta, plus, ranges);
                // (n_spp, _) = theta.dim();

            }
            if val - l > ranges[j].0{
                let mut minus = Array::zeros(spp.len());
                minus[j] = -l;
                minus = minus + spp;
                evaluate_spp(theta, minus, ranges);
                // (n_spp, _) = theta.dim();
            }
        }
    }
    theta.to_owned()
}

fn evaluate_spp(theta: &mut ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, candidate: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>, limits: &[(f64,f64)]){
    for spp in theta.rows(){
        let mut dist: f64 = 0.;
        for (i, val) in candidate.clone().into_iter().enumerate(){
            dist += (val - spp.get(i).unwrap()).abs() / (limits[i].1 - limits[i].0);
        }
        if dist <= THETA_D {
            return;
        }
    }
    theta.push_row(candidate.view()).unwrap();

}



fn norm_zero(a: &Array1<f64>) -> f64{
    let zeros:Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}