use ndarray::{stack, Axis, ArrayBase, ViewRepr, Dim, Array, OwnedRepr};
use ndarray_stats::{QuantileExt, DeviationExt};

use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::encode::pattern::PatternEncoder;
use log4rs::config::{Appender, Config, Root};

use crate::prelude::*;

const THETA_E: f64 = 1e-4; //convergence Criteria
const THETA_G: f64 = 1e-4; //objf stop criteria
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

pub fn npag<S>(sim_eng: Engine<S>, ranges: Vec<(f64,f64)>, settings_path: String, seed: u32)
where
    S: Simulate
{
    let logfile = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
        .build("log/output.log").unwrap();

    let config = Config::builder()
        .appender(Appender::builder().build("logfile", Box::new(logfile)))
        .build(Root::builder()
                   .appender("logfile")
                   .build(LevelFilter::Info)).unwrap();

    log4rs::init_config(config).unwrap();




    let settings = settings::read(settings_path);
    let mut theta = lds::sobol(settings.config.init_points, &ranges, seed);
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
        log::info!("Cycle: {}", cycle);
        
        // psi n_sub rows, nspp columns
        psi = prob(&sim_eng, &scenarios, &theta);
        (lambda,_) = match ipm::burke(&psi){
            Ok((lambda,objf)) => (lambda, objf),
            Err(err) =>{
                //todo: write out report
                panic!("Error in IPM: {:?}",err);
            }
        };

        
        let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        // let mut lambda_tmp: Vec<f64> = vec![];
        for (index,lam) in lambda.iter().enumerate(){
            if lam > &1e-8 && lam > &(lambda.max().unwrap()/1000 as f64){
                theta_rows.push(theta.row(index));
                psi_columns.push(psi.column(index));
                // lambda_tmp.push(lam.clone());
            }
        }
        theta = stack(Axis(0),&theta_rows).unwrap();
        let psi2 = stack(Axis(1),&psi_columns).unwrap();
        // lambda = Array::from(lambda_tmp);
        
        // let psi2 = prob(&sim_eng, &scenarios, &theta);
        (lambda,objf) = match ipm::burke(&psi2){
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
            if lam > &(lambda.max().unwrap()/1000 as f64){
                theta_rows.push(theta.row(index));
                psi_columns.push(psi2.column(index));
                lambda_tmp.push(lam.clone());
            }
        }
        theta = stack(Axis(0),&theta_rows).unwrap();
        let psi2 = stack(Axis(1),&psi_columns).unwrap();
        let w = Array::from(lambda_tmp);
        let pyl = psi2.dot(&w);
        
        log::info!("Spp: {}", theta.shape()[0]);
        log::info!("{:?}",&theta);
        log::info!("{:?}",&w);
        // dbg!(&theta);
        log::info!("Objf: {}", &objf);
        // if last_objf > objf{
        //     log::error!("Objf decreased");
        //     break;
        // }
        

        if (last_objf-objf).abs() <= THETA_G && eps>THETA_E{
            eps = eps/2.;
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

        if cycle >= settings.config.cycles{
            break;
        }
        theta = adaptative_grid(theta, eps, &ranges);
        // dbg!(&theta);
        cycle = cycle+1; 
        last_objf = objf;
    }
}

fn adaptative_grid(theta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, eps: f64, ranges: &Vec<(f64,f64)>) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let (n_spp, _dim) = theta.dim();
    // dbg!(theta.dim());
    let mut new_theta = theta.clone();
    for i in 0..n_spp{
        let spp = theta.row(i);
        for (j, val) in spp.into_iter().enumerate(){
            let l = eps * (ranges[j].1 - ranges[j].0);//abs?
            if val + l > ranges[j].0{
                let mut plus = Array::zeros(spp.len());
                plus[j] = l;
                plus = plus + spp;
                evaluate_spp(&mut new_theta, plus, ranges[j]);
            }
            if val - l < ranges[j].1{
                let mut minus = Array::zeros(spp.len());
                minus[j] = -l;
                minus = minus + spp;
                evaluate_spp(&mut new_theta, minus, ranges[j]);
            }
        }
    }
    new_theta
}

fn evaluate_spp(theta: &mut ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, candidate: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>, limits: (f64,f64)){
    let mut dist = f64::INFINITY;

    for spp in theta.rows(){
        let new_dist = candidate.l1_dist(&spp).unwrap() / (limits.1 - limits.0);
        dist = dist.min(new_dist);
    }
    if dist > THETA_D{
        theta.push_row(candidate.view()).unwrap();
    }
}