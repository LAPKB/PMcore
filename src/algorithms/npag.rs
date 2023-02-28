use std::fs::File;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{stack, Axis, ArrayBase, ViewRepr, Dim, Array, OwnedRepr};
use ndarray_stats::QuantileExt;
use ndarray_csv::{Array2Reader, Array2Writer};

use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::encode::pattern::PatternEncoder;
use log4rs::config::{Appender, Config, Root};
use tokio::sync::mpsc::UnboundedSender;

use crate::prelude::*;
use crate::tui::state::AppState;

const THETA_E: f64 = 1e-4; //convergence Criteria
const THETA_G: f64 = 1e-4; //objf stop criteria
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-3;

pub fn npag<S>(sim_eng: Engine<S>, ranges: Vec<(f64,f64)>, settings_path: String, seed: u32, c: (f64,f64,f64,f64), tx: UnboundedSender<AppState>
)
where
    S: Simulate + std::marker::Sync
{
    let settings = settings::read(settings_path);
    setup_log(&settings);
    let mut theta = match &settings.paths.prior_dist {
        Some(prior_path) => {
            let file = File::open(prior_path).unwrap();
            let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
            let array_read: ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = reader.deserialize_array2_dynamic().unwrap();
            array_read
        },
        None => lds::sobol(settings.config.init_points, &ranges, seed)

    };

    // let mut theta = match theta0 {
    //     Some(theta) => theta,
    //     None => lds::sobol(settings.config.init_points, &ranges, seed)
    // };
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
        // let mut lambda_tmp: Vec<f64> = vec![];
        for (index,lam) in lambda.iter().enumerate(){
            if lam > &1e-8 && lam > &(lambda.max().unwrap()/1000_f64){
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
            if lam > &(lambda.max().unwrap()/1000_f64){
                theta_rows.push(theta.row(index));
                psi_columns.push(psi2.column(index));
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

        if cycle >= settings.config.cycles{
            break;
        }
        theta = adaptative_grid(&mut theta, eps, &ranges);
        // dbg!(&theta);
        cycle += 1; 
        last_objf = objf;
    }
    if let Some(theta_path) =  &settings.paths.posterior_dist {
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

fn setup_log(settings: &Data){
    if let Some(log_path) = &settings.paths.log_out {
        let logfile = FileAppender::builder()
            .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
            .build(log_path).unwrap();

        let config = Config::builder()
            .appender(Appender::builder().build("logfile", Box::new(logfile)))
            .build(Root::builder()
            .appender("logfile")
            .build(LevelFilter::Info)).unwrap();

        log4rs::init_config(config).unwrap();
    };
}