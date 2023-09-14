use std::fs::File;
use std::process::exit;
// use std::process::exit;

use crate::prelude::linalg::faer_qr_decomp;
use crate::prelude::output::{CycleWriter, NPCycle, NPResult};
use crate::prelude::predict::sim_obs;
use crate::prelude::sigma::{ErrorPoly, ErrorType};
use crate::prelude::*;
use csv::WriterBuilder;
use faer_core::ComplexField;
use linfa_linalg::qr::QR;
use ndarray::parallel::prelude::*;
use ndarray::{s, stack, Array, Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};
use ndarray_csv::Array2Writer;
use ndarray_stats::DeviationExt;
use ndarray_stats::QuantileExt;
use tokio::sync::mpsc::UnboundedSender;

const THETA_E: f64 = 1e-4; //convergence Criteria
const THETA_G: f64 = 1e-4; //objf stop criteria
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

pub fn npag<S>(
    sim_eng: &Engine<S>,
    ranges: Vec<(f64, f64)>,
    mut theta: Array2<f64>,
    scenarios: &Vec<Scenario>,
    c: (f64, f64, f64, f64),
    tx: UnboundedSender<NPCycle>,
    settings: &Data,
) -> NPResult
where
    S: Predict + std::marker::Sync,
{
    let mut psi: Array2<f64> = Array2::default((0, 0));
    let mut lambda: Array1<f64>;
    let mut w: Array1<f64> = Array1::default(0);

    let mut eps = 0.2;
    let mut last_objf = -1e30;
    let mut objf: f64 = f64::INFINITY;
    let mut f0 = -1e30;
    let mut f1: f64;
    let mut cycle = 1;
    let mut gamma_delta = 0.1;
    let mut gamma = settings.parsed.error.value;

    let error_type = match settings.parsed.error.class.as_str() {
        "additive" => ErrorType::Add,
        "proportional" => ErrorType::Prop,
        _ => panic!("Error type not supported"),
    };

    let mut converged = false;

    // cycles.csv
    //TODO: Move out of NPAG
    let par_names = &settings.computed.random.names;
    let mut cycle_writer = CycleWriter::new("cycles.csv", par_names.to_vec());

    // meta_rust.csv
    //TODO: Move out of NPAG
    let meta_file = File::create("meta_rust.csv").unwrap();
    let mut meta_writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(meta_file);
    meta_writer.write_field("converged").unwrap();
    meta_writer.write_field("ncycles").unwrap();
    meta_writer.write_record(None::<&[u8]>).unwrap();

    // Instead we're using NPCycle
    let mut cycle_log: Vec<NPCycle> = Vec::new();

    // let mut _pred: Array2<Vec<f64>>;
    let cache = settings.parsed.config.cache.unwrap_or(false);

    while eps > THETA_E {
        // log::info!("Cycle: {}", cycle);
        // psi n_sub rows, nspp columns
        let cache = if cycle == 1 { false } else { cache };
        let ypred = sim_obs(sim_eng, scenarios, &theta, cache);

        psi = prob(
            &ypred,
            scenarios,
            &ErrorPoly {
                c,
                gl: gamma,
                e_type: &error_type,
            },
        );
        {
            let file = File::create("psi.csv").unwrap();
            let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
            writer.serialize_array2(&psi).unwrap();
        }
        // psi = prob(sim_eng, scenarios, &theta, c, cache);
        // for (i, row) in psi.axis_iter(Axis(0)).into_iter().enumerate() {
        //     log::info!("sub {}, sum: {}", i, row.sum());
        // }
        // (psi,_pred) = prob(&sim_eng, &scenarios, &theta, c);
        // dbg!(&psi);
        (lambda, _) = match ipm::burke(&psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                //todo: write out report
                panic!("Error in IPM: {:?}", err);
            }
        };
        // log::info!("lambda: {}", &lambda);
        let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        for (index, lam) in lambda.iter().enumerate() {
            if *lam > lambda.max().unwrap() / 1000_f64 {
                theta_rows.push(theta.row(index));
                psi_columns.push(psi.column(index));
            }
        }
        theta = stack(Axis(0), &theta_rows).unwrap();
        psi = stack(Axis(1), &psi_columns).unwrap();

        // Normalize the rows of Psi
        let mut n_psi = psi.clone();
        n_psi
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| row /= row.sum());
        // for row in n_psi.rows_mut() {
        //     let row_sum = row.sum();
        //     for elem in row {
        //         *elem /= row_sum
        //     }
        // }
        // // permutate the columns of n_Psi
        // let perm = n_psi.sort_axis_by(Axis(1), |i, j| {
        //     norm_zero(&n_psi.column(i).to_owned()) > norm_zero(&n_psi.column(j).to_owned())
        // });
        // // dbg!(perm);
        // // exit(1);

        // n_psi = n_psi.permute_axis(Axis(1), &perm);
        // // let r = n_psi.qr().unwrap().into_r();
        // // dbg!(r);

        // // for (i, col) in n_psi.columns().into_iter().enumerate() {
        // //     dbg!(col.get(i).unwrap() / norm_zero(&col.to_owned()));
        // // }
        // // exit(1);
        // // QR decomposition
        // match n_psi.qr() {
        //     Ok(qr) => {
        //         let r = qr.into_r();
        //         // Keep the valuable spp
        //         let mut keep = 0;
        //         //The minimum between the number of subjects and the actual number of support points
        //         let lim_loop = psi.nrows().min(psi.ncols());

        //         let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        //         let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        //         for i in 0..lim_loop {
        //             let test = norm_zero(&r.column(i).to_owned()); //the full column? or the triangular one?
        //             if r.get((i, i)).unwrap() / test >= 1e-8 {
        //                 theta_rows.push(theta.row(perm.indices[i]));
        //                 psi_columns.push(psi.column(perm.indices[i]));
        //                 keep += 1;
        //             }
        //         }
        //         theta = stack(Axis(0), &theta_rows).unwrap();
        //         psi = stack(Axis(1), &psi_columns).unwrap();
        //         log::info!(
        //             "QR decomp, cycle {}, keep: {}, thrown {}",
        //             cycle,
        //             keep,
        //             lim_loop - keep
        //         );
        //     }
        //     Err(_) => {
        //         log::info!("Cycle {}, #support points was {}", cycle, psi.ncols());
        //         let nsub = psi.nrows();
        //         // let perm = psi.sort_axis_by(Axis(1), |i, j| psi.column(i).sum() > psi.column(j).sum());
        //         psi = psi.permute_axis(Axis(1), &perm);
        //         theta = theta.permute_axis(Axis(0), &perm);
        //         psi = psi.slice(s![.., ..nsub]).to_owned();
        //         theta = theta.slice(s![..nsub, ..]).to_owned();
        //         log::info!("Pushed down to {}", psi.ncols());
        //     }
        // }
        {
            let file = File::create("n_psi.csv").unwrap();
            let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
            writer.serialize_array2(&n_psi).unwrap();
        }
        let (_r, perm) = faer_qr_decomp(&n_psi);
        n_psi = n_psi.permute_axis(
            Axis(1),
            &Permutation {
                indices: perm.clone(),
            },
        );
        let r = n_psi.qr().unwrap().into_r();
        {
            let file = File::create("r.csv").unwrap();
            let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
            writer.serialize_array2(&r).unwrap();
        }
        // for i in 0..20 {
        //     println!("i={}, r[i,i]={}", i, r.get((i, i)).unwrap());
        // }
        // exit(-1);
        // Keep the valuable spp
        let mut keep = 0;
        //The minimum between the number of subjects and the actual number of support points
        let lim_loop = psi.nrows().min(psi.ncols());
        let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        for i in 0..lim_loop {
            let test = norm_zero(&r.column(i).to_owned()); //the full column? or the triangular one?
                                                           // dbg!(r.get((i, i)).unwrap());
                                                           // dbg!(test);
                                                           // dbg!(r.get((i, i)).unwrap().abs() / test);
            let ratio = r.get((i, i)).unwrap() / test; // what happen if the diagonal value is negative?
            if ratio.abs() >= 1e-8 || ratio.is_nan() {
                theta_rows.push(theta.row(perm[i]));
                psi_columns.push(psi.column(perm[i]));
                keep += 1;
            } else {
                // dbg!(i);
                // dbg!(r.get((i, i)).unwrap());
                // dbg!(test);
                // dbg!(&r.column(i));
                // exit(-1);
            }
        }
        theta = stack(Axis(0), &theta_rows).unwrap();
        psi = stack(Axis(1), &psi_columns).unwrap();

        log::info!(
            "QR decomp, cycle {}, keep: {}, thrown {}",
            cycle,
            keep,
            lim_loop - keep
        );
        (lambda, objf) = match ipm::burke(&psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                //todo: write out report
                panic!("Error in IPM: {:?}", err);
            }
        };

        //Gam/Lam optimization
        let gamma_up = gamma * (1.0 + gamma_delta);
        let gamma_down = gamma / (1.0 + gamma_delta);
        let ypred = sim_obs(sim_eng, scenarios, &theta, cache);
        let psi_up = prob(
            &ypred,
            scenarios,
            &ErrorPoly {
                c,
                gl: gamma_up,
                e_type: &error_type,
            },
        );
        let psi_down = prob(
            &ypred,
            scenarios,
            &ErrorPoly {
                c,
                gl: gamma_down,
                e_type: &error_type,
            },
        );
        let (lambda_up, objf_up) = match ipm::burke(&psi_up) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                //todo: write out report
                panic!("Error in IPM: {:?}", err);
            }
        };
        let (lambda_down, objf_down) = match ipm::burke(&psi_down) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                //todo: write out report
                panic!("Error in IPM: {:?}", err);
            }
        };
        if objf_up > objf {
            gamma = gamma_up;
            objf = objf_up;
            gamma_delta *= 4.;
            lambda = lambda_up;
            psi = psi_up;
        }
        if objf_down > objf {
            gamma = gamma_down;
            objf = objf_down;
            gamma_delta *= 4.;
            lambda = lambda_down;
            psi = psi_down;
        }
        gamma_delta *= 0.5;
        if gamma_delta <= 0.01 {
            gamma_delta = 0.1;
        }

        // let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        // let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        // let mut lambda_tmp: Vec<f64> = vec![];
        // for (index, lam) in lambda.iter().enumerate() {
        //     if lam > &(lambda.max().unwrap() / 1000_f64) {
        //         theta_rows.push(theta.row(index));
        //         psi_columns.push(psi.column(index));
        //         lambda_tmp.push(*lam);
        //     }
        // }

        // theta = stack(Axis(0), &theta_rows).unwrap();
        // psi = stack(Axis(1), &psi_columns).unwrap();
        w = Array::from(lambda);
        let pyl = psi.dot(&w);

        //TODO: Move out of NPAG
        if let Some(output) = &settings.parsed.config.pmetrics_outputs {
            if *output {
                //cycles.csv
                cycle_writer.write(cycle, objf, gamma, &theta);
            }
        }

        // Write cycle output
        if let Some(true) = &settings.parsed.config.pmetrics_outputs {
            cycle_writer.write(cycle, objf, gamma, &theta);
        }

        let mut state = NPCycle {
            cycle,
            objf: -2. * objf,
            delta_objf: (last_objf - objf).abs(),
            nspp: theta.shape()[0],
            stop_text: "".to_string(),
            theta: theta.clone(),
            gamlam: gamma,
        };
        tx.send(state.clone()).unwrap();

        // If the objective function decreased, log an error.
        // Increasing objf signals instability of model misspecification.
        if last_objf > objf {
            log::error!("Objective function decreased");
        }

        // Stop if we have reached convergence criteria
        if (last_objf - objf).abs() <= THETA_G && eps > THETA_E {
            eps /= 2.;
            if eps <= THETA_E {
                f1 = pyl.mapv(|x| x.ln()).sum();
                if (f1 - f0).abs() <= THETA_F {
                    log::info!("Likelihood criteria convergence");
                    meta_writer.write_field("true").unwrap();
                    meta_writer.write_field(format!("{}", cycle)).unwrap();
                    meta_writer.write_record(None::<&[u8]>).unwrap();
                    converged = true;
                    state.stop_text = "The run converged!".to_string();
                    tx.send(state).unwrap();
                    break;
                } else {
                    f0 = f1;
                    eps = 0.2;
                }
            }
        }

        // Stop if we have reached maximum number of cycles
        if cycle >= settings.parsed.config.cycles {
            log::info!("Maximum number of cycles reached");
            meta_writer.write_field("false").unwrap();
            meta_writer.write_field(format!("{}", cycle)).unwrap();
            meta_writer.write_record(None::<&[u8]>).unwrap();
            state.stop_text = "No (max cycle)".to_string();
            tx.send(state).unwrap();
            break;
        }

        // Stop if stopfile exists
        let stopfile_found = std::path::Path::new("stop").exists();
        if stopfile_found {
            log::info!("Stopfile detected - breaking");
            meta_writer.write_field("false").unwrap();
            meta_writer.write_field(format!("{}", cycle)).unwrap();
            meta_writer.write_record(None::<&[u8]>).unwrap();
            state.stop_text = "No (stopped)".to_string();
            tx.send(state).unwrap();
            break;
        }
        // Append cycle info to cycle_log
        cycle_log.push(state);

        theta = adaptative_grid(&mut theta, eps, &ranges);
        // dbg!(&theta);
        cycle += 1;
        last_objf = objf;
    }
    cycle_writer.flush();
    NPResult {
        theta,
        psi,
        w,
        objf,
        cycles: cycle,
        converged,
        cycle_log,
    }
    // (theta, psi, w, objf, cycle, converged)
}

fn adaptative_grid(theta: &mut Array2<f64>, eps: f64, ranges: &[(f64, f64)]) -> Array2<f64> {
    let old_theta = theta.clone();
    for spp in old_theta.rows() {
        for (j, val) in spp.into_iter().enumerate() {
            let l = eps * (ranges[j].1 - ranges[j].0); //abs?
            if val + l < ranges[j].1 {
                let mut plus = Array::zeros(spp.len());
                plus[j] = l;
                plus = plus + spp;
                evaluate_spp(theta, plus, ranges);
                // (n_spp, _) = theta.dim();
            }
            if val - l > ranges[j].0 {
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

fn evaluate_spp(theta: &mut Array2<f64>, candidate: Array1<f64>, limits: &[(f64, f64)]) {
    for spp in theta.rows() {
        let mut dist: f64 = 0.;
        for (i, val) in candidate.clone().into_iter().enumerate() {
            dist += (val - spp.get(i).unwrap()).abs() / (limits[i].1 - limits[i].0);
        }
        if dist <= THETA_D {
            return;
        }
    }
    theta.push_row(candidate.view()).unwrap();
}

fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}

// what is pmetrics?
