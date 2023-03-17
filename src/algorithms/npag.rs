use std::fs::File;

use crate::prelude::*;
use csv::WriterBuilder;
use linfa_linalg::qr::QR;
use ndarray::parallel::prelude::*;
use ndarray::{s, stack, Array, Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};
use ndarray_stats::DeviationExt;
use ndarray_stats::QuantileExt;
use tokio::sync::mpsc::UnboundedSender;

use crate::tui::state::AppState;

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
    tx: UnboundedSender<AppState>,
    settings: &Data,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, f64, usize, bool)
where
    S: Simulate + std::marker::Sync,
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

    let mut converged = false;

    let cycles_file = File::create("cycles.csv").unwrap();
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(cycles_file);

    // let mut _pred: Array2<Vec<f64>>;

    while eps > THETA_E {
        // log::info!("Cycle: {}", cycle);
        // psi n_sub rows, nspp columns
        psi = prob(sim_eng, scenarios, &theta, c);
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
            if lam > &1e-8 && lam > &(lambda.max().unwrap() / 1000_f64) {
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
        // permutate the columns of Psi
        let perm = n_psi.sort_axis_by(Axis(1), |i, j| {
            n_psi.column(i).sum() > n_psi.column(j).sum()
        });
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
                for i in 0..lim_loop {
                    let test = norm_zero(&r.column(i).to_owned());
                    if r.get((i, i)).unwrap() / test >= 1e-8 {
                        theta_rows.push(theta.row(perm.indices[keep]));
                        psi_columns.push(psi.column(perm.indices[keep]));
                        keep += 1;
                    }
                }
                theta = stack(Axis(0), &theta_rows).unwrap();
                psi = stack(Axis(1), &psi_columns).unwrap();
            }
            Err(_) => {
                log::info!("Cycle {}, #support points was {}", cycle, psi.ncols());
                let nsub = psi.nrows();
                // let perm = psi.sort_axis_by(Axis(1), |i, j| psi.column(i).sum() > psi.column(j).sum());
                psi = psi.permute_axis(Axis(1), &perm);
                theta = theta.permute_axis(Axis(0), &perm);
                psi = psi.slice(s![.., ..nsub]).to_owned();
                theta = theta.slice(s![..nsub, ..]).to_owned();
                log::info!("Pushed down to {}", psi.ncols());
            }
        }

        (lambda, objf) = match ipm::burke(&psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                //todo: write out report
                panic!("Error in IPM: {:?}", err);
            }
        };

        let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut lambda_tmp: Vec<f64> = vec![];
        for (index, lam) in lambda.iter().enumerate() {
            if lam > &(lambda.max().unwrap() / 1000_f64) {
                theta_rows.push(theta.row(index));
                psi_columns.push(psi.column(index));
                lambda_tmp.push(*lam);
            }
        }
        theta = stack(Axis(0), &theta_rows).unwrap();
        let psi = stack(Axis(1), &psi_columns).unwrap();
        w = Array::from(lambda_tmp);

        let pyl = psi.dot(&w);
        // log::info!("Spp: {}", theta.nrows());
        // log::info!("{:?}", &theta);
        // let mut thetaw = theta.clone();
        // thetaw.push_column(w.clone().t()).unwrap();
        // w.clone().to_vec();
        // log::info!("{:?}",&w);
        // log::info!("Objf: {}", -2.*objf);
        // if last_objf > objf{
        //     log::error!("Objf decreased");
        //     break;
        // }
        if let Some(output) = &settings.config.pmetrics_outputs {
            if *output {
                //cycles.csv
                //TODO: I need some sort of reader/writer, so I can keep building over the file
                writer.write_field(format!("{}", &cycle)).unwrap();
                writer.write_field(format!("{}", -2. * objf)).unwrap();
                writer.write_field(format!("{}", theta.nrows())).unwrap();

                for param in theta.axis_iter(Axis(1)).into_iter() {
                    writer
                        .write_field(format!("{}", param.mean().unwrap()))
                        .unwrap();
                }
                for param in theta.axis_iter(Axis(1)).into_iter() {
                    writer
                        .write_field(format!("{}", median(param.to_owned().to_vec())))
                        .unwrap();
                }
                for param in theta.axis_iter(Axis(1)).into_iter() {
                    writer
                        .write_field(format!("{}", param.std(1.)))
                        .unwrap();
                }

                writer.write_record(None::<&[u8]>).unwrap();
            }
        }
        let state = AppState {
            cycle,
            objf: -2. * objf,
            theta: theta.clone(),
        };
        tx.send(state).unwrap();

        if (last_objf - objf).abs() <= THETA_G && eps > THETA_E {
            eps /= 2.;
            if eps <= THETA_E {
                f1 = pyl.mapv(|x| x.ln()).sum();
                if (f1 - f0).abs() <= THETA_F {
                    log::info!("Likelihood criteria convergence");
                    converged = true;
                    break;
                } else {
                    f0 = f1;
                    eps = 0.2;
                }
            }
        }

        if cycle >= settings.config.cycles {
            log::info!("Maximum number of cycles reached");
            break;
        }
        theta = adaptative_grid(&mut theta, eps, &ranges);
        // dbg!(&theta);
        cycle += 1;
        last_objf = objf;
    }
    writer.flush().unwrap();
    (theta, psi, w, objf, cycle, converged)
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

fn median(data: Vec<f64>) -> f64 {
    let size = data.len();
    match size {
        even if even % 2 == 0 => {
            let fst = data.get((even / 2 - 1) as usize).unwrap();
            let snd = data.get((even / 2) as usize).unwrap();
            (fst + snd) / 2.0
        }
        odd => *data.get(odd / 2 as usize).unwrap(),
    }
}
