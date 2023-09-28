use crate::prelude::*;
use datafile::Scenario;
use ndarray::{stack, Array, Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};
use output::{CycleLog, NPCycle, NPResult};
use predict::{sim_obs, Engine, Predict};
use settings::run::Data;
use sigma::{ErrorPoly, ErrorType};
// use ndarray_csv::Array2Writer;
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
    let mut cycle_log = CycleLog::new(&settings.computed.random.names);

    // let mut _pred: Array2<Vec<f64>>;
    let cache = settings.parsed.config.cache.unwrap_or(false);

    while eps > THETA_E {
        // log::info!("Cycle: {}", cycle);
        // psi n_sub rows, nspp columns
        let cache = if cycle == 1 { false } else { cache };
        let ypred = sim_obs(sim_eng, scenarios, &theta, cache);

        psi = prob::calculate_psi(
            &ypred,
            scenarios,
            &ErrorPoly {
                c,
                gl: gamma,
                e_type: &error_type,
            },
        );
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

        //Rank-Revealing Factorization
        let (r, perm) = qr::calculate_r(&psi);
        let nspp = psi.ncols();
        let mut keep = 0;
        //The minimum between the number of subjects and the actual number of support points
        let lim_loop = psi.nrows().min(psi.ncols());
        let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
        for i in 0..lim_loop {
            let test = norm_zero(&r.column(i).to_owned());
            let ratio = r.get((i, i)).unwrap() / test;
            if ratio.abs() >= 1e-8 {
                theta_rows.push(theta.row(*perm.get(i).unwrap()));
                psi_columns.push(psi.column(*perm.get(i).unwrap()));
                keep += 1;
            }
        }
        theta = stack(Axis(0), &theta_rows).unwrap();
        psi = stack(Axis(1), &psi_columns).unwrap();

        log::info!(
            "QR decomp, cycle {}, kept: {}, thrown {}",
            cycle,
            keep,
            nspp - keep
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
        let psi_up = prob::calculate_psi(
            &ypred,
            scenarios,
            &ErrorPoly {
                c,
                gl: gamma_up,
                e_type: &error_type,
            },
        );
        let psi_down = prob::calculate_psi(
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

        w = Array::from(lambda);
        let pyl = psi.dot(&w);

        // Stop if we have reached convergence criteria
        if (last_objf - objf).abs() <= THETA_G && eps > THETA_E {
            eps /= 2.;
            if eps <= THETA_E {
                f1 = pyl.mapv(|x| x.ln()).sum();
                if (f1 - f0).abs() <= THETA_F {
                    log::info!("Likelihood criteria convergence");
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
            state.stop_text = "No (max cycle)".to_string();
            tx.send(state).unwrap();
            break;
        }

        // Stop if stopfile exists
        if std::path::Path::new("stop").exists() {
            log::info!("Stopfile detected - breaking");
            state.stop_text = "No (stopped)".to_string();
            tx.send(state).unwrap();
            break;
        }
        cycle_log.push_and_write(state, settings.parsed.config.pmetrics_outputs.unwrap());

        theta = expansion::adaptative_grid(&mut theta, eps, &ranges, THETA_D);
        cycle += 1;
        last_objf = objf;
    }

    NPResult::new(
        scenarios.clone(),
        theta,
        psi,
        w,
        objf,
        cycle,
        converged,
        cycle_log,
        settings,
    )
}

fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}
