use crate::prelude::{
    algorithms::Algorithm,
    datafile::Scenario,
    evaluation::sigma::{ErrorPoly, ErrorType},
    expansion, ipm,
    output::NPResult,
    output::{CycleLog, NPCycle},
    prob, qr,
    settings::run::Data,
    simulation::predict::Engine,
    simulation::predict::{sim_obs, Predict},
};

use ndarray::{stack, Array, Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};
use ndarray_stats::{DeviationExt, QuantileExt};
use tokio::sync::mpsc::UnboundedSender;

const THETA_E: f64 = 1e-4; //convergence Criteria
const THETA_G: f64 = 1e-4; //objf stop criteria
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;
pub struct NPAG<S>
where
    S: Predict + std::marker::Sync + Clone,
{
    engine: Engine<S>,
    ranges: Vec<(f64, f64)>,
    psi: Array2<f64>,
    theta: Array2<f64>,
    lambda: Array1<f64>,
    w: Array1<f64>,
    eps: f64,
    last_objf: f64,
    objf: f64,
    f0: f64,
    f1: f64,
    cycle: usize,
    gamma_delta: f64,
    gamma: f64,
    error_type: ErrorType,
    converged: bool,
    cycle_log: CycleLog,
    cache: bool,
    scenarios: Vec<Scenario>,
    c: (f64, f64, f64, f64),
    tx: UnboundedSender<NPCycle>,
    settings: Data,
}

impl<S> Algorithm<S> for NPAG<S>
where
    S: Predict + std::marker::Sync + Clone,
{
    // fn initialize(
    //     sim_eng: Engine<S>,
    //     ranges: Vec<(f64, f64)>,
    //     theta: Array2<f64>,
    //     scenarios: Vec<Scenario>,
    //     c: (f64, f64, f64, f64),
    //     tx: UnboundedSender<NPCycle>,
    //     settings: Data,
    // ) -> Self {
    //     NPAG::new(sim_eng, ranges, theta, scenarios, c, tx, settings)
    // }
    fn fit(&mut self) -> NPResult {
        self.run()
    }
}

impl<S> NPAG<S>
where
    S: Predict + std::marker::Sync + Clone,
{
    pub fn new(
        sim_eng: Engine<S>,
        ranges: Vec<(f64, f64)>,
        theta: Array2<f64>,
        scenarios: Vec<Scenario>,
        c: (f64, f64, f64, f64),
        tx: UnboundedSender<NPCycle>,
        settings: Data,
    ) -> Self
    where
        S: Predict + std::marker::Sync,
    {
        Self {
            engine: sim_eng,
            ranges,
            psi: Array2::default((0, 0)),
            theta,
            lambda: Array1::default(0),
            w: Array1::default(0),
            eps: 0.2,
            last_objf: -1e30,
            objf: f64::INFINITY,
            f0: -1e30,
            f1: f64::default(),
            cycle: 1,
            gamma_delta: 0.1,
            gamma: settings.parsed.error.value,
            error_type: match settings.parsed.error.class.as_str() {
                "additive" => ErrorType::Add,
                "proportional" => ErrorType::Prop,
                _ => panic!("Error type not supported"),
            },
            converged: false,
            cycle_log: CycleLog::new(&settings.computed.random.names),
            cache: settings.parsed.config.cache.unwrap_or(false),
            tx,
            settings,
            scenarios,
            c,
        }
    }

    pub fn run(&mut self) -> NPResult {
        while self.eps > THETA_E {
            // log::info!("Cycle: {}", cycle);
            // psi n_sub rows, nspp columns
            let cache = if self.cycle == 1 { false } else { self.cache };
            let ypred = sim_obs(&self.engine, &self.scenarios, &self.theta, cache);

            self.psi = prob::calculate_psi(
                &ypred,
                &self.scenarios,
                &ErrorPoly {
                    c: self.c,
                    gl: self.gamma,
                    e_type: &self.error_type,
                },
            );
            (self.lambda, _) = match ipm::burke(&self.psi) {
                Ok((lambda, objf)) => (lambda, objf),
                Err(err) => {
                    //todo: write out report
                    panic!("Error in IPM: {:?}", err);
                }
            };
            // log::info!("lambda: {}", &lambda);
            let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
            let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
            for (index, lam) in self.lambda.iter().enumerate() {
                if *lam > self.lambda.max().unwrap() / 1000_f64 {
                    theta_rows.push(self.theta.row(index));
                    psi_columns.push(self.psi.column(index));
                }
            }
            self.theta = stack(Axis(0), &theta_rows).unwrap();
            self.psi = stack(Axis(1), &psi_columns).unwrap();

            //Rank-Revealing Factorization
            let (r, perm) = qr::calculate_r(&self.psi);
            let nspp = self.psi.ncols();
            let mut keep = 0;
            //The minimum between the number of subjects and the actual number of support points
            let lim_loop = self.psi.nrows().min(self.psi.ncols());
            let mut theta_rows: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
            let mut psi_columns: Vec<ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>> = vec![];
            for i in 0..lim_loop {
                let test = norm_zero(&r.column(i).to_owned());
                let ratio = r.get((i, i)).unwrap() / test;
                if ratio.abs() >= 1e-8 {
                    theta_rows.push(self.theta.row(*perm.get(i).unwrap()));
                    psi_columns.push(self.psi.column(*perm.get(i).unwrap()));
                    keep += 1;
                }
            }
            self.theta = stack(Axis(0), &theta_rows).unwrap();
            self.psi = stack(Axis(1), &psi_columns).unwrap();

            log::info!(
                "QR decomp, cycle {}, kept: {}, thrown {}",
                self.cycle,
                keep,
                nspp - keep
            );
            (self.lambda, self.objf) = match ipm::burke(&self.psi) {
                Ok((lambda, objf)) => (lambda, objf),
                Err(err) => {
                    //todo: write out report
                    panic!("Error in IPM: {:?}", err);
                }
            };

            //Gam/Lam optimization
            let gamma_up = self.gamma * (1.0 + self.gamma_delta);
            let gamma_down = self.gamma / (1.0 + self.gamma_delta);
            let ypred = sim_obs(&self.engine, &self.scenarios, &self.theta, cache);
            let psi_up = prob::calculate_psi(
                &ypred,
                &self.scenarios,
                &ErrorPoly {
                    c: self.c,
                    gl: gamma_up,
                    e_type: &self.error_type,
                },
            );
            let psi_down = prob::calculate_psi(
                &ypred,
                &self.scenarios,
                &ErrorPoly {
                    c: self.c,
                    gl: gamma_down,
                    e_type: &self.error_type,
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
            if objf_up > self.objf {
                self.gamma = gamma_up;
                self.objf = objf_up;
                self.gamma_delta *= 4.;
                self.lambda = lambda_up;
                self.psi = psi_up;
            }
            if objf_down > self.objf {
                self.gamma = gamma_down;
                self.objf = objf_down;
                self.gamma_delta *= 4.;
                self.lambda = lambda_down;
                self.psi = psi_down;
            }
            self.gamma_delta *= 0.5;
            if self.gamma_delta <= 0.01 {
                self.gamma_delta = 0.1;
            }

            let mut state = NPCycle {
                cycle: self.cycle,
                objf: -2. * self.objf,
                delta_objf: (self.last_objf - self.objf).abs(),
                nspp: self.theta.shape()[0],
                stop_text: "".to_string(),
                theta: self.theta.clone(),
                gamlam: self.gamma,
            };
            self.tx.send(state.clone()).unwrap();

            // If the objective function decreased, log an error.
            // Increasing objf signals instability of model misspecification.
            if self.last_objf > self.objf {
                log::error!("Objective function decreased");
            }

            self.w = self.lambda.clone();
            let pyl = self.psi.dot(&self.w);

            // Stop if we have reached convergence criteria
            if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
                self.eps /= 2.;
                if self.eps <= THETA_E {
                    self.f1 = pyl.mapv(|x| x.ln()).sum();
                    if (self.f1 - self.f0).abs() <= THETA_F {
                        log::info!("Likelihood criteria convergence");
                        self.converged = true;
                        state.stop_text = "The run converged!".to_string();
                        self.tx.send(state).unwrap();
                        break;
                    } else {
                        self.f0 = self.f1;
                        self.eps = 0.2;
                    }
                }
            }

            // Stop if we have reached maximum number of cycles
            if self.cycle >= self.settings.parsed.config.cycles {
                log::info!("Maximum number of cycles reached");
                state.stop_text = "No (max cycle)".to_string();
                self.tx.send(state).unwrap();
                break;
            }

            // Stop if stopfile exists
            if std::path::Path::new("stop").exists() {
                log::info!("Stopfile detected - breaking");
                state.stop_text = "No (stopped)".to_string();
                self.tx.send(state).unwrap();
                break;
            }
            self.cycle_log
                .push_and_write(state, self.settings.parsed.config.pmetrics_outputs.unwrap());

            self.theta =
                expansion::adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D);
            self.cycle += 1;
            self.last_objf = self.objf;
        }

        NPResult::new(
            self.scenarios.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            self.objf,
            self.cycle,
            self.converged,
            self.settings.clone(),
        )
    }
}
fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}
