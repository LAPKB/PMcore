use crate::{
    prelude::{
        algorithms::Algorithm,
        datafile::Scenario,
        evaluation::sigma::{ErrorModel, ErrorType},
        ipm::burke,
        output::{CycleLog, NPCycle, NPResult},
        prob, qr,
        settings::Settings,
        simulation::predict::Predict,
    },
    routines::expansion::adaptative_grid::adaptative_grid,
    simulator::Equation,
    tui::ui::Comm,
};

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_stats::{DeviationExt, QuantileExt};
use tokio::sync::mpsc::UnboundedSender;

use super::{data::Subject, get_population_predictions};

const THETA_E: f64 = 1e-4; // Convergence criteria
const THETA_G: f64 = 1e-4; // Objective function convergence criteria
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

pub struct NPAG {
    equation: Equation,
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
    subjects: Vec<Subject>,
    c: (f64, f64, f64, f64),
    tx: Option<UnboundedSender<Comm>>,
    settings: Settings,
}

impl Algorithm for NPAG {
    fn fit(&mut self) -> NPResult {
        self.run()
    }
    fn to_npresult(&self) -> NPResult {
        NPResult::new(
            self.subjects.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            -2. * self.objf,
            self.cycle,
            self.converged,
            self.settings.clone(),
        )
    }
}

impl NPAG {
    /// Creates a new NPAG instance.
    ///
    /// # Parameters
    ///
    /// - `sim_eng`: An instance of the prediction engine.
    /// - `ranges`: A vector of value ranges for each parameter.
    /// - `theta`: An initial parameter matrix.
    /// - `scenarios`: A vector of scenarios.
    /// - `c`: A tuple containing coefficients for the error polynomial.
    /// - `tx`: An unbounded sender for communicating progress.
    /// - `settings`: Data settings and configurations.
    ///
    /// # Returns
    ///
    /// Returns a new `NPAG` instance.
    pub fn new(
        equation: Equation,
        ranges: Vec<(f64, f64)>,
        theta: Array2<f64>,
        subjects: Vec<Subject>,
        c: (f64, f64, f64, f64),
        tx: Option<UnboundedSender<Comm>>,
        settings: Settings,
    ) -> Self {
        Self {
            equation,
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
            gamma: settings.error.value,
            error_type: match settings.error.class.to_lowercase().as_str() {
                "additive" => ErrorType::Add,
                "lambda" => ErrorType::Add,
                "l" => ErrorType::Add,
                "proportional" => ErrorType::Prop,
                "gamma" => ErrorType::Prop,
                "g" => ErrorType::Prop,
                _ => panic!("Error type not supported"),
            },
            converged: false,
            cycle_log: CycleLog::new(&settings.random.names()),
            cache: settings.config.cache,
            tx,
            settings,
            subjects,
            c,
        }
    }

    fn optim_gamma(&mut self) {
        //Gam/Lam optimization
        // TODO: Move this to e.g. /evaluation/error.rs
        let gamma_up = self.gamma * (1.0 + self.gamma_delta);
        let gamma_down = self.gamma / (1.0 + self.gamma_delta);
        let obs_pred =
            get_population_predictions(&self.equation, &self.subjects, &self.theta, self.cache);

        let psi_up = obs_pred.get_psi(&ErrorModel {
            c: self.c,
            gl: gamma_up,
            e_type: &self.error_type,
        });
        let psi_down = obs_pred.get_psi(&ErrorModel {
            c: self.c,
            gl: gamma_down,
            e_type: &self.error_type,
        });

        let (lambda_up, objf_up) = match burke(&psi_up) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                //todo: write out report
                panic!("Error in IPM: {:?}", err);
            }
        };
        let (lambda_down, objf_down) = match burke(&psi_down) {
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
    }

    fn adaptative_grid(&mut self) {
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D);
    }

    pub fn run(&mut self) -> NPResult {
        while self.eps > THETA_E {
            // Enter a span for each cycle, providing context for further errors
            let cycle_span = tracing::span!(tracing::Level::INFO, "Cycle", cycle = self.cycle);
            let _enter = cycle_span.enter();

            let cache = if self.cycle == 1 { false } else { self.cache };
            {
                let obs_pred =
                    get_population_predictions(&self.equation, &self.subjects, &self.theta, cache);

                self.psi = obs_pred.get_psi(&ErrorModel {
                    c: self.c,
                    gl: self.gamma,
                    e_type: &self.error_type,
                });
            }

            (self.lambda, _) = match burke(&self.psi) {
                Ok((lambda, objf)) => (lambda, objf),
                Err(err) => {
                    //todo: write out report
                    panic!("Error in IPM: {:?}", err);
                }
            };

            let mut keep = Vec::<usize>::new();
            for (index, lam) in self.lambda.iter().enumerate() {
                if *lam > self.lambda.max().unwrap() / 1000_f64 {
                    keep.push(index);
                }
            }

            self.theta = self.theta.select(Axis(0), &keep);
            self.psi = self.psi.select(Axis(1), &keep);

            //Rank-Revealing Factorization
            let (r, perm) = qr::calculate_r(&self.psi);

            let mut keep = Vec::<usize>::new();
            //The minimum between the number of subjects and the actual number of support points
            let lim_loop = self.psi.nrows().min(self.psi.ncols());
            for i in 0..lim_loop {
                let test = norm_zero(&r.column(i).to_owned());
                let ratio = r.get((i, i)).unwrap() / test;
                if ratio.abs() >= 1e-8 {
                    keep.push(*perm.get(i).unwrap());
                }
            }

            // If a support point is dropped, log it as a debug message
            if self.psi.ncols() != keep.len() {
                tracing::debug!(
                    "QRD dropped {} support point(s)",
                    self.psi.ncols() - keep.len(),
                );
            }

            self.theta = self.theta.select(Axis(0), &keep);
            self.psi = self.psi.select(Axis(1), &keep);

            (self.lambda, self.objf) = match burke(&self.psi) {
                Ok((lambda, objf)) => (lambda, objf),
                Err(err) => {
                    //todo: write out report
                    panic!("Error in IPM: {:?}", err);
                }
            };

            self.optim_gamma();

            let state = NPCycle {
                cycle: self.cycle,
                objf: -2. * self.objf,
                delta_objf: (self.last_objf - self.objf).abs(),
                nspp: self.theta.shape()[0],
                theta: self.theta.clone(),
                gamlam: self.gamma,
            };
            match &self.tx {
                Some(tx) => tx.send(Comm::NPCycle(state.clone())).unwrap(),
                None => (),
            }

            // Increasing objf signals instability or model misspecification.
            if self.last_objf > self.objf {
                tracing::warn!(
                    "Objective function decreased from {} to {}",
                    self.last_objf,
                    self.objf
                );
            }

            self.w = self.lambda.clone();
            let pyl = self.psi.dot(&self.w);

            self.cycle_log
                .push_and_write(state, self.settings.config.output);

            // Stop if we have reached convergence criteria
            if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
                self.eps /= 2.;
                if self.eps <= THETA_E {
                    self.f1 = pyl.mapv(|x| x.ln()).sum();
                    if (self.f1 - self.f0).abs() <= THETA_F {
                        tracing::info!("The run converged");
                        self.converged = true;
                        break;
                    } else {
                        self.f0 = self.f1;
                        self.eps = 0.2;
                    }
                }
            }

            // Stop if we have reached maximum number of cycles
            if self.cycle >= self.settings.config.cycles {
                tracing::warn!("Maximum number of cycles reached");
                break;
            }

            // Stop if stopfile exists
            if std::path::Path::new("stop").exists() {
                tracing::warn!("Stopfile detected - breaking");
                break;
            }

            // If we have not reached convergence or otherwise stopped, expand grid and prepare for new cycle
            self.adaptative_grid();
            self.cycle += 1;
            self.last_objf = self.objf;
        }

        self.to_npresult()
    }
}
fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}
