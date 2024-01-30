use crate::{
    prelude::{
        algorithms::Algorithm,
        condensation::prune::prune,
        datafile::Scenario,
        evaluation::sigma::{ErrorPoly, ErrorType},
        ipm,
        optimization::d_optimizer::SppOptimizer,
        output::NPResult,
        output::{CycleLog, NPCycle},
        prob, qr,
        settings::Settings,
        simulation::predict::Engine,
        simulation::predict::{sim_obs, Predict},
    },
    tui::ui::Comm,
};
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_stats::{DeviationExt, QuantileExt};
use tokio::sync::mpsc::UnboundedSender;

const THETA_D: f64 = 1e-4;
const THETA_F: f64 = 1e-2;

pub struct NPOD<S>
where
    S: Predict<'static> + std::marker::Sync + Clone,
{
    engine: Engine<S>,
    ranges: Vec<(f64, f64)>,
    psi: Array2<f64>,
    theta: Array2<f64>,
    lambda: Array1<f64>,
    w: Array1<f64>,
    last_objf: f64,
    objf: f64,
    cycle: usize,
    gamma_delta: f64,
    gamma: f64,
    error_type: ErrorType,
    converged: bool,
    cycle_log: CycleLog,
    cache: bool,
    scenarios: Vec<Scenario>,
    c: (f64, f64, f64, f64),
    tx: UnboundedSender<Comm>,
    settings: Settings,
}

impl<S> Algorithm for NPOD<S>
where
    S: Predict<'static> + std::marker::Sync + Clone,
{
    fn fit(&mut self) -> NPResult {
        self.run()
    }
    fn to_npresult(&self) -> NPResult {
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

impl<S> NPOD<S>
where
    S: Predict<'static> + std::marker::Sync + Clone,
{
    /// Creates a new NPOD instance.
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
    /// Returns a new `NPOD` instance.
    pub fn new(
        sim_eng: Engine<S>,
        ranges: Vec<(f64, f64)>,
        theta: Array2<f64>,
        scenarios: Vec<Scenario>,
        c: (f64, f64, f64, f64),
        tx: UnboundedSender<Comm>,
        settings: Settings,
    ) -> Self
    where
        S: Predict<'static> + std::marker::Sync,
    {
        Self {
            engine: sim_eng,
            ranges,
            psi: Array2::default((0, 0)),
            theta,
            lambda: Array1::default(0),
            w: Array1::default(0),
            last_objf: -1e30,
            objf: f64::INFINITY,
            cycle: 1,
            gamma_delta: 0.1,
            gamma: settings.error.value,
            error_type: match settings.error.class.as_str() {
                "additive" => ErrorType::Add,
                "proportional" => ErrorType::Prop,
                _ => panic!("Error type not supported"),
            },
            converged: false,
            cycle_log: CycleLog::new(&settings.random.names()),
            cache: settings.config.cache,
            tx,
            settings,
            scenarios,
            c,
        }
    }

    fn optim_gamma(&mut self) {
        //Gam/Lam optimization
        // TODO: Move this to e.g. /evaluation/error.rs
        let gamma_up = self.gamma * (1.0 + self.gamma_delta);
        let gamma_down = self.gamma / (1.0 + self.gamma_delta);
        let ypred = sim_obs(&self.engine, &self.scenarios, &self.theta, self.cache);
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
    }

    pub fn run(&mut self) -> NPResult {
        while (self.last_objf - self.objf).abs() > THETA_F {
            self.last_objf = self.objf;
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
            tracing::info!(
                "QR decomp, cycle {}, kept: {}, thrown {}",
                self.cycle,
                keep.len(),
                self.psi.ncols() - keep.len()
            );
            self.theta = self.theta.select(Axis(0), &keep);
            self.psi = self.psi.select(Axis(1), &keep);

            (self.lambda, self.objf) = match ipm::burke(&self.psi) {
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
            self.tx.send(Comm::NPCycle(state.clone())).unwrap();

            // If the objective function decreased, log an error.
            // Increasing objf signals instability of model misspecification.
            if self.last_objf > self.objf {
                tracing::error!("Objective function decreased");
            }

            self.w = self.lambda.clone();
            let pyl = self.psi.dot(&self.w);

            // Add new point to theta based on the optimization of the D function
            let sigma = ErrorPoly {
                c: self.c,
                gl: self.gamma,
                e_type: &self.error_type,
            };
            // for spp in self.theta.clone().rows() {
            //     let optimizer = SppOptimizer::new(&self.engine, &self.scenarios, &sigma, &pyl);
            //     let candidate_point = optimizer.optimize_point(spp.to_owned()).unwrap();
            //     prune(&mut self.theta, candidate_point, &self.ranges, THETA_D);
            // }
            let mut candididate_points: Vec<Array1<f64>> = Vec::default();
            for spp in self.theta.clone().rows() {
                candididate_points.push(spp.to_owned());
            }
            candididate_points.par_iter_mut().for_each(|spp| {
                let optimizer = SppOptimizer::new(&self.engine, &self.scenarios, &sigma, &pyl);
                let candidate_point = optimizer.optimize_point(spp.to_owned()).unwrap();
                *spp = candidate_point;
            });
            for cp in candididate_points {
                prune(&mut self.theta, cp, &self.ranges, THETA_D);
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
            //TODO: the cycle migh break before reaching this point
            self.cycle_log
                .push_and_write(state, self.settings.config.output);

            self.cycle += 1;

            // log::info!("cycle: {}, objf: {}", self.cycle, self.objf);
            // dbg!((self.last_objf - self.objf).abs());
        }

        self.to_npresult()
    }
}
fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}
