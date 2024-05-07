use crate::{
    prelude::{
        algorithms::Algorithm,
        condensation::prune::prune,
        evaluation::sigma::{ErrorModel, ErrorType},
        ipm::burke,
        optimization::d_optimizer::SppOptimizer,
        output::{CycleLog, NPCycle, NPResult},
        qr,
        settings::Settings,
    },
    simulator::{likelihood::PopulationPredictions, Equation},
    tui::ui::Comm,
};
use ndarray::parallel::prelude::*;
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_stats::{DeviationExt, QuantileExt};
use tokio::sync::mpsc::UnboundedSender;

use super::{data::Subject, get_population_predictions};

const THETA_D: f64 = 1e-4;
const THETA_F: f64 = 1e-2;

pub struct NPOD {
    equation: Equation,
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
    subjects: Vec<Subject>,
    c: (f64, f64, f64, f64),
    tx: Option<UnboundedSender<Comm>>,
    population_predictions: PopulationPredictions,
    settings: Settings,
}

impl Algorithm for NPOD {
    fn fit(&mut self) -> NPResult {
        self.run()
    }
    fn to_npresult(&self) -> NPResult {
        NPResult::new(
            self.subjects.clone(),
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

impl NPOD {
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
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            cycle: 1,
            gamma_delta: 0.1,
            gamma: settings.error.value,
            error_type: settings.error.error_type(),
            converged: false,
            cycle_log: CycleLog::new(&settings),
            cache: settings.config.cache,
            tx,
            settings,
            subjects,
            c,
            population_predictions: PopulationPredictions::default(),
        }
    }

    fn optim_gamma(&mut self) {
        //Gam/Lam optimization
        // TODO: Move this to e.g. /evaluation/error.rs
        let gamma_up = self.gamma * (1.0 + self.gamma_delta);
        let gamma_down = self.gamma / (1.0 + self.gamma_delta);

        let psi_up = self.population_predictions.get_psi(&ErrorModel {
            c: self.c,
            gl: gamma_up,
            e_type: &self.error_type,
        });
        let psi_down = self.population_predictions.get_psi(&ErrorModel {
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

    pub fn run(&mut self) -> NPResult {
        loop {
            // Enter a span for each cycle, providing context for further errors
            let cycle_span = tracing::span!(tracing::Level::INFO, "Cycle", cycle = self.cycle);
            let _enter = cycle_span.enter();

            self.last_objf = self.objf;

            let cache = if self.cycle == 1 { false } else { self.cache };

            self.population_predictions =
                get_population_predictions(&self.equation, &self.subjects, &self.theta, cache);

            self.psi = self.population_predictions.get_psi(&ErrorModel {
                c: self.c,
                gl: self.gamma,
                e_type: &self.error_type,
            });

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

            let removed = self.theta.shape()[0] - keep.len();
            if removed > 0 {
                tracing::debug!(
                    "Removed {} support point(s) with weight < 1/1000 of max",
                    removed
                );
            }
            self.theta = self.theta.select(Axis(0), &keep);
            self.psi = self.psi.select(Axis(1), &keep);
            self.population_predictions.subject_predictions = self
                .population_predictions
                .subject_predictions
                .select(Axis(1), &keep);

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
            self.population_predictions.subject_predictions = self
                .population_predictions
                .subject_predictions
                .select(Axis(1), &keep);

            (self.lambda, self.objf) = match burke(&self.psi) {
                Ok((lambda, objf)) => (lambda, objf),
                Err(err) => {
                    //todo: write out report
                    panic!("Error in IPM: {:?}", err);
                }
            };

            self.optim_gamma();

            // Increasing objf signals instability or model misspecification.
            if self.last_objf > self.objf {
                tracing::warn!(
                    "Objective function decreased from {:.4} to {:.4} (delta = {})",
                    -2.0 * self.last_objf,
                    -2.0 * self.objf,
                    -2.0 * self.last_objf - -2.0 * self.objf
                );
            }

            self.w = self.lambda.clone();

            // Perform checks for convergence or termination
            let mut stop = false;
            // Stop if objective function convergence is reached
            if (self.last_objf - self.objf).abs() <= THETA_F {
                tracing::info!("Objective function convergence reached");
                self.converged = true;
                stop = true;
            }
            // Stop if we have reached maximum number of cycles
            if self.cycle >= self.settings.config.cycles {
                tracing::warn!("Maximum number of cycles reached");
                stop = true;
            }

            // Stop if stopfile exists
            if std::path::Path::new("stop").exists() {
                tracing::warn!("Stopfile detected - breaking");
                stop = true;
            }

            // Create a new NPCycle state and log it
            let state = NPCycle {
                cycle: self.cycle,
                objf: -2. * self.objf,
                delta_objf: (self.last_objf - self.objf).abs(),
                nspp: self.theta.shape()[0],
                theta: self.theta.clone(),
                gamlam: self.gamma,
                converged: self.converged,
            };

            // Log relevant cycle information
            tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
            tracing::debug!("Support points: {}", self.theta.shape()[0]);
            tracing::debug!("Gamma = {:.4}", self.gamma);

            match &self.tx {
                Some(tx) => tx.send(Comm::NPCycle(state.clone())).unwrap(),
                None => (),
            }

            self.cycle_log
                .push_and_write(state, self.settings.config.output);

            if stop {
                break;
            }

            // If no stop signal, add new point to theta based on the optimization of the D function
            let pyl = self.psi.dot(&self.w);

            // Add new point to theta based on the optimization of the D function
            let sigma = ErrorModel {
                c: self.c,
                gl: self.gamma,
                e_type: &self.error_type,
            };

            let mut candididate_points: Vec<Array1<f64>> = Vec::default();
            for spp in self.theta.clone().rows() {
                candididate_points.push(spp.to_owned());
            }
            candididate_points.par_iter_mut().for_each(|spp| {
                let optimizer =
                    SppOptimizer::new(self.equation.clone(), &self.subjects, &sigma, &pyl);
                let candidate_point = optimizer.optimize_point(spp.to_owned()).unwrap();
                *spp = candidate_point;
                // add spp to theta
                // recalculate psi
                // re-run ipm to re-calculate w
                // re-calculate pyl
                // re-define a new optimization
            });
            for cp in candididate_points {
                prune(&mut self.theta, cp, &self.ranges, THETA_D);
            }

            // Increment the cycle count and prepare for the next cycle
            self.cycle += 1;
        }

        self.to_npresult()
    }
}
fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}
