use pharmsol::prelude::{
    data::{Data, ErrorModel, ErrorType},
    simulator::{get_population_predictions, Equation, PopulationPredictions},
};

use crate::{
    prelude::{
        algorithms::Algorithm,
        ipm::burke,
        output::{CycleLog, NPCycle, NPResult},
        qr,
        settings::Settings,
    },
    routines::expansion::adaptative_grid::adaptative_grid,
    tui::ui::Comm,
};

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_stats::{DeviationExt, QuantileExt};
use tokio::sync::mpsc::UnboundedSender;

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
    data: Data,
    c: (f64, f64, f64, f64),
    tx: Option<UnboundedSender<Comm>>,
    population_predictions: PopulationPredictions,
    settings: Settings,
}

impl Algorithm for NPAG {
    fn fit(&mut self) -> anyhow::Result<NPResult> {
        self.run()
    }
    fn to_npresult(&self) -> NPResult {
        NPResult::new(
            self.data.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            -2. * self.objf,
            self.cycle,
            self.converged,
            self.settings.clone(),
            self.cycle_log.clone(),
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
        data: Data,
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
            objf: f64::NEG_INFINITY,
            f0: -1e30,
            f1: f64::default(),
            cycle: 1,
            gamma_delta: 0.1,
            gamma: settings.error.value,
            error_type: settings.error.error_type(),
            converged: false,
            cycle_log: CycleLog::new(),
            cache: settings.config.cache,
            tx,
            settings,
            data,
            c,
            population_predictions: PopulationPredictions::default(),
        }
    }

    fn validate_psi(&mut self) {
        // check all the elements of psi that are NaN, Inf or -Inf and report the columns and rows
        // let mut bad: Vec<(usize, usize)> = Vec::new();
        // let mut invalid = false;
        for i in 0..self.psi.nrows() {
            for j in 0..self.psi.ncols() {
                let val = self.psi.get_mut((i, j)).unwrap();
                if val.is_nan() || val.is_infinite() {
                    // invalid = true;
                    tracing::warn!("Invalid psi value: psi[{}, {}] = {}", i, j, val);
                    tracing::warn!("Set to 0.0");
                    *val = 0.0;
                    // let obspred = self
                    //     .population_predictions
                    //     .subject_predictions
                    //     .get((i, j))
                    //     .unwrap();
                    // tracing::debug!("Observed values: {:?}", &obspred.flat_observations());
                    // tracing::debug!("Predicted values: {:?}", &obspred.flat_predictions());
                    // tracing::debug!("====================================================");
                }
            }
        }
        // if invalid {
        //     panic!("Invalid psi matrix");
        // }
    }

    fn optim_gamma(&mut self) {
        //Gam/Lam optimization
        // TODO: Move this to e.g. /evaluation/error.rs
        let gamma_up = self.gamma * (1.0 + self.gamma_delta);
        let gamma_down = self.gamma / (1.0 + self.gamma_delta);

        let psi_up = self.population_predictions.get_psi(&ErrorModel::new(
            self.c,
            gamma_up,
            &self.error_type,
        ));
        let psi_down = self.population_predictions.get_psi(&ErrorModel::new(
            self.c,
            gamma_down,
            &self.error_type,
        ));

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

    pub fn run(&mut self) -> anyhow::Result<NPResult> {
        loop {
            // Enter a span for each cycle, providing context for further errors
            let cycle_span = tracing::span!(tracing::Level::INFO, "Cycle", cycle = self.cycle);
            let _enter = cycle_span.enter();

            let cache = if self.cycle == 1 { false } else { self.cache };

            self.population_predictions =
                get_population_predictions(&self.equation, &self.data, &self.theta, cache);

            self.psi = self.population_predictions.get_psi(&ErrorModel::new(
                self.c,
                self.gamma,
                &self.error_type,
            ));

            self.validate_psi();

            (self.lambda, _) = match burke(&self.psi) {
                Ok((lambda, objf)) => (lambda, objf),
                Err(err) => {
                    //todo: write out report
                    panic!("Error in IPM: {:?}", err);
                }
            };

            let mut keep = Vec::<usize>::new();
            for (index, lam) in self.lambda.iter().enumerate() {
                if *lam > self.lambda.max()? / 1000_f64 {
                    keep.push(index);
                }
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
                    "QR decomposition dropped {} support point(s)",
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

            // Log relevant cycle information
            tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
            tracing::debug!("Support points: {}", self.theta.shape()[0]);
            tracing::debug!("Gamma = {:.4}", self.gamma);
            tracing::debug!("EPS = {:.4}", self.eps);

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
            let pyl = self.psi.dot(&self.w);

            // Stop if we have reached convergence criteria
            let mut stop = false;
            if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
                self.eps /= 2.;
                if self.eps <= THETA_E {
                    self.f1 = pyl.mapv(|x| x.ln()).sum();
                    if (self.f1 - self.f0).abs() <= THETA_F {
                        tracing::info!(
                            "The run converged with the following criteria: Log-Likelihood"
                        );
                        self.converged = true;
                        stop = true;
                    } else {
                        self.f0 = self.f1;
                        self.eps = 0.2;
                    }
                }
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

            // Create state object
            let state = NPCycle {
                cycle: self.cycle,
                objf: -2. * self.objf,
                delta_objf: (self.last_objf - self.objf).abs(),
                nspp: self.theta.shape()[0],
                theta: self.theta.clone(),
                gamlam: self.gamma,
                converged: self.converged,
            };

            // Update TUI with current state
            match &self.tx {
                Some(tx) => tx.send(Comm::NPCycle(state.clone())).unwrap(),
                None => (),
            }

            // Write cycle log
            self.cycle_log.push(state);

            // Break if stop criteria are met
            if stop {
                break;
            }

            // If we have not reached convergence or otherwise stopped, expand grid and prepare for new cycle
            self.adaptative_grid();
            self.cycle += 1;
            self.last_objf = self.objf;
        }

        Ok(self.to_npresult())
    }
}
fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}
