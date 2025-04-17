use crate::prelude::algorithms::Algorithms;

pub use crate::routines::evaluation::ipm::burke;
pub use crate::routines::evaluation::qr;
use crate::routines::settings::Settings;

use crate::routines::output::{CycleLog, NPCycle, NPResult};

use anyhow::bail;
use anyhow::Result;
use pharmsol::prelude::{
    data::{Data, ErrorModel, ErrorType},
    simulator::{psi, Equation},
};

use crate::routines::initialization;

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_stats::{DeviationExt, QuantileExt};

use crate::routines::expansion::adaptative_grid::adaptative_grid;

const THETA_G: f64 = 1e-4; // inner loop; if df < theta_g then reduce eps and check for convergence
const THETA_F: f64 = 1e-2; // big loop; f1 - f0
const THETA_E: f64 = 1e-4; // min eps; if converged, exit, else reset eps to max
const MAX_EPS: f64 = 0.2;
const THETA_D: f64 = 1e-4; // adaptive grid minimum distance

#[derive(Debug)]
pub struct NPAG<E: Equation> {
    equation: E,
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
    data: Data,
    settings: Settings,
}

impl<E: Equation> Algorithms<E> for NPAG<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        Ok(Box::new(Self {
            equation,
            ranges: settings.parameters().ranges(),
            psi: Array2::default((0, 0)),
            theta: Array2::zeros((0, 0)),
            lambda: Array1::default(0),
            w: Array1::default(0),
            eps: MAX_EPS,
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            f0: -1e30,
            f1: f64::default(),
            cycle: 0,
            gamma_delta: 0.1,
            gamma: settings.error().value,
            error_type: settings.error().error_type(),
            converged: false,
            cycle_log: CycleLog::new(),
            settings,
            data,
        }))
    }

    fn equation(&self) -> &E {
        &self.equation
    }
    fn into_npresult(&self) -> NPResult<E> {
        NPResult::new(
            self.equation.clone(),
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

    fn get_settings(&self) -> &Settings {
        &self.settings
    }

    fn get_data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Array2<f64> {
        initialization::sample_space(&self.settings, &self.data, &self.equation).unwrap()
    }

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn inc_cycle(&mut self) -> usize {
        self.cycle += 1;
        self.cycle
    }

    fn get_cycle(&self) -> usize {
        self.cycle
    }

    fn set_theta(&mut self, theta: Array2<f64>) {
        self.theta = theta;
    }

    fn get_theta(&self) -> &Array2<f64> {
        &self.theta
    }

    fn psi(&self) -> &Array2<f64> {
        &self.psi
    }

    fn convergence_evaluation(&mut self) {
        if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
            self.eps /= 2.;
            if self.eps <= THETA_E {
                let pyl = self.psi.dot(&self.w);
                self.f1 = pyl.mapv(|x| x.ln()).sum();
                if (self.f1 - self.f0).abs() <= THETA_F {
                    tracing::info!("The model converged after {} cycles", self.cycle,);
                    self.converged = true;
                } else {
                    self.f0 = self.f1;
                    self.eps = MAX_EPS;
                }
            }
        }

        // Stop if we have reached maximum number of cycles
        if self.cycle >= self.settings.config().cycles {
            tracing::warn!("Maximum number of cycles reached");
            self.converged = true;
        }

        // Stop if stopfile exists
        if std::path::Path::new("stop").exists() {
            tracing::warn!("Stopfile detected - breaking");
            self.converged = true;
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

        // Write cycle log
        self.cycle_log.push(state);
        self.last_objf = self.objf;
    }

    fn converged(&self) -> bool {
        self.converged
    }

    fn evaluation(&mut self) -> Result<()> {
        self.psi = psi(
            &self.equation,
            &self.data,
            &self.theta,
            &ErrorModel::new(self.settings.error().poly, self.gamma, &self.error_type),
            self.cycle == 1 && self.settings.log().write,
            self.cycle != 1,
        );

        if let Err(err) = self.validate_psi() {
            bail!(err);
        }

        (self.lambda, _) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                bail!("Error in IPM: {:?}", err);
            }
        };
        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        let max_lambda = match self.lambda.max() {
            Ok(max_lambda) => max_lambda,
            Err(err) => bail!("Error in IPM: {:?}", err),
        };

        let mut keep = Vec::<usize>::new();
        for (index, lam) in self.lambda.iter().enumerate() {
            if *lam > max_lambda / 1000_f64 {
                keep.push(index);
            }
        }
        if self.psi.ncols() != keep.len() {
            tracing::debug!(
                "Lambda (max/1000) dropped {} support point(s)",
                self.psi.ncols() - keep.len(),
            );
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
                "QR decomposition dropped {} support point(s)",
                self.psi.ncols() - keep.len(),
            );
        }

        self.theta = self.theta.select(Axis(0), &keep);
        self.psi = self.psi.select(Axis(1), &keep);

        (self.lambda, self.objf) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                return Err(anyhow::anyhow!("Error in IPM: {:?}", err));
            }
        };
        self.w = self.lambda.clone();
        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        // Gam/Lam optimization
        // TODO: Move this to e.g. /evaluation/error.rs
        let gamma_up = self.gamma * (1.0 + self.gamma_delta);
        let gamma_down = self.gamma / (1.0 + self.gamma_delta);

        let psi_up = psi(
            &self.equation,
            &self.data,
            &self.theta,
            &ErrorModel::new(self.settings.error().poly, gamma_up, &self.error_type),
            false,
            true,
        );
        let psi_down = psi(
            &self.equation,
            &self.data,
            &self.theta,
            &ErrorModel::new(self.settings.error().poly, gamma_down, &self.error_type),
            false,
            true,
        );

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
                //panic!("Error in IPM: {:?}", err);
                tracing::warn!("Error in IPM: {:?}. Trying to recover.", err);
                (Array1::zeros(1), f64::NEG_INFINITY)
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
        Ok(())
    }

    fn logs(&self) {
        tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
        tracing::debug!("Support points: {}", self.theta.shape()[0]);
        tracing::debug!("Gamma = {:.16}", self.gamma);
        tracing::debug!("EPS = {:.4}", self.eps);
        // Increasing objf signals instability or model misspecification.
        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective function decreased from {:.4} to {:.4} (delta = {})",
                -2.0 * self.last_objf,
                -2.0 * self.objf,
                -2.0 * self.last_objf - -2.0 * self.objf
            );
        }
    }

    fn expansion(&mut self) -> Result<()> {
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D);
        Ok(())
    }
}

fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}
