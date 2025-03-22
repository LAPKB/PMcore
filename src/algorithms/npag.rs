use crate::prelude::algorithms::Algorithms;

pub use crate::routines::evaluation::ipm::burke;
pub use crate::routines::evaluation::qr;
use crate::routines::settings::Settings;

use crate::routines::output::{CycleLog, NPCycle, NPResult};
use crate::structs::psi::{calculate_psi, Psi};
use crate::structs::theta::Theta;

use anyhow::bail;
use anyhow::Result;
use faer::linalg::zip::IntoView;
use pharmsol::prelude::{
    data::{Data, ErrorModel, ErrorType},
    simulator::Equation,
};

use faer::Col;

use crate::routines::initialization;

use crate::routines::expansion::adaptative_grid::adaptative_grid;
use ndarray::Array1;
use ndarray_stats::QuantileExt;

const THETA_E: f64 = 1e-4; // Convergence criteria
const THETA_G: f64 = 1e-4; // Objective function convergence criteria
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

#[derive(Debug)]
pub struct NPAG<E: Equation> {
    equation: E,
    ranges: Vec<(f64, f64)>,
    psi: Psi,
    theta: Theta,
    lambda: Col<f64>,
    w: Col<f64>,
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
            psi: Psi::new(),
            theta: Theta::new(),
            lambda: Col::zeros(0),
            w: Col::zeros(0),
            eps: 0.2,
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            f0: -1e30,
            f1: f64::default(),
            cycle: 0,
            gamma_delta: 0.1,
            gamma: settings.error().value,
            error_type: settings.error().error_model().into(),
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

    fn get_prior(&self) -> Theta {
        initialization::sample_space(&self.settings).unwrap().into()
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

    fn set_theta(&mut self, theta: Theta) {
        self.theta = theta;
    }

    fn get_theta(&self) -> &Theta {
        &self.theta
    }

    fn psi(&self) -> &Psi {
        &self.psi
    }

    fn convergence_evaluation(&mut self) {
        let psi = self.psi.matrix();
        let w = &self.w;
        if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
            self.eps /= 2.;
            if self.eps <= THETA_E {
                let pyl = psi * w;
                self.f1 = pyl.iter().map(|x| x.ln()).sum();
                if (self.f1 - self.f0).abs() <= THETA_F {
                    tracing::info!("The model converged after {} cycles", self.cycle,);
                    self.converged = true;
                } else {
                    self.f0 = self.f1;
                    self.eps = 0.2;
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
            nspp: self.theta.nspp(),
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
        self.psi = calculate_psi(
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
                bail!("Error in IPM during evaluation: {:?}", err);
            }
        };
        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        // Filter out the support points with lambda < max(lambda)/1000
        let lambda: Array1<f64> = self.lambda.clone().into_view().iter().cloned().collect();
        let max_lambda = match lambda.max() {
            Ok(max_lambda) => max_lambda,
            Err(err) => bail!("Error in condensation: {:?}", err),
        };

        let mut keep = Vec::<usize>::new();
        for (index, lam) in self.lambda.iter().enumerate() {
            if *lam > max_lambda / 1000_f64 {
                keep.push(index);
            }
        }
        if self.psi.matrix().ncols() != keep.len() {
            tracing::debug!(
                "Lambda (max/1000) dropped {} support point(s)",
                self.psi.matrix().ncols() - keep.len(),
            );
        }

        self.theta.filter_indices(keep.as_slice());
        self.psi.filter_column_indices(keep.as_slice());

        //Rank-Revealing Factorization
        let (r, perm) = qr::calculate_r(&self.psi);

        let mut keep = Vec::<usize>::new();

        // The minimum between the number of subjects and the actual number of support points
        let keep_n = self.psi.matrix().ncols().min(self.psi.matrix().nrows());
        for i in 0..keep_n {
            let test = r.col(i).norm_l2();
            let r_diag_val = r.get(i, i);
            let ratio = r_diag_val / test;
            if ratio.abs() >= 1e-8 {
                keep.push(*perm.get(i).unwrap());
            }
        }

        // If a support point is dropped, log it as a debug message
        if self.psi.matrix().ncols() != keep.len() {
            tracing::debug!(
                "QR decomposition dropped {} support point(s)",
                self.psi.matrix().ncols() - keep.len(),
            );
        }

        // Filter to keep only the support points (rows) that are in the `keep` vector
        self.theta.filter_indices(keep.as_slice());
        // Filter to keep only the support points (columns) that are in the `keep` vector
        self.psi.filter_column_indices(keep.as_slice());

        self.validate_psi()?;
        (self.lambda, self.objf) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "Error in IPM during condensation: {:?}",
                    err
                ));
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

        let psi_up = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &ErrorModel::new(self.settings.error().poly, gamma_up, &self.error_type),
            false,
            true,
        );
        let psi_down = calculate_psi(
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
                return Err(anyhow::anyhow!("Error in IPM during optim: {:?}", err));
            }
        };
        let (lambda_down, objf_down) = match burke(&psi_down) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                //todo: write out report
                //panic!("Error in IPM: {:?}", err);
                return Err(anyhow::anyhow!("Error in IPM during optim: {:?}", err));
                //(Array1::zeros(1), f64::NEG_INFINITY)
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
        tracing::debug!("Support points: {}", self.theta.nspp());
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
