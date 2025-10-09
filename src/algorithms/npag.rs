use crate::algorithms::Status;
use crate::prelude::algorithms::Algorithms;

pub use crate::routines::evaluation::ipm::burke;
pub use crate::routines::evaluation::qr;
use crate::routines::settings::Settings;

use crate::routines::output::{cycles::CycleLog, cycles::NPCycle, NPResult};
use crate::structs::psi::{calculate_psi, Psi};
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;

use anyhow::bail;
use anyhow::Result;
use pharmsol::prelude::{
    data::{Data, ErrorModels},
    simulator::Equation,
};

use pharmsol::prelude::ErrorModel;

use crate::routines::initialization;

use crate::routines::expansion::adaptative_grid::adaptative_grid;

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
    lambda: Weights,
    w: Weights,
    eps: f64,
    last_objf: f64,
    objf: f64,
    f0: f64,
    f1: f64,
    cycle: usize,
    gamma_delta: Vec<f64>,
    error_models: ErrorModels,
    converged: bool,
    status: Status,
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
            lambda: Weights::default(),
            w: Weights::default(),
            eps: 0.2,
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            f0: -1e30,
            f1: f64::default(),
            cycle: 0,
            gamma_delta: vec![0.1; settings.errormodels().len()],
            error_models: settings.errormodels().clone(),
            converged: false,
            status: Status::Starting,
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
            self.status.clone(),
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
        initialization::sample_space(&self.settings).unwrap()
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

    fn theta(&self) -> &Theta {
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
                let pyl = psi * w.weights();
                self.f1 = pyl.iter().map(|x| x.ln()).sum();
                if (self.f1 - self.f0).abs() <= THETA_F {
                    tracing::info!("The model converged after {} cycles", self.cycle,);
                    self.converged = true;
                    self.status = Status::Converged;
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
            self.status = Status::MaxCycles;
        }

        // Stop if stopfile exists
        if std::path::Path::new("stop").exists() {
            tracing::warn!("Stopfile detected - breaking");
            self.status = Status::ManualStop;
        }

        // Create state object
        let state = NPCycle::new(
            self.cycle,
            -2. * self.objf,
            self.error_models.clone(),
            self.theta.clone(),
            self.theta.nspp(),
            (self.last_objf - self.objf).abs(),
            self.status.clone(),
        );

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
            &self.error_models,
            self.cycle == 1 && self.settings.config().progress,
            self.cycle != 1,
        )?;

        if let Err(err) = self.validate_psi() {
            bail!(err);
        }

        (self.lambda, _) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda.into(), objf),
            Err(err) => {
                bail!("Error in IPM during evaluation: {:?}", err);
            }
        };
        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        // Filter out the support points with lambda < max(lambda)/1000

        let max_lambda = self
            .lambda
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        let mut keep = Vec::<usize>::new();
        for (index, lam) in self.lambda.iter().enumerate() {
            if lam > max_lambda / 1000_f64 {
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
        let (r, perm) = qr::qrd(&self.psi)?;

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
            Ok((lambda, objf)) => (lambda.into(), objf),
            Err(err) => {
                return Err(anyhow::anyhow!(
                    "Error in IPM during condensation: {:?}",
                    err
                ));
            }
        };
        self.w = self.lambda.clone().into();
        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        self.error_models
            .clone()
            .iter_mut()
            .filter_map(|(outeq, em)| {
                if em.optimize() {
                    Some((outeq, em))
                } else {
                    None
                }
            })
            .try_for_each(|(outeq, em)| -> Result<()> {
                // OPTIMIZATION

                let gamma_up = em.factor()? * (1.0 + self.gamma_delta[outeq]);
                let gamma_down = em.factor()? / (1.0 + self.gamma_delta[outeq]);

                let mut error_model_up = self.error_models.clone();
                error_model_up.set_factor(outeq, gamma_up)?;

                let mut error_model_down = self.error_models.clone();
                error_model_down.set_factor(outeq, gamma_down)?;

                let psi_up = calculate_psi(
                    &self.equation,
                    &self.data,
                    &self.theta,
                    &error_model_up,
                    false,
                    true,
                )?;
                let psi_down = calculate_psi(
                    &self.equation,
                    &self.data,
                    &self.theta,
                    &error_model_down,
                    false,
                    true,
                )?;

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
                    self.error_models.set_factor(outeq, gamma_up)?;
                    self.objf = objf_up;
                    self.gamma_delta[outeq] *= 4.;
                    self.lambda = lambda_up;
                    self.psi = psi_up;
                }
                if objf_down > self.objf {
                    self.error_models.set_factor(outeq, gamma_down)?;
                    self.objf = objf_down;
                    self.gamma_delta[outeq] *= 4.;
                    self.lambda = lambda_down;
                    self.psi = psi_down;
                }
                self.gamma_delta[outeq] *= 0.5;
                if self.gamma_delta[outeq] <= 0.01 {
                    self.gamma_delta[outeq] = 0.1;
                }
                Ok(())
            })?;

        Ok(())
    }

    fn logs(&self) {
        tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
        tracing::debug!("Support points: {}", self.theta.nspp());

        self.error_models.iter().for_each(|(outeq, em)| {
            if ErrorModel::None == *em {
                return;
            }
            tracing::debug!(
                "Error model for outeq {}: {:.16}",
                outeq,
                em.factor().unwrap_or_default()
            );
        });

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
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D)?;
        Ok(())
    }

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn status(&self) -> &Status {
        &self.status
    }
}
