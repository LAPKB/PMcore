use crate::algorithms::{NativeNonparametricConfig, NonparametricAlgorithmInput, StopReason};
use crate::estimation::nonparametric::ipm::burke;
use crate::estimation::nonparametric::qr;
use crate::estimation::nonparametric::{
    calculate_psi, CycleLog, NPCycle, NonparametricWorkspace, Psi, Theta, Weights,
};
use crate::{algorithms::Status, prelude::algorithms::Algorithms};
use pharmsol::SppOptimizer;

use anyhow::bail;
use anyhow::Result;
use pharmsol::{prelude::AssayErrorModel, AssayErrorModels};
use pharmsol::{
    prelude::{data::Data, simulator::Equation},
    Subject,
};

use ndarray::Array1;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

pub struct NPOD<E: Equation + Send + 'static> {
    equation: E,
    psi: Psi,
    theta: Theta,
    lambda: Weights,
    w: Weights,
    last_objf: f64,
    objf: f64,
    cycle: usize,
    gamma_delta: Vec<f64>,
    error_models: AssayErrorModels,
    converged: bool,
    status: Status,
    cycle_log: CycleLog,
    data: Data,
    config: NativeNonparametricConfig,
}

impl<E: Equation + Send + 'static> Algorithms<E> for NPOD<E> {
    fn into_workspace(&self) -> Result<NonparametricWorkspace<E>> {
        NonparametricWorkspace::new(
            self.equation.clone(),
            self.data.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            -2. * self.objf,
            self.cycle,
            self.status.clone(),
            self.config.run_configuration.clone(),
            self.cycle_log.clone(),
        )
    }

    fn equation(&self) -> &E {
        &self.equation
    }

    fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Theta {
        crate::estimation::nonparametric::sample_space_for_parameters(
            &self.config.parameter_space,
            &self.config.prior,
        )
        .unwrap()
    }

    fn increment_cycle(&mut self) -> usize {
        self.cycle += 1;
        self.cycle
    }

    fn cycle(&self) -> usize {
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

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn log_cycle_state(&mut self) {
        let state = NPCycle::new(
            self.cycle,
            -2. * self.objf,
            self.error_models.clone(),
            self.theta.clone(),
            self.theta.nspp(),
            (self.last_objf - self.objf).abs(),
            self.status.clone(),
        );
        self.cycle_log.push(state);
        self.last_objf = self.objf;
    }

    fn evaluation(&mut self) -> Result<Status> {
        tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
        tracing::debug!("Support points: {}", self.theta.nspp());
        self.error_models.iter().for_each(|(outeq, em)| {
            if AssayErrorModel::None == *em {
                return;
            }
            tracing::debug!(
                "Error model for outeq {}: {:.16}",
                outeq,
                em.factor().unwrap_or_default()
            );
        });
        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective function decreased from {:.4} to {:.4} (delta = {})",
                -2.0 * self.last_objf,
                -2.0 * self.objf,
                -2.0 * self.last_objf - -2.0 * self.objf
            );
        }

        if (self.last_objf - self.objf).abs() <= THETA_F {
            tracing::info!("Objective function convergence reached");
            self.converged = true;
            self.set_status(Status::Stop(StopReason::Converged));
            self.log_cycle_state();
            return Ok(self.status.clone());
        }

        if self.cycle >= self.config.max_cycles {
            tracing::warn!("Maximum number of cycles reached");
            self.converged = true;
            self.set_status(Status::Stop(StopReason::MaxCycles));
            self.log_cycle_state();
            return Ok(self.status.clone());
        }

        if std::path::Path::new("stop").exists() {
            tracing::warn!("Stopfile detected - breaking");
            self.converged = true;
            self.set_status(Status::Stop(StopReason::Stopped));
            self.log_cycle_state();
            return Ok(self.status.clone());
        }

        self.status = Status::Continue;
        self.log_cycle_state();
        Ok(self.status.clone())
    }

    fn estimation(&mut self) -> Result<()> {
        let error_model: AssayErrorModels = self.error_models.clone();

        self.psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &error_model,
            self.cycle == 1 && self.config.progress,
        )?;

        if let Err(err) = self.validate_psi() {
            bail!(err);
        }

        (self.lambda, _) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                bail!(err);
            }
        };
        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
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

        let (r, perm) = qr::qrd(&self.psi)?;

        let mut keep = Vec::<usize>::new();
        let keep_n = self.psi.matrix().ncols().min(self.psi.matrix().nrows());
        for i in 0..keep_n {
            let test = r.col(i).norm_l2();
            let r_diag_val = r.get(i, i);
            let ratio = r_diag_val / test;
            if ratio.abs() >= 1e-8 {
                keep.push(*perm.get(i).unwrap());
            }
        }

        if self.psi.matrix().ncols() != keep.len() {
            tracing::debug!(
                "QR decomposition dropped {} support point(s)",
                self.psi.matrix().ncols() - keep.len(),
            );
        }

        self.theta.filter_indices(keep.as_slice());
        self.psi.filter_column_indices(keep.as_slice());

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
        self.error_models
            .clone()
            .iter_mut()
            .filter_map(|(outeq, em)| {
                if *em == AssayErrorModel::None || em.is_factor_fixed().unwrap_or(true) {
                    None
                } else {
                    Some((outeq, em))
                }
            })
            .try_for_each(|(outeq, em)| -> Result<()> {
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
                )?;
                let psi_down = calculate_psi(
                    &self.equation,
                    &self.data,
                    &self.theta,
                    &error_model_down,
                    false,
                )?;

                let (lambda_up, objf_up) = match burke(&psi_up) {
                    Ok((lambda, objf)) => (lambda, objf),
                    Err(err) => {
                        bail!("Error in IPM during optim: {:?}", err);
                    }
                };
                let (lambda_down, objf_down) = match burke(&psi_down) {
                    Ok((lambda, objf)) => (lambda, objf),
                    Err(err) => {
                        bail!("Error in IPM during optim: {:?}", err);
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

    fn expansion(&mut self) -> Result<()> {
        let pyl_col = self.psi().matrix().as_ref() * self.w.weights().as_ref();
        let pyl: Array1<f64> = pyl_col.iter().copied().collect();

        let error_model: AssayErrorModels = self.error_models.clone();

        let mut candididate_points: Vec<Array1<f64>> = Vec::default();
        for spp in self.theta.matrix().row_iter() {
            let candidate: Vec<f64> = spp.iter().cloned().collect();
            let spp = Array1::from(candidate);
            candididate_points.push(spp.to_owned());
        }
        candididate_points.par_iter_mut().for_each(|spp| {
            let optimizer = SppOptimizer::new(&self.equation, &self.data, &error_model, &pyl);
            let candidate_point = optimizer.optimize_point(spp.to_owned()).unwrap();
            *spp = candidate_point;
        });
        for cp in candididate_points {
            self.theta.suggest_point(cp.to_vec().as_slice(), THETA_D)?;
        }
        Ok(())
    }
}

impl<E: Equation + Send + 'static> NPOD<E> {
    pub(crate) fn from_input(input: NonparametricAlgorithmInput<E>) -> Result<Box<Self>> {
        let config = input.native_config()?;
        let error_models = input.error_models().clone();
        let gamma_delta = vec![0.1; error_models.len()];
        let equation = input.equation;
        let data = input.data;

        Ok(Box::new(Self {
            equation,
            psi: Psi::new(),
            theta: Theta::new(),
            lambda: Weights::default(),
            w: Weights::default(),
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            cycle: 0,
            gamma_delta,
            error_models,
            converged: false,
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            data,
            config,
        }))
    }

    fn validate_psi(&mut self) -> Result<()> {
        let mut psi = self.psi().matrix().to_owned();
        // First coerce all NaN and infinite in psi to 0.0
        let mut has_bad_values = false;
        for i in 0..psi.nrows() {
            for j in 0..psi.ncols() {
                let val = psi[(i, j)];
                if val.is_nan() || val.is_infinite() {
                    has_bad_values = true;
                    psi[(i, j)] = 0.0;
                }
            }
        }
        if has_bad_values {
            tracing::warn!("Psi contains NaN or Inf values, coercing to 0.0");
        }

        // Calculate row sums and check for zero-probability subjects
        let nrows = psi.nrows();
        let ncols = psi.ncols();
        let indices: Vec<usize> = (0..nrows)
            .filter(|&i| {
                let row_sum: f64 = (0..ncols).map(|j| psi[(i, j)]).sum();
                let w: f64 = 1.0 / row_sum;
                w.is_nan() || w.is_infinite()
            })
            .collect();

        // If any elements in `w` are NaN or infinite, return the subject IDs for each index
        if !indices.is_empty() {
            let subject: Vec<&Subject> = self.data.subjects();
            let zero_probability_subjects: Vec<&String> =
                indices.iter().map(|&i| subject[i].id()).collect();

            return Err(anyhow::anyhow!(
                "The probability of one or more subjects, given the model, is zero. The following subjects have zero probability: {:?}", zero_probability_subjects
            ));
        }

        Ok(())
    }
}
