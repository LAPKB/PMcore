use crate::routines::initialization::sample_space;
use crate::routines::output::{cycles::CycleLog, cycles::NPCycle, NPResult};
use crate::structs::weights::Weights;
use crate::{
    algorithms::Status,
    prelude::{
        algorithms::Algorithms,
        routines::{
            evaluation::{ipm::burke, qr},
            settings::Settings,
        },
    },
    structs::{
        psi::{calculate_psi, Psi},
        theta::Theta,
    },
};
use pharmsol::SppOptimizer;

use anyhow::bail;
use anyhow::Result;
use faer_ext::IntoNdarray;
use pharmsol::{prelude::ErrorModel, ErrorModels};
use pharmsol::{
    prelude::{data::Data, simulator::Equation},
    Subject,
};

use ndarray::{
    parallel::prelude::{IntoParallelRefMutIterator, ParallelIterator},
    Array, Array1, ArrayBase, Dim, OwnedRepr,
};

const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

pub struct NPOD<E: Equation> {
    equation: E,
    psi: Psi,
    theta: Theta,
    lambda: Weights,
    w: Weights,
    last_objf: f64,
    objf: f64,
    cycle: usize,
    gamma_delta: Vec<f64>,
    error_models: ErrorModels,
    converged: bool,
    status: Status,
    cycle_log: CycleLog,
    data: Data,
    settings: Settings,
}

impl<E: Equation> Algorithms<E> for NPOD<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        Ok(Box::new(Self {
            equation,
            psi: Psi::new(),
            theta: Theta::new(),
            lambda: Weights::default(),
            w: Weights::default(),
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
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

    fn equation(&self) -> &E {
        &self.equation
    }

    fn get_settings(&self) -> &Settings {
        &self.settings
    }

    fn get_data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Theta {
        sample_space(&self.settings).unwrap()
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

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn convergence_evaluation(&mut self) {
        if (self.last_objf - self.objf).abs() <= THETA_F {
            tracing::info!("Objective function convergence reached");
            self.converged = true;
            self.status = Status::Converged;
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
            self.converged = true;
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
        let error_model: ErrorModels = self.error_models.clone();

        self.psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &error_model,
            self.cycle == 1 && self.settings.config().progress,
            self.cycle != 1,
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
                if *em == ErrorModel::None || em.is_factor_fixed().unwrap_or(true) {
                    None
                } else {
                    Some((outeq, em))
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
        // If no stop signal, add new point to theta based on the optimization of the D function
        let psi = self.psi().matrix().as_ref().into_ndarray().to_owned();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        // Add new point to theta based on the optimization of the D function
        let error_model: ErrorModels = self.error_models.clone();

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
            // add spp to theta
            // recalculate psi
            // re-run ipm to re-calculate w
            // re-calculate pyl
            // re-define a new optimization
        });
        for cp in candididate_points {
            self.theta.suggest_point(cp.to_vec().as_slice(), THETA_D)?;
        }
        Ok(())
    }
}

impl<E: Equation> NPOD<E> {
    fn validate_psi(&mut self) -> Result<()> {
        let mut psi = self.psi().matrix().as_ref().into_ndarray().to_owned();
        // First coerce all NaN and infinite in psi to 0.0
        if psi.iter().any(|x| x.is_nan() || x.is_infinite()) {
            tracing::warn!("Psi contains NaN or Inf values, coercing to 0.0");
            for i in 0..psi.nrows() {
                for j in 0..psi.ncols() {
                    let val = psi.get_mut((i, j)).unwrap();
                    if val.is_nan() || val.is_infinite() {
                        *val = 0.0;
                    }
                }
            }
        }

        // Calculate the sum of each column in psi
        let (_, col) = psi.dim();
        let ecol: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::ones(col);
        let plam = psi.dot(&ecol);
        let w = 1. / &plam;

        // Get the index of each element in `w` that is NaN or infinite
        let indices: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_nan() || x.is_infinite())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

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
