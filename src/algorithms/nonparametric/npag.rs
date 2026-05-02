use crate::algorithms::{
    NativeNonparametricConfig, NonparametricAlgorithmInput, Status, StopReason,
};
use crate::api::estimation_problem::NonparametricMethod;
use crate::api::Npag;
use crate::estimation::nonparametric::{
    calculate_psi, CycleLog, NPCycle, NonparametricWorkspace, Psi, Theta, Weights,
};
use crate::prelude::algorithms::Algorithms;

pub(crate) use crate::estimation::nonparametric::ipm::burke;
pub(crate) use crate::estimation::nonparametric::qr;

use anyhow::bail;
use anyhow::Result;
use pharmsol::prelude::{
    data::{AssayErrorModels, Data},
    simulator::Equation,
};

use pharmsol::prelude::AssayErrorModel;

use crate::estimation::nonparametric::sample_space_for_parameters;

use crate::estimation::nonparametric::adaptative_grid;

#[derive(Debug)]
pub struct NPAG<E: Equation + Send + 'static> {
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
    error_models: AssayErrorModels,
    status: Status,
    cycle_log: CycleLog,
    data: Data,
    config: NativeNonparametricConfig,
    settings: Npag,
}

impl<E: Equation + Send + 'static> NPAG<E> {
    pub(crate) fn from_config(
        equation: E,
        data: Data,
        error_models: AssayErrorModels,
        config: NativeNonparametricConfig,
        settings: Npag,
    ) -> Box<Self> {
        let ranges = config.ranges.clone();
        let gamma_delta = vec![settings.error_step; error_models.len()];

        Box::new(Self {
            equation,
            ranges,
            psi: Psi::new(),
            theta: Theta::new(),
            lambda: Weights::default(),
            w: Weights::default(),
            eps: settings.eps,
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            f0: -1e30,
            f1: f64::default(),
            cycle: 0,
            gamma_delta,
            error_models,
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            data,
            config,
            settings,
        })
    }

    pub(crate) fn from_input(input: NonparametricAlgorithmInput<E>) -> Result<Box<Self>> {
        let method = match input.method {
            NonparametricMethod::Npag(method) => method,
            _ => unreachable!("NPAG::from_input requires an NPAG method"),
        };
        let config = input.native_config()?;
        let error_models = input.error_models().clone();
        let equation = input.equation;
        let data = input.data;

        Ok(Self::from_config(equation, data, error_models, config, method))
    }
}

impl<E: Equation + Send + 'static> Algorithms<E> for NPAG<E> {
    fn equation(&self) -> &E {
        &self.equation
    }
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

    fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Theta {
        sample_space_for_parameters(&self.config.parameter_space, &self.config.prior).unwrap()
    }

    fn likelihood(&self) -> f64 {
        self.objf
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

    fn evaluation(&mut self) -> Result<Status> {
        tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
        tracing::debug!("Support points: {}", self.theta.nspp());

        self.error_models.iter().for_each(|(outeq, em)| {
            if AssayErrorModel::None == *em {
                return;
            }
            tracing::debug!(
                "Error model for outeq {}: {:.2}",
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

        let psi = self.psi.matrix();
        let w = &self.w;
        if (self.last_objf - self.objf).abs() <= self.settings.objective_tolerance
            && self.eps > self.settings.min_eps
        {
            self.eps /= 2.;
            if self.eps <= self.settings.min_eps {
                let pyl = psi * w.weights();
                self.f1 = pyl.iter().map(|x| x.ln()).sum();
                if (self.f1 - self.f0).abs() <= self.settings.pyl_tolerance {
                    tracing::info!("The model converged after {} cycles", self.cycle,);
                    self.set_status(Status::Stop(StopReason::Converged));
                    self.log_cycle_state();
                    return Ok(self.status().clone());
                } else {
                    self.f0 = self.f1;
                    self.eps = self.settings.eps;
                }
            }
        }

        // Stop if we have reached maximum number of cycles
        if self.cycle >= self.config.max_cycles {
            tracing::warn!("Maximum number of cycles reached");
            self.set_status(Status::Stop(StopReason::MaxCycles));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // Stop if stopfile exists
        if std::path::Path::new("stop").exists() {
            tracing::warn!("Stopfile detected - breaking");
            self.set_status(Status::Stop(StopReason::Stopped));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // Continue with normal operation
        self.set_status(Status::Continue);
        self.log_cycle_state();
        Ok(self.status().clone())
    }

    fn estimation(&mut self) -> Result<()> {
        self.psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &self.error_models,
            self.cycle == 1 && self.config.progress,
        )?;

        if let Err(err) = self.validate_psi() {
            bail!(err);
        }

        (self.lambda, _) = match burke(&self.psi) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                bail!("Error in IPM during estimation: {:?}", err);
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
            if lam > max_lambda * self.settings.prune_threshold {
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
            if ratio.abs() >= self.settings.qr_tolerance {
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
                    self.gamma_delta[outeq] *= self.settings.error_step_growth;
                    self.lambda = lambda_up;
                    self.psi = psi_up;
                }
                if objf_down > self.objf {
                    self.error_models.set_factor(outeq, gamma_down)?;
                    self.objf = objf_down;
                    self.gamma_delta[outeq] *= self.settings.error_step_growth;
                    self.lambda = lambda_down;
                    self.psi = psi_down;
                }
                self.gamma_delta[outeq] *= self.settings.error_step_shrink;
                if self.gamma_delta[outeq] <= self.settings.min_error_step {
                    self.gamma_delta[outeq] = self.settings.error_step;
                }
                Ok(())
            })?;

        Ok(())
    }

    fn expansion(&mut self) -> Result<()> {
        adaptative_grid(
            &mut self.theta,
            self.eps,
            &self.ranges,
            self.settings.grid_tolerance,
        )?;
        Ok(())
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{OutputPlan, RuntimeOptions};
    use crate::model::{ModelDefinition, Parameter};
    use pharmsol::{fa, fetch_params, lag, AssayErrorModel, ErrorPoly, Subject, SubjectBuilderExt};

    fn simple_equation() -> pharmsol::equation::ODE {
        pharmsol::equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(
            pharmsol::equation::metadata::new("npag_settings_test")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["0"])
                .route(pharmsol::equation::Route::bolus("0").to_state("central")),
        )
        .expect("metadata attachment should validate")
    }

    fn simple_data() -> Data {
        let subject = Subject::builder("1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .build();

        Data::new(vec![subject])
    }

    #[test]
    fn from_input_uses_npag_method_settings() -> Result<()> {
        let method = Npag::new()
            .eps(0.125)
            .min_eps(0.0025)
            .objective_tolerance(3e-5)
            .pyl_tolerance(4e-3)
            .prune_threshold(2e-3)
            .qr_tolerance(9e-7)
            .grid_tolerance(8e-4)
            .error_step(0.2)
            .min_error_step(0.03)
            .error_step_growth(3.0)
            .error_step_shrink(0.25);

        let model = ModelDefinition::builder(simple_equation())
            .parameter(Parameter::bounded("ke", 0.1, 1.0))?
            .parameter(Parameter::bounded("v", 1.0, 20.0))?
            .build()?;

        let error_models = AssayErrorModels::new().add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0),
        )?;

        let input = NonparametricAlgorithmInput::new(
            NonparametricMethod::Npag(method),
            model,
            simple_data(),
            error_models,
            OutputPlan::disabled(),
            RuntimeOptions::default(),
        );

        let algorithm = NPAG::from_input(input)?;

        assert_eq!(algorithm.settings, method);
        assert_eq!(algorithm.eps, method.eps);
        assert_eq!(algorithm.gamma_delta, vec![method.error_step]);
        Ok(())
    }
}
