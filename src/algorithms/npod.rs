use crate::prelude::{
    algorithms::Algorithms,
    routines::evaluation::ipm::burke,
    routines::evaluation::qr,
    routines::output::{CycleLog, NPCycle, NPResult},
    routines::settings::Settings,
};
use anyhow::bail;
use anyhow::Result;
use pharmsol::{
    prelude::{
        data::{Data, ErrorModel},
        simulator::{psi, Equation},
    },
    Subject,
};

use ndarray::{
    parallel::prelude::{IntoParallelRefMutIterator, ParallelIterator},
    Array, Array1, Array2, ArrayBase, Axis, Dim, OwnedRepr,
};
use ndarray_stats::{DeviationExt, QuantileExt};

use crate::routines::{condensation::prune, initialization, optimization::SppOptimizer};

const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

pub struct NPOD<E: Equation> {
    equation: E,
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
    converged: bool,
    cycle_log: CycleLog,
    data: Data,
    settings: Settings,
}

impl<E: Equation> Algorithms<E> for NPOD<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        Ok(Box::new(Self {
            equation,
            ranges: settings.parameters().ranges(),
            psi: Array2::default((0, 0)),
            theta: Array2::zeros((0, 0)),
            lambda: Array1::default(0),
            w: Array1::default(0),
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            cycle: 0,
            gamma_delta: 0.1,
            gamma: settings.error().value,
            converged: false,
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
            self.converged,
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

    fn get_prior(&self) -> Array2<f64> {
        initialization::sample_space(&self.settings, &self.data, &self.equation).unwrap()
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

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn convergence_evaluation(&mut self) {
        if (self.last_objf - self.objf).abs() <= THETA_F {
            tracing::info!("Objective function convergence reached");
            self.converged = true;
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
            &ErrorModel::new(
                self.settings.error().poly,
                self.gamma,
                &self.settings.error().error_model().into(),
            ),
            self.cycle == 1,
            self.cycle != 1,
        );

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
        let max_lambda = match self.lambda.max() {
            Ok(max_lambda) => max_lambda,
            Err(err) => bail!(err),
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
                bail!(err);
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
            &ErrorModel::new(
                self.settings.error().poly,
                self.gamma,
                &self.settings.error().error_model().into(),
            ),
            false,
            true,
        );
        let psi_down = psi(
            &self.equation,
            &self.data,
            &self.theta,
            &ErrorModel::new(
                self.settings.error().poly,
                self.gamma,
                &self.settings.error().error_model().into(),
            ),
            false,
            true,
        );

        let (lambda_up, objf_up) = match burke(&psi_up) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
                panic!("Error in IPM: {:?}", err);
            }
        };
        let (lambda_down, objf_down) = match burke(&psi_down) {
            Ok((lambda, objf)) => (lambda, objf),
            Err(err) => {
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
        let pyl = self.psi.dot(&self.w);

        // Add new point to theta based on the optimization of the D function
        let error_type = self.settings.error().error_model().into();
        let sigma = &ErrorModel::new(self.settings.error().poly, self.gamma, &error_type);

        let mut candididate_points: Vec<Array1<f64>> = Vec::default();
        for spp in self.theta.clone().rows() {
            candididate_points.push(spp.to_owned());
        }
        candididate_points.par_iter_mut().for_each(|spp| {
            let optimizer = SppOptimizer::new(&self.equation, &self.data, &sigma, &pyl);
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
        Ok(())
    }
}

impl<E: Equation> NPOD<E> {
    fn validate_psi(&mut self) -> Result<()> {
        // First coerce all NaN and infinite in psi to 0.0
        if self.psi.iter().any(|x| x.is_nan() || x.is_infinite()) {
            tracing::warn!("Psi contains NaN or Inf values, coercing to 0.0");
            for i in 0..self.psi.nrows() {
                for j in 0..self.psi.ncols() {
                    let val = self.psi.get_mut((i, j)).unwrap();
                    if val.is_nan() || val.is_infinite() {
                        *val = 0.0;
                    }
                }
            }
        }

        let psi = self.psi.clone();

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
            let subject: Vec<&Subject> = self.data.get_subjects();
            let zero_probability_subjects: Vec<&String> =
                indices.iter().map(|&i| subject[i].id()).collect();

            return Err(anyhow::anyhow!(
                "The probability of one or more subjects, given the model, is zero. The following subjects have zero probability: {:?}", zero_probability_subjects
            ));
        }

        Ok(())
    }
}

fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}
