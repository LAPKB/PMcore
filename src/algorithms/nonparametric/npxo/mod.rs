//! # NPXO: Non-Parametric Crossover Optimization
//!
//! Uses genetic crossover operators to explore the space between good support points.
//!
//! ## Key Innovation
//!
//! Instead of perturbing single points (SA) or following velocity (PSO), NPXO "breeds"
//! pairs of high-weight support points to create offspring in between them.
//!
//! ## Crossover Operators
//!
//! 1. **Arithmetic crossover**: child = α·parent1 + (1-α)·parent2
//! 2. **BLX-α crossover**: child sampled from extended box between parents
//! 3. **Simulated Binary Crossover (SBX)**: Mimics single-point crossover for continuous
//!
//! ## Why Crossover?
//!
//! 1. **Exploits correlations**: If two points are good, region between may be too
//! 2. **Preserves structure**: New points inherit properties from good parents
//! 3. **Fast convergence**: Directly targets promising regions
//! 4. **Low computational cost**: No gradient or surrogate needed

mod constants;
mod crossover;

pub use constants::*;

use crate::algorithms::StopReason;
use crate::routines::initialization::sample_space;
use crate::routines::output::{cycles::CycleLog, cycles::NPCycle, NPResult};
use crate::structs::nonparametric::psi::{calculate_psi, Psi};
use crate::structs::nonparametric::theta::Theta;
use crate::structs::nonparametric::weights::Weights;
use crate::{
    algorithms::Status,
    prelude::{
        algorithms::Algorithms,
        routines::{
            estimation::{ipm::burke, qr},
            settings::Settings,
        },
    },
};

use anyhow::{bail, Result};
use faer_ext::IntoNdarray;
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use pharmsol::prelude::data::Data;
use pharmsol::prelude::simulator::Equation;
use pharmsol::{prelude::AssayErrorModel, AssayErrorModels, Subject};
use rand::prelude::*;
use rand::SeedableRng;

// ============================================================================
// NPXO STRUCT
// ============================================================================

pub struct NPXO<E: Equation + Send + 'static> {
    equation: E,
    ranges: Vec<(f64, f64)>,
    psi: Psi,
    theta: Theta,
    lambda: Weights,
    w: Weights,
    last_objf: f64,
    objf: f64,
    best_objf: f64,
    cycle: usize,
    gamma_delta: Vec<f64>,
    error_models: AssayErrorModels,
    status: Status,
    cycle_log: CycleLog,
    data: Data,
    settings: Settings,

    // Crossover specific
    objf_history: Vec<f64>,
    rng: StdRng,
}

// ============================================================================
// ALGORITHMS TRAIT
// ============================================================================

impl<E: Equation + Send + 'static> Algorithms<E> for NPXO<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>> {
        let seed = settings.prior().seed().unwrap_or(42) as u64;
        let ranges = settings.parameters().ranges();

        Ok(Box::new(Self {
            equation,
            ranges,
            psi: Psi::new(),
            theta: Theta::new(),
            lambda: Weights::default(),
            w: Weights::default(),
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            best_objf: f64::NEG_INFINITY,
            cycle: 0,
            gamma_delta: vec![0.1; settings.errormodels().len()],
            error_models: settings.errormodels().clone(),
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            data,
            settings,
            objf_history: Vec::with_capacity(500),
            rng: StdRng::seed_from_u64(seed),
        }))
    }

    fn equation(&self) -> &E {
        &self.equation
    }

    fn settings(&self) -> &Settings {
        &self.settings
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Theta {
        sample_space(&self.settings).unwrap()
    }

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn increment_cycle(&mut self) -> usize {
        self.cycle += 1;
        if self.objf > self.best_objf + THETA_G {
            self.best_objf = self.objf;
        }
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

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn log_cycle_state(&mut self) {
        let state = NPCycle::new(
            self.cycle,
            -2.0 * self.objf,
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
            if AssayErrorModel::None != *em {
                tracing::debug!(
                    "Error model outeq {}: {:.4}",
                    outeq,
                    em.factor().unwrap_or_default()
                );
            }
        });

        self.objf_history.push(self.objf);

        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective decreased: {:.4} → {:.4}",
                -2.0 * self.last_objf,
                -2.0 * self.objf
            );
        }

        // Check convergence
        let converged = self.check_convergence();
        let max_cycles = self.settings.config().cycles;

        if converged {
            tracing::info!("NPXO converged at cycle {}", self.cycle);
            self.status = Status::Stop(StopReason::Converged);
        } else if self.cycle >= max_cycles {
            tracing::info!("NPXO max cycles: {}", max_cycles);
            self.status = Status::Stop(StopReason::MaxCycles);
        } else if std::path::Path::new("stop").exists() {
            tracing::warn!("Stop file detected");
            self.status = Status::Stop(StopReason::Stopped);
        }

        self.log_cycle_state();
        Ok(self.status.clone())
    }

    fn estimation(&mut self) -> Result<()> {
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

        let (lambda, objf) = burke(&self.psi)?;
        self.lambda = lambda;
        self.objf = objf;

        tracing::debug!(
            "NPXO cycle {}: -2LL = {:.4}, {} SPP",
            self.cycle,
            -2.0 * objf,
            self.theta.nspp()
        );

        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        // Lambda threshold pruning
        let max_lambda = self.lambda.iter().fold(f64::NEG_INFINITY, |acc, x| x.max(acc));
        let mut keep: Vec<usize> = self
            .lambda
            .iter()
            .enumerate()
            .filter(|(_, lam)| *lam > max_lambda / 1000.0)
            .map(|(i, _)| i)
            .collect();

        if self.psi.matrix().ncols() != keep.len() {
            tracing::debug!(
                "Lambda pruning dropped {} SPP",
                self.psi.matrix().ncols() - keep.len()
            );
        }

        self.theta.filter_indices(&keep);
        self.psi.filter_column_indices(&keep);

        // QR decomposition
        let (r, perm) = qr::qrd(&self.psi)?;
        let keep_n = self.psi.matrix().ncols().min(self.psi.matrix().nrows());

        keep.clear();
        for i in 0..keep_n {
            let test = r.col(i).norm_l2();
            let r_diag = r.get(i, i);
            if (r_diag / test).abs() >= 1e-8 {
                keep.push(*perm.get(i).unwrap());
            }
        }

        if self.psi.matrix().ncols() != keep.len() {
            tracing::debug!("QR dropped {} SPP", self.psi.matrix().ncols() - keep.len());
        }

        self.theta.filter_indices(&keep);
        self.psi.filter_column_indices(&keep);

        let (lambda, objf) = burke(&self.psi)?;
        self.lambda = lambda;
        self.objf = objf;
        self.w = self.lambda.clone();

        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        self.optimize_error_models()?;
        Ok(())
    }

    fn expansion(&mut self) -> Result<()> {
        // Generate offspring via crossover
        let offspring = crossover::generate_offspring(
            &self.theta,
            &self.w,
            &self.ranges,
            CROSSOVER_COUNT,
            &mut self.rng,
        )?;

        // Add offspring to theta
        for point in offspring {
            self.theta.suggest_point(&point, THETA_D)?;
        }

        tracing::debug!("NPXO: Expanded to {} SPP", self.theta.nspp());
        Ok(())
    }

    fn into_npresult(&self) -> Result<NPResult<E>> {
        NPResult::new(
            self.equation.clone(),
            self.data.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            -2.0 * self.objf,
            self.cycle,
            self.status.clone(),
            self.settings.clone(),
            self.cycle_log.clone(),
        )
    }
}

// ============================================================================
// NPXO SPECIFIC METHODS
// ============================================================================

impl<E: Equation + Send + 'static> NPXO<E> {
    fn check_convergence(&self) -> bool {
        if self.cycle < MIN_CYCLES {
            return false;
        }

        if self.objf_history.len() < STABLE_CYCLES {
            return false;
        }

        let recent: Vec<f64> = self.objf_history.iter().rev().take(STABLE_CYCLES).copied().collect();
        let max_val = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = recent.iter().cloned().fold(f64::INFINITY, f64::min);

        (max_val - min_val).abs() < OBJF_TOLERANCE
    }

    fn optimize_error_models(&mut self) -> Result<()> {
        for (outeq, em) in self.error_models.clone().iter_mut() {
            if *em == AssayErrorModel::None || em.is_factor_fixed().unwrap_or(true) {
                continue;
            }

            let gamma_up = em.factor()? * (1.0 + self.gamma_delta[outeq]);
            let gamma_down = em.factor()? / (1.0 + self.gamma_delta[outeq]);

            let mut em_up = self.error_models.clone();
            em_up.set_factor(outeq, gamma_up)?;

            let mut em_down = self.error_models.clone();
            em_down.set_factor(outeq, gamma_down)?;

            let psi_up = calculate_psi(&self.equation, &self.data, &self.theta, &em_up, false, true)?;
            let psi_down = calculate_psi(&self.equation, &self.data, &self.theta, &em_down, false, true)?;

            let (lambda_up, objf_up) = burke(&psi_up)?;
            let (lambda_down, objf_down) = burke(&psi_down)?;

            if objf_up > self.objf {
                self.error_models.set_factor(outeq, gamma_up)?;
                self.objf = objf_up;
                self.gamma_delta[outeq] *= 4.0;
                self.lambda = lambda_up;
                self.psi = psi_up;
            }
            if objf_down > self.objf {
                self.error_models.set_factor(outeq, gamma_down)?;
                self.objf = objf_down;
                self.gamma_delta[outeq] *= 4.0;
                self.lambda = lambda_down;
                self.psi = psi_down;
            }

            self.gamma_delta[outeq] *= 0.5;
            if self.gamma_delta[outeq] <= 0.01 {
                self.gamma_delta[outeq] = 0.1;
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn validate_psi(&self) -> Result<()> {
        let psi = self.psi.matrix().as_ref().into_ndarray();
        let (_, col) = psi.dim();
        let ecol: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::ones(col);
        let plam = psi.dot(&ecol);
        let w = 1.0 / &plam;

        let bad_indices: Vec<usize> = w
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_nan() || x.is_infinite())
            .map(|(i, _)| i)
            .collect();

        if !bad_indices.is_empty() {
            let subjects: Vec<&Subject> = self.data.subjects();
            let bad_subjects: Vec<&String> = bad_indices.iter().map(|&i| subjects[i].id()).collect();
            bail!("Zero probability for subjects: {:?}", bad_subjects);
        }

        Ok(())
    }
}
