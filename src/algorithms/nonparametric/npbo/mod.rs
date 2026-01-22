//! Non-Parametric Bayesian Optimization (NPBO)
//!
//! Uses Gaussian Process surrogate modeling with Expected Improvement acquisition
//! to efficiently explore the parameter space. The GP learns D-optimality scores
//! across the space and uses Bayesian optimization to suggest high-value regions.
//!
//! Key Features:
//! - GP surrogate model learns D-criterion landscape
//! - Expected Improvement for exploration-exploitation balance
//! - Warm-up phase with Sobol sampling for initial coverage
//! - Standard NP estimation/condensation cycle
//! - NPAG-style convergence criteria

mod constants;
mod gp;

use constants::*;
use gp::GaussianProcess;

use crate::algorithms::{Status, StopReason};
use crate::prelude::algorithms::Algorithms;
use crate::routines::estimation::ipm::burke;
use crate::routines::estimation::qr;
use crate::routines::expansion::adaptative_grid::adaptative_grid;
use crate::routines::initialization;
use crate::routines::output::{cycles::CycleLog, NPResult};
use crate::routines::settings::Settings;
use crate::structs::nonparametric::psi::{calculate_psi, Psi};
use crate::structs::nonparametric::theta::Theta;
use crate::structs::nonparametric::weights::Weights;

use anyhow::{bail, Result};
use pharmsol::prelude::{
    data::{Data, AssayErrorModels},
    simulator::Equation,
    AssayErrorModel,
};

use sobol_burley::sample;

// NPAG-style convergence constants
const THETA_E: f64 = 1e-4;
const THETA_G: f64 = 1e-4;
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

/// NPBO Algorithm State
#[derive(Debug)]
pub struct NPBO<E: Equation + Send + 'static> {
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
    settings: Settings,
    // Bayesian Optimization state
    gp: GaussianProcess,
    y_best: f64,
    warmup_complete: bool,
    stagnation_count: usize,
}

impl<E: Equation + Send + 'static> Algorithms<E> for NPBO<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        let ranges = settings.parameters().ranges();
        let n_dims = ranges.len();

        Ok(Box::new(Self {
            equation,
            ranges: ranges.clone(),
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
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            settings,
            data,
            // BO state
            gp: GaussianProcess::new(n_dims, &ranges),
            y_best: f64::NEG_INFINITY,
            warmup_complete: false,
            stagnation_count: 0,
        }))
    }

    fn equation(&self) -> &E {
        &self.equation
    }

    fn into_npresult(&self) -> Result<NPResult<E>> {
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

    fn settings(&self) -> &Settings {
        &self.settings
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn get_prior(&self) -> Theta {
        // Start with Sobol sampling for good initial coverage
        initialization::sample_space(&self.settings).unwrap()
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

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn log_cycle_state(&mut self) {
        use crate::routines::output::cycles::NPCycle;
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
        tracing::debug!("GP training points: {}", self.gp.n_points());

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

        // Update stagnation tracking
        if self.objf > self.y_best + EI_CONVERGENCE_THRESHOLD {
            self.y_best = self.objf;
            self.stagnation_count = 0;
        } else {
            self.stagnation_count += 1;
        }

        // NPAG-style convergence check
        let psi = self.psi.matrix();
        let w = &self.w;
        if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
            self.eps /= 2.;
            if self.eps <= THETA_E {
                let pyl = psi * w.weights();
                self.f1 = pyl.iter().map(|x| x.ln()).sum();
                if (self.f1 - self.f0).abs() <= THETA_F {
                    // Additional global optimality check via GP
                    if self.check_global_optimality() {
                        tracing::info!("NPBO converged after {} cycles", self.cycle);
                        self.set_status(Status::Stop(StopReason::Converged));
                        self.log_cycle_state();
                        return Ok(self.status().clone());
                    } else {
                        // Reset and continue exploring
                        self.f0 = self.f1;
                        self.eps = 0.2;
                        tracing::debug!("GP suggests unexplored regions, continuing...");
                    }
                } else {
                    self.f0 = self.f1;
                    self.eps = 0.2;
                }
            }
        }

        // Stop if stagnated too long
        if self.stagnation_count >= MAX_STAGNATION_CYCLES && self.warmup_complete {
            tracing::info!("NPBO converged (stagnation) after {} cycles", self.cycle);
            self.set_status(Status::Stop(StopReason::Converged));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // Stop if maximum cycles reached
        if self.cycle >= self.settings.config().cycles {
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
            self.cycle == 1 && self.settings.config().progress,
            self.cycle != 1,
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
        // Filter by lambda threshold (max/1000)
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

        // QR-based rank revealing factorization
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
        self.last_objf = self.objf;

        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        // Update GP with current support points and their D-optimal contributions
        self.update_gp_model()?;

        // Standard error model optimization
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

                let (lambda_up, objf_up) = burke(&psi_up)?;
                let (lambda_down, objf_down) = burke(&psi_down)?;

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
        // During warmup: use grid expansion for coverage
        if self.cycle <= WARMUP_CYCLES {
            self.warmup_complete = false;
            return self.grid_expansion();
        }

        self.warmup_complete = true;

        // After warmup: use GP-guided Bayesian optimization
        self.bo_expansion()
    }
}

// ============================================================================
// NPBO-Specific Methods
// ============================================================================

impl<E: Equation + Send + 'static> NPBO<E> {
    /// Standard grid-based expansion during warmup
    fn grid_expansion(&mut self) -> Result<()> {
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D)?;
        tracing::debug!("Grid expansion: {} total support points", self.theta.nspp());
        Ok(())
    }

    /// Bayesian optimization guided expansion
    fn bo_expansion(&mut self) -> Result<()> {
        // First, fit the GP if we have enough points
        if self.gp.n_points() >= MIN_GP_POINTS {
            // Optimize hyperparameters periodically
            if self.cycle % 5 == 0 {
                self.gp.optimize_hyperparameters(50);
            } else {
                let _ = self.gp.fit();
            }
        }

        // Generate candidate points using acquisition function
        let mut candidates = Vec::new();

        if self.gp.n_points() >= MIN_GP_POINTS {
            // Use Expected Improvement to find promising regions
            candidates.extend(self.optimize_acquisition(BATCH_SIZE));
        }

        // Also add some grid-refined points for local improvement
        if self.cycle % 3 == 0 {
            let sparse_eps = self.eps * 0.5;
            adaptative_grid(&mut self.theta, sparse_eps, &self.ranges, THETA_D * 2.0)?;
        }

        // Add Sobol exploration points for diversity
        candidates.extend(self.sobol_points(BATCH_SIZE / 2));

        // Add candidates that pass distance check
        let mut added = 0;
        for spp in candidates {
            if self.theta.check_point(&spp, THETA_D) {
                let _ = self.theta.add_point(&spp);
                added += 1;
            }
        }

        tracing::debug!(
            "BO expansion: added {} points, {} total support points",
            added,
            self.theta.nspp()
        );

        Ok(())
    }

    /// Update GP model with current support point evaluations
    fn update_gp_model(&mut self) -> Result<()> {
        // Compute D-criterion contribution for each current support point
        // Use the weight (lambda) as a proxy for importance in the D-optimal design
        let n_spp = self.theta.nspp();

        for (i, w) in self.w.iter().enumerate().take(n_spp) {
            // Get support point coordinates
            let spp: Vec<f64> = (0..self.theta.matrix().ncols())
                .map(|c| *self.theta.matrix().get(i, c))
                .collect();

            // Use log-weight as target (approximates D-criterion contribution)
            let target = w.max(1e-10).ln();

            self.gp.add_point(&spp, target);
        }

        // Prune GP if too many points
        self.gp.prune_if_needed();

        Ok(())
    }

    /// Optimize acquisition function to find promising points
    fn optimize_acquisition(&mut self, n_points: usize) -> Vec<Vec<f64>> {
        let mut best_points = Vec::new();
        let n_dims = self.ranges.len();

        // Multi-start optimization of Expected Improvement
        for restart in 0..ACQUISITION_RESTARTS {
            // Start from a random point
            let seed = (self.cycle * ACQUISITION_RESTARTS + restart) as u32;
            let mut x: Vec<f64> = (0..n_dims)
                .map(|d| {
                    let u = sample(restart as u32, d as u32, seed) as f64;
                    let (lo, hi) = self.ranges[d];
                    lo + u * (hi - lo)
                })
                .collect();

            // Simple gradient-free local optimization
            let mut best_ei = self.gp.expected_improvement(&x, self.y_best);

            for _ in 0..50 {
                let mut improved = false;

                for d in 0..n_dims {
                    let (lo, hi) = self.ranges[d];
                    let step = (hi - lo) * 0.05;

                    // Try increasing
                    let old_val = x[d];
                    x[d] = (x[d] + step).min(hi);
                    let ei = self.gp.expected_improvement(&x, self.y_best);
                    if ei > best_ei {
                        best_ei = ei;
                        improved = true;
                    } else {
                        x[d] = old_val;
                    }

                    // Try decreasing
                    x[d] = (x[d] - step).max(lo);
                    let ei = self.gp.expected_improvement(&x, self.y_best);
                    if ei > best_ei {
                        best_ei = ei;
                        improved = true;
                    } else {
                        x[d] = old_val;
                    }
                }

                if !improved {
                    break;
                }
            }

            if best_ei > EI_CONVERGENCE_THRESHOLD {
                best_points.push((x, best_ei));
            }
        }

        // Sort by EI and take top n_points
        best_points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        best_points.truncate(n_points);
        best_points.into_iter().map(|(x, _)| x).collect()
    }

    /// Generate Sobol sequence points for exploration
    fn sobol_points(&self, n: usize) -> Vec<Vec<f64>> {
        let n_dims = self.ranges.len();
        let seed = (self.cycle * 1000) as u32;

        (0..n)
            .map(|i| {
                (0..n_dims)
                    .map(|d| {
                        let u = sample((i + self.cycle * n) as u32, d as u32, seed) as f64;
                        let (lo, hi) = self.ranges[d];
                        lo + u * (hi - lo)
                    })
                    .collect()
            })
            .collect()
    }

    /// Check if GP suggests we've found global optimum
    fn check_global_optimality(&self) -> bool {
        if self.gp.n_points() < MIN_GP_POINTS {
            return false;
        }

        // Sample random points and check if any have high EI
        let test_points = 100;
        let seed = (self.cycle * 999) as u32;

        for i in 0..test_points {
            let x: Vec<f64> = (0..self.ranges.len())
                .map(|d| {
                    let u = sample(i as u32, d as u32, seed) as f64;
                    let (lo, hi) = self.ranges[d];
                    lo + u * (hi - lo)
                })
                .collect();

            let ei = self.gp.expected_improvement(&x, self.y_best);
            if ei > EI_CONVERGENCE_THRESHOLD * 10.0 {
                // Found a point with significant expected improvement
                return false;
            }
        }

        // No high-EI points found, likely at global optimum
        true
    }
}
