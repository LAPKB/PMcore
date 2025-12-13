//! # NPSA-H: Non-Parametric Simulated Annealing Hybrid Algorithm
//!
//! This module implements the NPSA-H algorithm, which combines:
//! - **NPAG's systematic grid exploration** for broad parameter space coverage
//! - **NPOD's D-optimal refinement** for information-driven point placement
//! - **Simulated Annealing** for global mode discovery and escaping local optima
//!
//! ## Algorithm Phases
//!
//! ### Phase 1: Warm-up (Cycles 1-N)
//! Uses NPAG-style grid expansion to ensure broad parameter space sampling.
//!
//! ### Phase 2: Hybrid Expansion (Subsequent cycles)
//! - **Sparse Grid Expansion**: Adaptive NPAG-style expansion in low-density regions
//! - **D-Optimal Refinement**: Full optimization for high-weight support points
//! - **SA Injection**: Random point injection with Metropolis acceptance
//!
//! ## Convergence
//! Multi-criterion convergence checking:
//! 1. Objective function stability over consecutive cycles
//! 2. Global optimality check via Monte Carlo sampling
//! 3. Support point location stability

use crate::algorithms::{Status, StopReason};
use crate::prelude::algorithms::Algorithms;
use crate::routines::estimation::ipm::burke;
use crate::routines::estimation::qr;
use crate::routines::expansion::adaptative_grid::adaptative_grid;
use crate::routines::initialization::sample_space;
use crate::routines::output::{cycles::CycleLog, cycles::NPCycle, NPResult};
use crate::routines::settings::Settings;
use crate::structs::psi::{calculate_psi, Psi};
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;

use anyhow::{bail, Result};
use faer_ext::IntoNdarray;
use ndarray::parallel::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use ndarray::Array1;
use pharmsol::prelude::ErrorModel;
use pharmsol::prelude::{
    data::{Data, ErrorModels},
    simulator::Equation,
};
use rand::prelude::*;

// ============================================================================
// ALGORITHM CONSTANTS
// ============================================================================

/// Grid spacing convergence threshold
const THETA_E: f64 = 1e-4;
/// Objective function convergence threshold
const THETA_G: f64 = 1e-4;
/// P(Y|L) convergence criterion
const THETA_F: f64 = 1e-2;
/// Minimum distance between support points
const THETA_D: f64 = 1e-4;

/// Number of warm-up cycles using NPAG-style expansion
const WARMUP_CYCLES: usize = 5;
/// Initial temperature for simulated annealing
const INITIAL_TEMPERATURE: f64 = 1.0;
/// Temperature cooling rate per cycle
const COOLING_RATE: f64 = 0.95;
/// Number of SA points to inject per cycle
const SA_INJECT_COUNT: usize = 10;
/// Threshold for considering a support point "high importance"
const HIGH_IMPORTANCE_THRESHOLD: f64 = 0.1;
/// Maximum Nelder-Mead iterations for high-importance points
const HIGH_IMPORTANCE_MAX_ITERS: u64 = 100;
/// Maximum Nelder-Mead iterations for low-importance points
const LOW_IMPORTANCE_MAX_ITERS: u64 = 10;
/// Number of consecutive stable cycles required for convergence
const CONVERGENCE_WINDOW: usize = 3;
/// Number of Monte Carlo samples for global optimality check
const GLOBAL_OPTIMALITY_SAMPLES: usize = 500;
/// Threshold for D-criterion in global optimality check
const GLOBAL_OPTIMALITY_THRESHOLD: f64 = 0.01;
/// Minimum temperature before SA injection stops
const MIN_TEMPERATURE: f64 = 0.01;

// ============================================================================
// NPSA-H STRUCT
// ============================================================================

/// NPSA-H: Non-Parametric Simulated Annealing Hybrid Algorithm
///
/// Combines NPAG grid exploration, NPOD D-optimal refinement, and simulated
/// annealing for robust non-parametric population PK/PD modeling.
#[derive(Debug)]
pub struct NPSAH<E: Equation + Send + 'static> {
    /// The pharmacometric equation/model
    equation: E,
    /// Parameter ranges for each dimension
    ranges: Vec<(f64, f64)>,
    /// Probability matrix: P(y_i | θ_j)
    psi: Psi,
    /// Support points (parameter values)
    theta: Theta,
    /// Weights from IPM before condensation
    lambda: Weights,
    /// Final weights after condensation
    w: Weights,
    /// Current grid spacing (NPAG-style)
    eps: f64,
    /// Previous objective function value
    last_objf: f64,
    /// Current objective function value
    objf: f64,
    /// P(Y|L) values for convergence checking
    f0: f64,
    f1: f64,
    /// Current cycle number
    cycle: usize,
    /// Step sizes for error model optimization
    gamma_delta: Vec<f64>,
    /// Error models for observations
    error_models: ErrorModels,
    /// Algorithm status
    status: Status,
    /// Cycle log for tracking progress
    cycle_log: CycleLog,
    /// Subject data
    data: Data,
    /// Algorithm settings
    settings: Settings,

    // NPSA-H specific fields
    /// Current simulated annealing temperature
    temperature: f64,
    /// History of objective function values for convergence checking
    objf_history: Vec<f64>,
    /// Random number generator for SA
    rng: StdRng,
    /// Flag indicating if we're in warm-up phase
    in_warmup: bool,
    /// Maximum D-criterion value found in global search
    max_global_d: f64,
}

// ============================================================================
// ALGORITHMS TRAIT IMPLEMENTATION
// ============================================================================

impl<E: Equation + Send + 'static> Algorithms<E> for NPSAH<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        let seed = settings.prior().seed().unwrap_or(42);

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
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            data,
            settings,
            // NPSA-H specific initialization
            temperature: INITIAL_TEMPERATURE,
            objf_history: Vec::new(),
            rng: StdRng::seed_from_u64(seed as u64),
            in_warmup: true,
            max_global_d: f64::INFINITY,
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
        sample_space(&self.settings).unwrap()
    }

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn increment_cycle(&mut self) -> usize {
        self.cycle += 1;

        // Check if we're exiting warm-up phase
        if self.cycle > WARMUP_CYCLES && self.in_warmup {
            self.in_warmup = false;
            tracing::info!("NPSA-H: Exiting warm-up phase, entering hybrid expansion mode");
        }

        // Cool the temperature
        self.temperature *= COOLING_RATE;
        if self.temperature < MIN_TEMPERATURE {
            self.temperature = MIN_TEMPERATURE;
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

    fn evaluation(&mut self) -> Result<Status> {
        tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
        tracing::debug!("Support points: {}", self.theta.nspp());
        tracing::debug!(
            "Phase: {} | Temperature: {:.4}",
            if self.in_warmup { "Warm-up" } else { "Hybrid" },
            self.temperature
        );

        self.error_models.iter().for_each(|(outeq, em)| {
            if ErrorModel::None == *em {
                return;
            }
            tracing::debug!(
                "Error model for outeq {}: {:.4}",
                outeq,
                em.factor().unwrap_or_default()
            );
        });

        tracing::debug!("EPS = {:.4}", self.eps);

        // Track objective function history
        self.objf_history.push(self.objf);

        // Warn if objective function decreased (instability)
        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective function decreased from {:.4} to {:.4} (delta = {:.6})",
                -2.0 * self.last_objf,
                -2.0 * self.objf,
                -2.0 * self.last_objf - -2.0 * self.objf
            );
        }

        // Multi-criterion convergence check
        let converged = self.check_convergence()?;

        if converged {
            tracing::info!(
                "NPSA-H converged after {} cycles (multi-criterion)",
                self.cycle
            );
            self.set_status(Status::Stop(StopReason::Converged));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // Standard convergence check (NPAG-style)
        let psi = self.psi.matrix();
        let w = &self.w;
        if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
            self.eps /= 2.;
            tracing::debug!("Halving eps to {:.6}", self.eps);

            if self.eps <= THETA_E {
                let pyl = psi * w.weights();
                self.f1 = pyl.iter().map(|x| x.ln()).sum();
                if (self.f1 - self.f0).abs() <= THETA_F {
                    tracing::info!(
                        "NPSA-H converged after {} cycles (P(Y|L) criterion)",
                        self.cycle
                    );
                    self.set_status(Status::Stop(StopReason::Converged));
                    self.log_cycle_state();
                    return Ok(self.status().clone());
                } else {
                    self.f0 = self.f1;
                    self.eps = 0.2;
                }
            }
        }

        // Check maximum cycles
        if self.cycle >= self.settings.config().cycles {
            tracing::warn!("Maximum number of cycles reached");
            self.set_status(Status::Stop(StopReason::MaxCycles));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // Check for stop file
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
        // Lambda-filter: Remove points with very low weight
        let max_lambda = self
            .lambda
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        let mut keep = Vec::<usize>::new();
        // Use more aggressive filtering (1/10000 instead of 1/1000)
        let filter_threshold = max_lambda / 10000_f64;
        for (index, lam) in self.lambda.iter().enumerate() {
            if lam > filter_threshold {
                keep.push(index);
            }
        }

        if self.psi.matrix().ncols() != keep.len() {
            tracing::debug!(
                "Lambda filter dropped {} support point(s)",
                self.psi.matrix().ncols() - keep.len(),
            );
        }

        self.theta.filter_indices(keep.as_slice());
        self.psi.filter_column_indices(keep.as_slice());

        // Rank-Revealing QR Factorization
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
        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        // Same error model optimization as NPAG/NPOD
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
        if self.in_warmup {
            // Phase 1: NPAG-style grid expansion for broad coverage
            self.npag_expansion()?;
        } else {
            // Phase 2: Hybrid expansion
            self.hybrid_expansion()?;
        }
        Ok(())
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

// ============================================================================
// NPSA-H SPECIFIC METHODS
// ============================================================================

impl<E: Equation + Send + 'static> NPSAH<E> {
    /// NPAG-style adaptive grid expansion
    fn npag_expansion(&mut self) -> Result<()> {
        tracing::debug!("Performing NPAG-style grid expansion (warm-up phase)");
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D)?;
        Ok(())
    }

    /// Hybrid expansion combining grid, D-optimal, and SA
    fn hybrid_expansion(&mut self) -> Result<()> {
        let initial_points = self.theta.nspp();

        // 2a. D-optimal refinement for EXISTING points (must happen before grid expansion)
        // because we need the weights which correspond to current theta
        self.d_optimal_refinement()?;
        let after_dopt = self.theta.nspp();

        // 2b. Sparse grid expansion in low-density regions
        self.sparse_grid_expansion()?;
        let after_grid = self.theta.nspp();

        // 2c. Simulated annealing injection
        if self.temperature > MIN_TEMPERATURE {
            self.sa_injection()?;
        }
        let after_sa = self.theta.nspp();

        tracing::debug!(
            "Hybrid expansion: {} → {} (D-opt) → {} (grid) → {} (SA)",
            initial_points,
            after_dopt,
            after_grid,
            after_sa
        );

        Ok(())
    }

    /// Sparse grid expansion: only expand in low-density regions with high D-criterion
    fn sparse_grid_expansion(&mut self) -> Result<()> {
        // Use a reduced epsilon for sparse expansion
        let sparse_eps = self.eps * 0.5;
        adaptative_grid(&mut self.theta, sparse_eps, &self.ranges, THETA_D * 2.0)?;
        Ok(())
    }

    /// D-optimal refinement with adaptive iteration count based on importance
    fn d_optimal_refinement(&mut self) -> Result<()> {
        let psi = self.psi().matrix().as_ref().into_ndarray().to_owned();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        let error_model: ErrorModels = self.error_models.clone();
        let max_weight = self.w.iter().fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        // Ensure we only iterate over points that have corresponding weights
        let n_points_with_weights = self.w.len().min(self.theta.nspp());

        let mut candidate_points: Vec<(Array1<f64>, f64)> = Vec::default();

        // Collect points with their importance (weight ratio)
        // Only process points that have corresponding weights
        for (idx, spp) in self
            .theta
            .matrix()
            .row_iter()
            .enumerate()
            .take(n_points_with_weights)
        {
            let candidate: Vec<f64> = spp.iter().cloned().collect();
            let importance = self.w[idx] / max_weight;
            candidate_points.push((Array1::from(candidate), importance));
        }

        // Optimize points in parallel with adaptive iterations
        candidate_points
            .par_iter_mut()
            .for_each(|(spp, importance)| {
                let max_iters = if *importance > HIGH_IMPORTANCE_THRESHOLD {
                    HIGH_IMPORTANCE_MAX_ITERS
                } else {
                    LOW_IMPORTANCE_MAX_ITERS
                };

                let optimizer = SppOptimizerAdaptive::new(
                    &self.equation,
                    &self.data,
                    &error_model,
                    &pyl,
                    max_iters,
                );

                if let Ok(candidate_point) = optimizer.optimize_point(spp.clone()) {
                    *spp = candidate_point;
                }
            });

        // Add optimized points to theta
        for (cp, _) in candidate_points {
            self.theta.suggest_point(cp.to_vec().as_slice(), THETA_D)?;
        }

        Ok(())
    }

    /// Simulated annealing point injection for global exploration
    fn sa_injection(&mut self) -> Result<()> {
        let psi = self.psi().matrix().as_ref().into_ndarray().to_owned();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        let n_inject = (SA_INJECT_COUNT as f64 * self.temperature).ceil() as usize;
        let mut accepted_points = 0;
        let mut max_d_found = f64::NEG_INFINITY;

        for _ in 0..n_inject * 10 {
            // Generate random point in parameter space
            let point: Vec<f64> = self
                .ranges
                .iter()
                .map(|(lo, hi)| self.rng.random_range(*lo..*hi))
                .collect();

            // Compute D-criterion for this point
            let d_value = self.compute_d_criterion(&point, &pyl)?;
            max_d_found = max_d_found.max(d_value);

            // Metropolis acceptance criterion
            let accept = if d_value > 0.0 {
                true
            } else {
                let p_accept = (d_value / self.temperature).exp();
                self.rng.random::<f64>() < p_accept
            };

            if accept {
                if self.theta.check_point(&point, THETA_D) {
                    self.theta.add_point(&point)?;
                    accepted_points += 1;
                }
            }

            if accepted_points >= n_inject {
                break;
            }
        }

        self.max_global_d = max_d_found;

        tracing::debug!(
            "SA injection: {} points accepted, max D = {:.6}, T = {:.4}",
            accepted_points,
            max_d_found,
            self.temperature
        );

        Ok(())
    }

    /// Compute D-criterion for a candidate point
    fn compute_d_criterion(&self, point: &[f64], pyl: &Array1<f64>) -> Result<f64> {
        let theta_single = ndarray::Array1::from(point.to_vec()).insert_axis(ndarray::Axis(0));

        let psi_single = pharmsol::prelude::simulator::psi(
            &self.equation,
            &self.data,
            &theta_single,
            &self.error_models,
            false,
            false,
        )?;

        let nsub = psi_single.nrows() as f64;
        let mut d_sum = -nsub;

        for (p_i, pyl_i) in psi_single.iter().zip(pyl.iter()) {
            d_sum += p_i / pyl_i;
        }

        Ok(d_sum)
    }

    /// Multi-criterion convergence check
    fn check_convergence(&mut self) -> Result<bool> {
        // Need at least CONVERGENCE_WINDOW cycles to check
        if self.objf_history.len() < CONVERGENCE_WINDOW {
            return Ok(false);
        }

        // Criterion 1: Objective function stability
        let recent_objfs: Vec<f64> = self
            .objf_history
            .iter()
            .rev()
            .take(CONVERGENCE_WINDOW)
            .cloned()
            .collect();

        let objf_stable = recent_objfs
            .windows(2)
            .all(|w| (w[0] - w[1]).abs() < THETA_G);

        if !objf_stable {
            return Ok(false);
        }

        // Criterion 2: Global optimality check (only if not in warmup)
        if !self.in_warmup && self.temperature > MIN_TEMPERATURE {
            let psi = self.psi().matrix().as_ref().into_ndarray().to_owned();
            let w: Array1<f64> = self.w.clone().iter().collect();
            let pyl = psi.dot(&w);

            let max_d = self.monte_carlo_global_check(&pyl)?;

            if max_d > GLOBAL_OPTIMALITY_THRESHOLD {
                tracing::debug!(
                    "Global optimality check failed: max_D = {:.6} > {:.6}",
                    max_d,
                    GLOBAL_OPTIMALITY_THRESHOLD
                );
                return Ok(false);
            }

            tracing::debug!("Global optimality check passed: max_D = {:.6}", max_d);
        }

        Ok(true)
    }

    /// Monte Carlo estimate of maximum D-criterion over parameter space
    fn monte_carlo_global_check(&mut self, pyl: &Array1<f64>) -> Result<f64> {
        let mut max_d = f64::NEG_INFINITY;

        for _ in 0..GLOBAL_OPTIMALITY_SAMPLES {
            let point: Vec<f64> = self
                .ranges
                .iter()
                .map(|(lo, hi)| self.rng.random_range(*lo..*hi))
                .collect();

            let d_value = self.compute_d_criterion(&point, pyl)?;
            max_d = max_d.max(d_value);
        }

        Ok(max_d)
    }
}

// ============================================================================
// ADAPTIVE SPP OPTIMIZER
// ============================================================================

use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};
use ndarray::Axis;

/// Support Point Optimizer with configurable iteration count
struct SppOptimizerAdaptive<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    sig: &'a ErrorModels,
    pyl: &'a Array1<f64>,
    max_iters: u64,
}

impl<E: Equation> CostFunction for SppOptimizerAdaptive<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, spp: &Self::Param) -> Result<Self::Output, Error> {
        let theta = Array1::from(spp.clone()).insert_axis(Axis(0));

        let psi = pharmsol::prelude::simulator::psi(
            self.equation,
            self.data,
            &theta,
            self.sig,
            false,
            false,
        )?;

        let nsub = psi.nrows() as f64;
        let mut sum = -nsub;
        for (p_i, pyl_i) in psi.iter().zip(self.pyl.iter()) {
            sum += p_i / pyl_i;
        }
        Ok(-sum) // Minimize negative D → Maximize D
    }
}

impl<'a, E: Equation> SppOptimizerAdaptive<'a, E> {
    fn new(
        equation: &'a E,
        data: &'a Data,
        sig: &'a ErrorModels,
        pyl: &'a Array1<f64>,
        max_iters: u64,
    ) -> Self {
        Self {
            equation,
            data,
            sig,
            pyl,
            max_iters,
        }
    }

    fn optimize_point(self, spp: Array1<f64>) -> Result<Array1<f64>, Error> {
        let simplex = create_initial_simplex(&spp.to_vec());
        let tolerance = if self.max_iters > 50 { 1e-4 } else { 1e-2 };
        let max_iters = self.max_iters;

        let solver: NelderMead<Vec<f64>, f64> =
            NelderMead::new(simplex).with_sd_tolerance(tolerance)?;

        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(max_iters))
            .run()?;

        Ok(Array1::from(res.state.best_param.unwrap()))
    }
}

/// Create initial simplex for Nelder-Mead optimization
fn create_initial_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let num_dimensions = initial_point.len();
    let perturbation_percentage = 0.05; // 5% perturbation

    let mut vertices = Vec::new();
    vertices.push(initial_point.to_vec());

    for i in 0..num_dimensions {
        let perturbation = if initial_point[i] == 0.0 {
            0.001
        } else {
            perturbation_percentage * initial_point[i].abs()
        };

        let mut perturbed_point = initial_point.to_vec();
        perturbed_point[i] += perturbation;
        vertices.push(perturbed_point);
    }

    vertices
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_simplex_creation() {
        let point = vec![1.0, 2.0, 3.0];
        let simplex = create_initial_simplex(&point);

        assert_eq!(simplex.len(), 4); // n+1 vertices for n dimensions
        assert_eq!(simplex[0], point); // First vertex is the initial point
    }

    #[test]
    fn test_initial_simplex_with_zero() {
        let point = vec![0.0, 1.0];
        let simplex = create_initial_simplex(&point);

        assert_eq!(simplex.len(), 3);
        // Zero should get special handling
        assert!(simplex[1][0] > 0.0);
    }

    #[test]
    fn test_convergence_window() {
        assert!(CONVERGENCE_WINDOW >= 2);
    }

    #[test]
    fn test_temperature_bounds() {
        assert!(INITIAL_TEMPERATURE > MIN_TEMPERATURE);
        assert!(COOLING_RATE > 0.0 && COOLING_RATE < 1.0);
    }
}
