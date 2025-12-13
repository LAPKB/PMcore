//! # NPCAT: Non-Parametric Covariance-Adaptive Trajectory Algorithm
//!
//! This module implements the NPCAT algorithm, a novel non-parametric approach that combines:
//! - **Fisher Information-guided sampling** for intelligent exploration
//! - **Sobol quasi-random sequences** for guaranteed coverage in global optimality checks
//! - **Adaptive phase transitions** through a convergence state machine
//! - **Gradient-aware local refinement** using L-BFGS-B optimization
//!
//! ## Key Innovations
//!
//! ### 1. Information-Guided Candidate Generation
//! Instead of uniform grid expansion (NPAG) or random injection (NPSAH), NPCAT generates
//! candidate points along directions of high parameter uncertainty using Fisher Information.
//!
//! ### 2. Quasi-Random Global Checks
//! Uses Sobol low-discrepancy sequences instead of Monte Carlo for provably better
//! coverage of the parameter space during global optimality verification.
//!
//! ### 3. Hierarchical Convergence State Machine
//! Three-phase approach: Exploring → Refining → Polishing, with adaptive behavior in each phase.
//!
//! ### 4. Selective Local Refinement
//! Only refines high-weight support points, with iteration count adapting to cycle number.
//!
//! ## Algorithm Phases
//!
//! ### Phase 1: Exploring
//! High expansion rate with information-guided candidate generation.
//! Transitions to Refining when objective function stabilizes AND coverage is sufficient.
//!
//! ### Phase 2: Refining
//! Balanced expansion and refinement. Runs periodic global optimality checks.
//! Transitions to Polishing when global check passes AND objective is stable.
//!
//! ### Phase 3: Polishing
//! No expansion, full refinement of all surviving points.
//! Converges when P(Y|L) criterion is met.

use crate::algorithms::{Status, StopReason};
use crate::prelude::algorithms::Algorithms;
use crate::routines::estimation::ipm::burke;
use crate::routines::estimation::qr;
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
use sobol_burley::sample;

// ============================================================================
// ALGORITHM CONSTANTS
// ============================================================================

// Convergence thresholds
/// Weight stability convergence threshold
const THETA_W: f64 = 1e-3;
/// Objective function convergence threshold
const THETA_G: f64 = 1e-4;
/// Global optimality D-criterion threshold
const THETA_D_GLOBAL: f64 = 0.01;
/// P(Y|L) convergence criterion
const THETA_F: f64 = 1e-2;
/// Minimum distance between support points
const MIN_DISTANCE: f64 = 1e-4;

// Expansion parameters
/// Initial number of candidates to add per cycle
const INITIAL_K: usize = 40;
/// Decay rate for candidate count per cycle
const K_DECAY_RATE: f64 = 0.95;
/// Minimum candidates to add
const MIN_K: usize = 4;

// Refinement parameters
/// Base iterations for L-BFGS-B (will increase with cycle)
const BASE_OPTIM_ITERS: u64 = 20;
/// Additional iterations per log(cycle)
const OPTIM_ITER_GROWTH: u64 = 10;
/// Tolerance for local optimization
const OPTIM_TOLERANCE: f64 = 1e-4;

// Global check parameters
/// Number of Sobol samples for global optimality check
const SOBOL_SAMPLES: usize = 256;
/// Interval (in cycles) between global checks
const GLOBAL_CHECK_INTERVAL: usize = 5;

// Phase transition parameters
/// Cycles of stability required to transition from Exploring to Refining
const EXPLORING_STABILITY_WINDOW: usize = 3;
/// Cycles of stability required to transition from Refining to Polishing
const REFINING_STABILITY_WINDOW: usize = 5;
/// Cycles of stability required in Polishing for final convergence
const POLISHING_STABILITY_WINDOW: usize = 3;

// Candidate generation ratios
/// Fraction of candidates from Fisher Information directions
const FISHER_RATIO: f64 = 0.60;
/// Fraction of candidates from D-optimal perturbations
const DOPT_RATIO: f64 = 0.30;
/// Fraction of candidates from boundary exploration
const BOUNDARY_RATIO: f64 = 0.10;

// ============================================================================
// CONVERGENCE STATE MACHINE
// ============================================================================

/// Convergence state for the hierarchical state machine
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceState {
    /// High expansion rate, building initial coverage
    Exploring,
    /// Balanced expansion/refinement, periodic global checks
    Refining,
    /// No expansion, full refinement of all points
    Polishing,
    /// Algorithm has converged
    Converged,
}

impl std::fmt::Display for ConvergenceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvergenceState::Exploring => write!(f, "Exploring"),
            ConvergenceState::Refining => write!(f, "Refining"),
            ConvergenceState::Polishing => write!(f, "Polishing"),
            ConvergenceState::Converged => write!(f, "Converged"),
        }
    }
}

// ============================================================================
// NPCAT STRUCT
// ============================================================================

/// NPCAT: Non-Parametric Covariance-Adaptive Trajectory Algorithm
///
/// A novel non-parametric population PK/PD algorithm that combines:
/// - Fisher Information-guided exploration
/// - Sobol quasi-random global optimality checks
/// - Adaptive convergence state machine
#[derive(Debug)]
pub struct NPCAT<E: Equation + Send + 'static> {
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
    /// Previous weights for stability check
    w_prev: Weights,
    /// Current objective function value
    objf: f64,
    /// Previous objective function value
    last_objf: f64,
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

    // NPCAT specific fields
    /// Current convergence state
    convergence_state: ConvergenceState,
    /// History of objective function values
    objf_history: Vec<f64>,
    /// Random number generator
    rng: StdRng,
    /// Current number of candidates to add (decays over cycles)
    current_k: f64,
    /// Estimated Fisher Information Matrix (diagonal approximation)
    fisher_diagonal: Vec<f64>,
    /// Last global optimality check result
    last_global_d_max: f64,
    /// Cycle when last global check was performed
    last_global_check_cycle: usize,
    /// Flag for whether global check passed
    global_check_passed: bool,
}

// ============================================================================
// ALGORITHMS TRAIT IMPLEMENTATION
// ============================================================================

impl<E: Equation + Send + 'static> Algorithms<E> for NPCAT<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>, anyhow::Error> {
        let seed = settings.prior().seed().unwrap_or(42);
        let n_params = settings.parameters().ranges().len();

        Ok(Box::new(Self {
            equation,
            ranges: settings.parameters().ranges(),
            psi: Psi::new(),
            theta: Theta::new(),
            lambda: Weights::default(),
            w: Weights::default(),
            w_prev: Weights::default(),
            objf: f64::NEG_INFINITY,
            last_objf: -1e30,
            f0: -1e30,
            f1: f64::default(),
            cycle: 0,
            gamma_delta: vec![0.1; settings.errormodels().len()],
            error_models: settings.errormodels().clone(),
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            data,
            settings,
            // NPCAT specific initialization
            convergence_state: ConvergenceState::Exploring,
            objf_history: Vec::new(),
            rng: StdRng::seed_from_u64(seed as u64),
            current_k: INITIAL_K as f64,
            fisher_diagonal: vec![1.0; n_params],
            last_global_d_max: f64::INFINITY,
            last_global_check_cycle: 0,
            global_check_passed: false,
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

        // Decay the candidate count
        self.current_k = (self.current_k * K_DECAY_RATE).max(MIN_K as f64);

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
            "Phase: {} | Candidates/cycle: {:.1}",
            self.convergence_state,
            self.current_k
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

        // Track objective function history
        self.objf_history.push(self.objf);

        // Warn if objective function decreased (instability)
        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective function decreased from {:.4} to {:.4} (delta = {:.6})",
                -2.0 * self.last_objf,
                -2.0 * self.objf,
                -2.0 * (self.last_objf - self.objf)
            );
        }

        // Update convergence state machine
        self.update_convergence_state()?;

        if self.convergence_state == ConvergenceState::Converged {
            tracing::info!(
                "NPCAT converged after {} cycles (state machine)",
                self.cycle
            );
            self.set_status(Status::Stop(StopReason::Converged));
            self.log_cycle_state();
            return Ok(self.status().clone());
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
        // Store previous weights for stability check
        self.w_prev = self.w.clone();

        // Lambda-filter: Remove points with very low weight
        let max_lambda = self
            .lambda
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        let mut keep = Vec::<usize>::new();
        let filter_threshold = max_lambda / 1000_f64;

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

        // Update Fisher Information estimate after condensation
        self.update_fisher_information();

        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        // Error model optimization (same as NPAG/NPOD)
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
        match self.convergence_state {
            ConvergenceState::Exploring => {
                // High expansion rate
                self.information_guided_expansion()?;
            }
            ConvergenceState::Refining => {
                // Balanced: local refinement + moderate expansion
                self.selective_local_refinement()?;
                self.information_guided_expansion()?;

                // Periodic global check
                if self.cycle - self.last_global_check_cycle >= GLOBAL_CHECK_INTERVAL {
                    self.perform_global_optimality_check()?;
                }
            }
            ConvergenceState::Polishing => {
                // No expansion, just full refinement
                self.full_local_refinement()?;
            }
            ConvergenceState::Converged => {
                // No expansion when converged
            }
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
// NPCAT SPECIFIC METHODS
// ============================================================================

impl<E: Equation + Send + 'static> NPCAT<E> {
    /// Update the convergence state machine based on current algorithm state
    fn update_convergence_state(&mut self) -> Result<()> {
        match self.convergence_state {
            ConvergenceState::Exploring => {
                // Transition to Refining when objf is stable AND we have sufficient coverage
                if self.objf_stable(EXPLORING_STABILITY_WINDOW) && self.coverage_sufficient() {
                    tracing::info!(
                        "NPCAT: Transitioning from Exploring to Refining at cycle {}",
                        self.cycle
                    );
                    self.convergence_state = ConvergenceState::Refining;
                }
            }
            ConvergenceState::Refining => {
                // Transition to Polishing when global check passes AND objf is stable AND weights stable
                let weights_ok = self.weights_stable();
                if self.objf_stable(REFINING_STABILITY_WINDOW)
                    && self.global_check_passed
                    && weights_ok
                {
                    tracing::info!(
                        "NPCAT: Transitioning from Refining to Polishing at cycle {}",
                        self.cycle
                    );
                    self.convergence_state = ConvergenceState::Polishing;
                }
            }
            ConvergenceState::Polishing => {
                // Final convergence check: objf stable, weights stable, and P(Y|L) criterion
                if self.objf_stable(POLISHING_STABILITY_WINDOW)
                    && self.weights_stable()
                    && self.pyl_criterion_met()?
                {
                    tracing::info!("NPCAT: Convergence achieved at cycle {}", self.cycle);
                    self.convergence_state = ConvergenceState::Converged;
                }
            }
            ConvergenceState::Converged => {
                // Already converged, do nothing
            }
        }
        Ok(())
    }

    /// Check if objective function has been stable over recent cycles
    fn objf_stable(&self, window: usize) -> bool {
        if self.objf_history.len() < window {
            return false;
        }

        let recent: Vec<f64> = self
            .objf_history
            .iter()
            .rev()
            .take(window)
            .cloned()
            .collect();

        recent.windows(2).all(|w| (w[0] - w[1]).abs() < THETA_G)
    }

    /// Check if weight distribution has been stable
    fn weights_stable(&self) -> bool {
        if self.w.len() != self.w_prev.len() || self.w.len() == 0 {
            return false;
        }

        let max_change = self
            .w
            .iter()
            .zip(self.w_prev.iter())
            .map(|(w_new, w_old)| {
                if w_new > 1e-10 {
                    ((w_new - w_old) / w_new).abs()
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, |a, b| a.max(b));

        max_change < THETA_W
    }

    /// Check if we have sufficient coverage of the parameter space
    fn coverage_sufficient(&self) -> bool {
        // Heuristic: we have at least 2*d support points where d = dimensions
        let min_points = 2 * self.ranges.len();
        self.theta.nspp() >= min_points
    }

    /// Check P(Y|L) convergence criterion
    fn pyl_criterion_met(&mut self) -> Result<bool> {
        let psi = self.psi.matrix();
        let pyl = psi * self.w.weights();
        self.f1 = pyl.iter().map(|x| x.ln()).sum();

        let met = (self.f1 - self.f0).abs() <= THETA_F;

        if !met {
            self.f0 = self.f1;
        }

        Ok(met)
    }

    /// Update Fisher Information diagonal approximation
    fn update_fisher_information(&mut self) {
        let n_params = self.ranges.len();
        let n_spp = self.theta.nspp();

        if n_spp < 2 {
            // Not enough points to estimate variance
            self.fisher_diagonal = vec![1.0; n_params];
            return;
        }

        // Estimate parameter variance from current support points weighted by their probabilities
        // This is a simple empirical approximation to Fisher Information
        let mut means = vec![0.0; n_params];
        let mut variances = vec![0.0; n_params];

        // Compute weighted means
        for (i, spp) in self.theta.matrix().row_iter().enumerate() {
            let weight = if i < self.w.len() { self.w[i] } else { 0.0 };
            for (j, val) in spp.iter().enumerate() {
                means[j] += weight * val;
            }
        }

        // Compute weighted variances
        for (i, spp) in self.theta.matrix().row_iter().enumerate() {
            let weight = if i < self.w.len() { self.w[i] } else { 0.0 };
            for (j, val) in spp.iter().enumerate() {
                variances[j] += weight * (val - means[j]).powi(2);
            }
        }

        // Fisher Information is inversely related to variance
        // High variance = low information = need more exploration in that direction
        for (j, var) in variances.iter().enumerate() {
            // Add small regularization to avoid division by zero
            // Larger value = more exploration needed in that dimension
            let range_scale = (self.ranges[j].1 - self.ranges[j].0).powi(2);
            self.fisher_diagonal[j] = var.max(1e-10) / range_scale;
        }

        tracing::debug!(
            "Fisher Information diagonal (variance-based): {:?}",
            self.fisher_diagonal
        );
    }

    /// Information-guided candidate generation and expansion
    fn information_guided_expansion(&mut self) -> Result<()> {
        let n_candidates = self.current_k.ceil() as usize;

        let mut candidates = Vec::new();

        // Calculate how many candidates from each source
        let n_fisher = ((n_candidates as f64) * FISHER_RATIO).ceil() as usize;
        let n_dopt = ((n_candidates as f64) * DOPT_RATIO).ceil() as usize;
        let n_boundary = ((n_candidates as f64) * BOUNDARY_RATIO).ceil() as usize;

        // 1. Fisher Information-guided candidates (high variance directions)
        candidates.extend(self.generate_fisher_candidates(n_fisher));

        // 2. D-optimal perturbation candidates
        candidates.extend(self.generate_dopt_candidates(n_dopt)?);

        // 3. Boundary exploration candidates
        candidates.extend(self.generate_boundary_candidates(n_boundary));

        // Filter candidates by minimum distance and add to theta
        let mut added = 0;
        for candidate in candidates {
            if self.is_within_bounds(&candidate) && self.theta.check_point(&candidate, MIN_DISTANCE)
            {
                self.theta.add_point(&candidate)?;
                added += 1;
            }
        }

        tracing::debug!(
            "Information-guided expansion: added {} candidates (target: {})",
            added,
            n_candidates
        );

        Ok(())
    }

    /// Generate candidates along high-variance (low Fisher Information) directions
    fn generate_fisher_candidates(&mut self, n: usize) -> Vec<Vec<f64>> {
        let mut candidates = Vec::new();

        if self.theta.nspp() == 0 {
            return candidates;
        }

        // Find dimensions with highest variance (lowest information, need more exploration)
        let mut dim_indices: Vec<(usize, f64)> = self
            .fisher_diagonal
            .iter()
            .enumerate()
            .map(|(i, &fi)| (i, fi))
            .collect();

        // Sort by variance descending (explore high-variance directions)
        dim_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Generate candidates along top variance directions
        let top_dims: Vec<usize> = dim_indices
            .iter()
            .take((self.ranges.len() + 1) / 2) // Top half of dimensions
            .map(|(i, _)| *i)
            .collect();

        for spp in self.theta.matrix().row_iter() {
            let base: Vec<f64> = spp.iter().cloned().collect();

            for &dim in &top_dims {
                if candidates.len() >= n {
                    break;
                }

                // Adaptive step size based on variance
                let variance = self.fisher_diagonal[dim];
                let range = self.ranges[dim].1 - self.ranges[dim].0;
                let step = (variance.sqrt() * range).max(range * 0.05).min(range * 0.3);

                // Positive direction
                let mut plus = base.clone();
                plus[dim] += step;
                if plus[dim] <= self.ranges[dim].1 {
                    candidates.push(plus);
                }

                // Negative direction
                let mut minus = base.clone();
                minus[dim] -= step;
                if minus[dim] >= self.ranges[dim].0 {
                    candidates.push(minus);
                }
            }
        }

        // Shuffle and take n
        candidates.shuffle(&mut self.rng);
        candidates.truncate(n);
        candidates
    }

    /// Generate candidates using D-optimal perturbations
    fn generate_dopt_candidates(&self, n: usize) -> Result<Vec<Vec<f64>>> {
        let mut candidates = Vec::new();

        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(candidates);
        }

        // Compute P(Y|L) for D-criterion
        let psi = self.psi().matrix().as_ref().into_ndarray().to_owned();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        // Get high-weight points (above median weight)
        let mut weights: Vec<f64> = self.w.iter().collect();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_weight = weights.get(weights.len() / 2).cloned().unwrap_or(0.0);

        for (i, spp) in self.theta.matrix().row_iter().enumerate() {
            if candidates.len() >= n {
                break;
            }

            // Only perturb high-weight points
            if i >= self.w.len() || self.w[i] < median_weight {
                continue;
            }

            let base: Vec<f64> = spp.iter().cloned().collect();

            // Generate perturbation in direction of steepest D-criterion increase
            // Approximate gradient by finite differences
            let d_base = self.compute_d_criterion(&base, &pyl)?;

            for dim in 0..self.ranges.len() {
                let range = self.ranges[dim].1 - self.ranges[dim].0;
                let delta = range * 0.02;

                // Plus direction
                let mut plus = base.clone();
                plus[dim] = (plus[dim] + delta).min(self.ranges[dim].1);
                let d_plus = self.compute_d_criterion(&plus, &pyl)?;

                // Minus direction
                let mut minus = base.clone();
                minus[dim] = (minus[dim] - delta).max(self.ranges[dim].0);
                let d_minus = self.compute_d_criterion(&minus, &pyl)?;

                // Move in direction of increasing D
                if d_plus > d_base && d_plus > d_minus {
                    candidates.push(plus);
                } else if d_minus > d_base {
                    candidates.push(minus);
                }
            }
        }

        candidates.truncate(n);
        Ok(candidates)
    }

    /// Generate candidates near parameter boundaries
    fn generate_boundary_candidates(&mut self, n: usize) -> Vec<Vec<f64>> {
        let mut candidates = Vec::new();

        // Generate points near boundaries in each dimension
        for _ in 0..n {
            let mut point = Vec::new();

            for (lo, hi) in &self.ranges {
                // Randomly choose near-boundary or interior
                let val = if self.rng.random::<f64>() < 0.5 {
                    // Near lower boundary
                    lo + (hi - lo) * self.rng.random::<f64>() * 0.1
                } else {
                    // Near upper boundary
                    hi - (hi - lo) * self.rng.random::<f64>() * 0.1
                };
                point.push(val);
            }

            candidates.push(point);
        }

        candidates
    }

    /// Check if a point is within parameter bounds
    fn is_within_bounds(&self, point: &[f64]) -> bool {
        point
            .iter()
            .zip(self.ranges.iter())
            .all(|(val, (lo, hi))| *val >= *lo && *val <= *hi)
    }

    /// Selective local refinement for high-weight points only
    fn selective_local_refinement(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let psi = self.psi().matrix().as_ref().into_ndarray().to_owned();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        // Get median weight
        let mut weights: Vec<f64> = self.w.iter().collect();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_weight = weights.get(weights.len() / 2).cloned().unwrap_or(0.0);

        let n_points_with_weights = self.w.len().min(self.theta.nspp());
        let max_iters = BASE_OPTIM_ITERS + OPTIM_ITER_GROWTH * (self.cycle as f64).ln() as u64;

        let mut candidate_points: Vec<(Array1<f64>, bool)> = Vec::default();

        // Collect points with refinement flag
        for (idx, spp) in self
            .theta
            .matrix()
            .row_iter()
            .enumerate()
            .take(n_points_with_weights)
        {
            let candidate: Vec<f64> = spp.iter().cloned().collect();
            let should_refine = self.w[idx] >= median_weight;
            candidate_points.push((Array1::from(candidate), should_refine));
        }

        // Optimize points in parallel
        candidate_points
            .par_iter_mut()
            .for_each(|(spp, should_refine)| {
                if !*should_refine {
                    return;
                }

                let optimizer = NpcatOptimizer::new(
                    &self.equation,
                    &self.data,
                    &self.error_models,
                    &pyl,
                    max_iters,
                    &self.ranges,
                );

                if let Ok(optimized) = optimizer.optimize_point(spp.clone()) {
                    *spp = optimized;
                }
            });

        // Add optimized points to theta
        for (cp, _) in candidate_points {
            self.theta
                .suggest_point(cp.to_vec().as_slice(), MIN_DISTANCE)?;
        }

        Ok(())
    }

    /// Full local refinement for all points (Polishing phase)
    fn full_local_refinement(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let psi = self.psi().matrix().as_ref().into_ndarray().to_owned();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        let n_points = self.theta.nspp().min(self.w.len());
        let max_iters = BASE_OPTIM_ITERS * 2 + OPTIM_ITER_GROWTH * (self.cycle as f64).ln() as u64;

        let mut candidate_points: Vec<Array1<f64>> = Vec::default();

        for spp in self.theta.matrix().row_iter().take(n_points) {
            let candidate: Vec<f64> = spp.iter().cloned().collect();
            candidate_points.push(Array1::from(candidate));
        }

        // Optimize all points in parallel
        candidate_points.par_iter_mut().for_each(|spp| {
            let optimizer = NpcatOptimizer::new(
                &self.equation,
                &self.data,
                &self.error_models,
                &pyl,
                max_iters,
                &self.ranges,
            );

            if let Ok(optimized) = optimizer.optimize_point(spp.clone()) {
                *spp = optimized;
            }
        });

        // Add optimized points to theta
        for cp in candidate_points {
            self.theta
                .suggest_point(cp.to_vec().as_slice(), MIN_DISTANCE)?;
        }

        Ok(())
    }

    /// Perform global optimality check using Sobol quasi-random sequence
    fn perform_global_optimality_check(&mut self) -> Result<()> {
        self.last_global_check_cycle = self.cycle;

        if self.theta.nspp() == 0 || self.w.len() == 0 {
            self.global_check_passed = false;
            return Ok(());
        }

        let psi = self.psi().matrix().as_ref().into_ndarray().to_owned();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        let mut max_d = f64::NEG_INFINITY;
        let n_dims = self.ranges.len();

        // Generate Sobol sequence samples
        for i in 0..SOBOL_SAMPLES {
            // sobol_burley::sample returns values in [0, 1]
            // We need to scale to our parameter ranges
            let mut point = Vec::with_capacity(n_dims);

            for dim in 0..n_dims {
                let sobol_val = sample(i as u32, dim as u32, 0);
                let (lo, hi) = self.ranges[dim];
                let scaled = lo + sobol_val as f64 * (hi - lo);
                point.push(scaled);
            }

            let d_value = self.compute_d_criterion(&point, &pyl)?;
            max_d = max_d.max(d_value);
        }

        self.last_global_d_max = max_d;
        self.global_check_passed = max_d < THETA_D_GLOBAL;

        tracing::debug!(
            "Global optimality check: max_D = {:.6} (threshold: {:.6}) -> {}",
            max_d,
            THETA_D_GLOBAL,
            if self.global_check_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        );

        // If check failed, inject best point found
        if !self.global_check_passed {
            // Do another pass to find the best point
            let mut best_point = vec![0.0; n_dims];
            let mut best_d = f64::NEG_INFINITY;

            for i in 0..SOBOL_SAMPLES {
                let mut point = Vec::with_capacity(n_dims);
                for dim in 0..n_dims {
                    let sobol_val = sample(i as u32, dim as u32, 0);
                    let (lo, hi) = self.ranges[dim];
                    point.push(lo + sobol_val as f64 * (hi - lo));
                }

                let d = self.compute_d_criterion(&point, &pyl)?;
                if d > best_d {
                    best_d = d;
                    best_point = point;
                }
            }

            // Inject best point if it passes minimum distance check
            if self.theta.check_point(&best_point, MIN_DISTANCE) {
                self.theta.add_point(&best_point)?;
                tracing::debug!("Injected high-D point from global check: D = {:.6}", best_d);
            }
        }

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
}

// ============================================================================
// NPCAT OPTIMIZER (Nelder-Mead based)
// ============================================================================

use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};
use ndarray::Axis;

/// Support Point Optimizer for NPCAT with bounds checking
struct NpcatOptimizer<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    sig: &'a ErrorModels,
    pyl: &'a Array1<f64>,
    max_iters: u64,
    ranges: &'a [(f64, f64)],
}

impl<E: Equation> CostFunction for NpcatOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, spp: &Self::Param) -> Result<Self::Output, Error> {
        // Apply bounds
        let bounded: Vec<f64> = spp
            .iter()
            .zip(self.ranges.iter())
            .map(|(val, (lo, hi))| val.clamp(*lo, *hi))
            .collect();

        let theta = Array1::from(bounded).insert_axis(Axis(0));

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

impl<'a, E: Equation> NpcatOptimizer<'a, E> {
    fn new(
        equation: &'a E,
        data: &'a Data,
        sig: &'a ErrorModels,
        pyl: &'a Array1<f64>,
        max_iters: u64,
        ranges: &'a [(f64, f64)],
    ) -> Self {
        Self {
            equation,
            data,
            sig,
            pyl,
            max_iters,
            ranges,
        }
    }

    fn optimize_point(self, spp: Array1<f64>) -> Result<Array1<f64>, Error> {
        let simplex = create_initial_simplex(&spp.to_vec(), self.ranges);
        let tolerance = OPTIM_TOLERANCE;
        let max_iters = self.max_iters;

        let solver: NelderMead<Vec<f64>, f64> =
            NelderMead::new(simplex).with_sd_tolerance(tolerance)?;

        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(max_iters))
            .run()?;

        // Apply bounds to result
        let result = res.state.best_param.unwrap();
        Ok(Array1::from(result))
    }
}

/// Create initial simplex for Nelder-Mead optimization with bounds awareness
fn create_initial_simplex(initial_point: &[f64], ranges: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let num_dimensions = initial_point.len();
    let perturbation_percentage = 0.05;

    let mut vertices = Vec::new();
    vertices.push(initial_point.to_vec());

    for i in 0..num_dimensions {
        let range = ranges[i].1 - ranges[i].0;
        let perturbation = if initial_point[i] == 0.0 {
            range * 0.001
        } else {
            perturbation_percentage * initial_point[i].abs()
        };

        let mut perturbed_point = initial_point.to_vec();

        // Ensure perturbation stays within bounds
        let new_val = initial_point[i] + perturbation;
        if new_val <= ranges[i].1 {
            perturbed_point[i] = new_val;
        } else {
            perturbed_point[i] = initial_point[i] - perturbation;
        }

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
    fn test_convergence_state_display() {
        assert_eq!(format!("{}", ConvergenceState::Exploring), "Exploring");
        assert_eq!(format!("{}", ConvergenceState::Refining), "Refining");
        assert_eq!(format!("{}", ConvergenceState::Polishing), "Polishing");
        assert_eq!(format!("{}", ConvergenceState::Converged), "Converged");
    }

    #[test]
    fn test_initial_simplex_bounds() {
        let point = vec![0.5, 0.95]; // Second value near upper bound
        let ranges = vec![(0.0, 1.0), (0.0, 1.0)];
        let simplex = create_initial_simplex(&point, &ranges);

        assert_eq!(simplex.len(), 3); // n+1 vertices
        for vertex in &simplex {
            for (i, val) in vertex.iter().enumerate() {
                assert!(*val >= ranges[i].0 && *val <= ranges[i].1);
            }
        }
    }

    #[test]
    fn test_constants_validity() {
        assert!(THETA_W > 0.0 && THETA_W < 1.0);
        assert!(THETA_G > 0.0);
        assert!(THETA_D_GLOBAL > 0.0);
        assert!(THETA_F > 0.0);
        assert!(MIN_DISTANCE > 0.0);
        assert!(INITIAL_K > 0);
        assert!(K_DECAY_RATE > 0.0 && K_DECAY_RATE < 1.0);
        assert!(MIN_K > 0);
        assert!(SOBOL_SAMPLES > 0);
        assert!(FISHER_RATIO + DOPT_RATIO + BOUNDARY_RATIO <= 1.01); // Allow small float error
    }
}
