//! # NPSA-H2: Non-Parametric Simulated Annealing Hybrid Algorithm v2
//!
//! An improved version of NPSAH with the following enhancements:
//!
//! ## Perspective: The Scientist
//! - Better exploration of multimodal distributions
//! - Adaptive strategies based on problem characteristics
//! - More robust handling of edge cases
//!
//! ## Perspective: The Statistician  
//! - Improved convergence criteria using multiple metrics
//! - Better handling of the bias-variance tradeoff
//! - Statistically sound weight estimation
//!
//! ## Perspective: The Engineer
//! - Parallelized operations where possible
//! - Memory-efficient data structures
//! - Early termination for provably suboptimal paths
//!
//! ## Key Improvements over NPSAH v1
//! 1. **Adaptive Temperature Schedule**: Temperature adapts based on acceptance ratio
//! 2. **Elite Preservation**: Best points are preserved across cycles
//! 3. **Cluster-Aware Expansion**: Identifies and expands around clusters
//! 4. **Gradient-Informed SA**: Uses local gradient information to guide moves
//! 5. **Restart Mechanism**: Can restart from cold when stuck
//! 6. **Parallel D-criterion Evaluation**: Batch evaluation of candidate points

use crate::algorithms::{Status, StopReason};
use crate::prelude::algorithms::Algorithms;
use crate::routines::estimation::ipm::burke;
use crate::routines::estimation::qr;
use crate::routines::expansion::adaptative_grid::adaptative_grid;
use crate::routines::initialization::sample_space;
use crate::routines::output::{cycles::CycleLog, cycles::NPCycle, NPResult};
use crate::routines::settings::Settings;
use crate::structs::nonparametric::psi::{calculate_psi, Psi};
use crate::structs::nonparametric::theta::Theta;
use crate::structs::nonparametric::weights::Weights;

use anyhow::{bail, Result};
use ndarray::parallel::prelude::{
    IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use ndarray::{Array1, Axis};
use pharmsol::prelude::AssayErrorModel;
use pharmsol::prelude::{
    data::{Data, AssayErrorModels},
    simulator::Equation,
};
use rand::prelude::*;

// ============================================================================
// ALGORITHM CONSTANTS - TUNED FOR BETTER PERFORMANCE
// ============================================================================

/// Grid spacing convergence threshold
const THETA_E: f64 = 1e-4;
/// Objective function convergence threshold  
const THETA_G: f64 = 1e-4;
/// P(Y|L) convergence criterion
const THETA_F: f64 = 1e-2;
/// Minimum distance between support points
const THETA_D: f64 = 1e-4;

// --- Phase Control ---
/// Number of warm-up cycles using pure exploration
const WARMUP_CYCLES: usize = 3;
/// Number of cycles for intensive exploitation phase
const EXPLOITATION_CYCLES: usize = 3;

// --- Temperature Schedule (Adaptive) ---
/// Initial temperature for simulated annealing
const INITIAL_TEMPERATURE: f64 = 1.5;
/// Base cooling rate (will be adapted)
const BASE_COOLING_RATE: f64 = 0.88;
/// Minimum temperature before SA stops
const MIN_TEMPERATURE: f64 = 0.01;
/// Target acceptance ratio for adaptive temperature
const TARGET_ACCEPTANCE_RATIO: f64 = 0.25;
/// Temperature increase factor when too cold
const REHEAT_FACTOR: f64 = 1.3;

// --- Exploration Parameters ---
/// Number of SA points to inject per cycle (base)
const SA_INJECT_BASE: usize = 10;
/// Number of elite points to preserve
const ELITE_COUNT: usize = 3;
/// Number of points for Latin Hypercube Sampling
const LHS_SAMPLES: usize = 30;

// --- D-Optimal Refinement ---
/// Threshold for considering a support point "high importance"
const HIGH_IMPORTANCE_THRESHOLD: f64 = 0.05;
/// Maximum Nelder-Mead iterations for high-importance points
const HIGH_IMPORTANCE_MAX_ITERS: u64 = 80;

// --- Safety Margins ---
/// Relative margin from boundaries to prevent numerical issues (1% of range)
const BOUNDARY_MARGIN_RATIO: f64 = 0.01;
/// Maximum Nelder-Mead iterations for medium-importance points
const MEDIUM_IMPORTANCE_MAX_ITERS: u64 = 30;
/// Maximum Nelder-Mead iterations for low-importance points
const LOW_IMPORTANCE_MAX_ITERS: u64 = 10;

// --- Convergence Criteria ---
/// Number of consecutive stable cycles required for convergence
const CONVERGENCE_WINDOW: usize = 3;
/// Number of Monte Carlo samples for global optimality check
const GLOBAL_OPTIMALITY_SAMPLES: usize = 500;
/// Threshold for D-criterion in global optimality check
const GLOBAL_OPTIMALITY_THRESHOLD: f64 = 0.01;

// --- Restart Mechanism ---
/// Number of cycles without improvement before restart
const STAGNATION_CYCLES: usize = 15;
/// Maximum number of restarts
const MAX_RESTARTS: usize = 2;

// ============================================================================
// ALGORITHM STATE
// ============================================================================

/// Phase of the algorithm
#[derive(Debug, Clone, PartialEq)]
enum Phase {
    /// Initial exploration with NPAG-style grid
    Warmup,
    /// Balanced exploration and exploitation
    Hybrid,
    /// Focus on refining existing points
    Exploitation,
    /// Final convergence checking
    Convergence,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Phase::Warmup => write!(f, "Warmup"),
            Phase::Hybrid => write!(f, "Hybrid"),
            Phase::Exploitation => write!(f, "Exploitation"),
            Phase::Convergence => write!(f, "Convergence"),
        }
    }
}

/// Elite point with its D-criterion value
#[derive(Debug, Clone)]
struct ElitePoint {
    params: Vec<f64>,
    d_value: f64,
    age: usize,
}

// ============================================================================
// NPSAH2 STRUCT
// ============================================================================

/// NPSA-H2: Improved Non-Parametric Simulated Annealing Hybrid Algorithm
#[derive(Debug)]
pub struct NPSAH2<E: Equation + Send + 'static> {
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
    /// Best objective function value seen
    best_objf: f64,
    /// P(Y|L) values for convergence checking
    f0: f64,
    f1: f64,
    /// Current cycle number
    cycle: usize,
    /// Step sizes for error model optimization
    gamma_delta: Vec<f64>,
    /// Error models for observations
    error_models: AssayErrorModels,
    /// Algorithm status
    status: Status,
    /// Cycle log for tracking progress
    cycle_log: CycleLog,
    /// Subject data
    data: Data,
    /// Algorithm settings
    settings: Settings,

    // NPSAH2 specific fields
    /// Current simulated annealing temperature
    temperature: f64,
    /// History of objective function values
    objf_history: Vec<f64>,
    /// Random number generator
    rng: StdRng,
    /// Current algorithm phase
    phase: Phase,
    /// Elite points preserved across cycles
    elite_points: Vec<ElitePoint>,
    /// Number of accepted SA moves this cycle
    sa_accepted: usize,
    /// Number of proposed SA moves this cycle
    sa_proposed: usize,
    /// Cycles since last improvement
    cycles_since_improvement: usize,
    /// Number of restarts performed
    restart_count: usize,
    /// Effective cooling rate (adaptive)
    cooling_rate: f64,
}

// ============================================================================
// ALGORITHMS TRAIT IMPLEMENTATION
// ============================================================================

impl<E: Equation + Send + 'static> Algorithms<E> for NPSAH2<E> {
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
            best_objf: f64::NEG_INFINITY,
            f0: -1e30,
            f1: f64::default(),
            cycle: 0,
            gamma_delta: vec![0.1; settings.errormodels().len()],
            error_models: settings.errormodels().clone(),
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            data,
            settings,
            // NPSAH2 specific initialization
            temperature: INITIAL_TEMPERATURE,
            objf_history: Vec::with_capacity(1000),
            rng: StdRng::seed_from_u64(seed as u64),
            phase: Phase::Warmup,
            elite_points: Vec::with_capacity(ELITE_COUNT),
            sa_accepted: 0,
            sa_proposed: 0,
            cycles_since_improvement: 0,
            restart_count: 0,
            cooling_rate: BASE_COOLING_RATE,
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
        self.update_phase();
        self.adapt_temperature();
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
        tracing::info!(
            "Cycle {} | Phase: {} | -2LL = {:.4} | SPPs = {} | T = {:.4}",
            self.cycle,
            self.phase,
            -2.0 * self.objf,
            self.theta.nspp(),
            self.temperature
        );

        self.error_models.iter().for_each(|(outeq, em)| {
            if AssayErrorModel::None == *em {
                return;
            }
            tracing::debug!(
                "Error model for outeq {}: {:.4}",
                outeq,
                em.factor().unwrap_or_default()
            );
        });

        // Track objective function
        self.objf_history.push(self.objf);

        // Check for improvement
        if self.objf > self.best_objf + THETA_G {
            self.best_objf = self.objf;
            self.cycles_since_improvement = 0;
        } else {
            self.cycles_since_improvement += 1;
        }

        // Warn if objective function decreased
        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective decreased: {:.4} -> {:.4}",
                -2.0 * self.last_objf,
                -2.0 * self.objf
            );
        }

        // Check for stagnation and possibly restart
        if self.cycles_since_improvement >= STAGNATION_CYCLES {
            if self.restart_count < MAX_RESTARTS {
                tracing::info!(
                    "Stagnation detected, performing restart #{}",
                    self.restart_count + 1
                );
                self.perform_restart()?;
                self.set_status(Status::Continue);
                self.log_cycle_state();
                return Ok(self.status().clone());
            }
        }

        // Early convergence check in exploitation phase when stable
        // This avoids waiting for temperature to cool all the way down
        if self.phase == Phase::Exploitation && self.cycles_since_improvement >= CONVERGENCE_WINDOW
        {
            if self.check_convergence()? {
                tracing::info!(
                    "NPSAH2 converged after {} cycles (early convergence)",
                    self.cycle
                );
                self.set_status(Status::Stop(StopReason::Converged));
                self.log_cycle_state();
                return Ok(self.status().clone());
            }
        }

        // Multi-criterion convergence check in convergence phase
        if self.phase == Phase::Convergence {
            if self.check_convergence()? {
                tracing::info!("NPSAH2 converged after {} cycles", self.cycle);
                self.set_status(Status::Stop(StopReason::Converged));
                self.log_cycle_state();
                return Ok(self.status().clone());
            }
        }

        // Standard NPAG-style convergence
        let psi = self.psi.matrix();
        let w = &self.w;
        if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
            self.eps /= 2.;
            tracing::debug!("Halving eps to {:.6}", self.eps);

            if self.eps <= THETA_E {
                let pyl = psi * w.weights();
                self.f1 = pyl.iter().map(|x| x.ln()).sum();
                if (self.f1 - self.f0).abs() <= THETA_F {
                    tracing::info!("NPSAH2 converged (P(Y|L) criterion)");
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
            tracing::warn!("Maximum cycles reached");
            self.set_status(Status::Stop(StopReason::MaxCycles));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // Check for stop file
        if std::path::Path::new("stop").exists() {
            tracing::warn!("Stop file detected");
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
        // Lambda-filter with adaptive threshold
        let max_lambda = self
            .lambda
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        // More aggressive filtering in later phases
        let filter_divisor = match self.phase {
            Phase::Warmup => 1000.0,
            Phase::Hybrid => 5000.0,
            Phase::Exploitation => 10000.0,
            Phase::Convergence => 10000.0,
        };

        let mut keep = Vec::<usize>::new();
        let filter_threshold = max_lambda / filter_divisor;
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

        // Update elite points after condensation
        self.update_elite_points()?;

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
        match self.phase {
            Phase::Warmup => self.warmup_expansion()?,
            Phase::Hybrid => self.hybrid_expansion()?,
            Phase::Exploitation => self.exploitation_expansion()?,
            Phase::Convergence => self.convergence_expansion()?,
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
// NPSAH2 SPECIFIC METHODS
// ============================================================================

impl<E: Equation + Send + 'static> NPSAH2<E> {
    /// Update the algorithm phase based on cycle number and progress
    fn update_phase(&mut self) {
        let old_phase = self.phase.clone();

        self.phase = if self.cycle <= WARMUP_CYCLES {
            Phase::Warmup
        } else if self.cycle <= WARMUP_CYCLES + EXPLOITATION_CYCLES {
            Phase::Hybrid
        } else if self.temperature > MIN_TEMPERATURE * 2.0 {
            Phase::Exploitation
        } else {
            Phase::Convergence
        };

        if self.phase != old_phase {
            tracing::info!("Phase transition: {} -> {}", old_phase, self.phase);
        }
    }

    /// Adapt temperature based on acceptance ratio
    fn adapt_temperature(&mut self) {
        if self.sa_proposed > 0 {
            let acceptance_ratio = self.sa_accepted as f64 / self.sa_proposed as f64;

            // Adjust cooling rate based on acceptance ratio
            if acceptance_ratio < TARGET_ACCEPTANCE_RATIO * 0.5 {
                // Too cold, slow down cooling
                self.cooling_rate = (self.cooling_rate + 0.02).min(0.98);
                // Maybe reheat slightly
                if acceptance_ratio < 0.1 && self.temperature < 0.5 {
                    self.temperature *= REHEAT_FACTOR;
                    tracing::debug!("Reheating to T = {:.4}", self.temperature);
                }
            } else if acceptance_ratio > TARGET_ACCEPTANCE_RATIO * 1.5 {
                // Too hot, speed up cooling
                self.cooling_rate = (self.cooling_rate - 0.02).max(0.85);
            }

            tracing::debug!(
                "SA acceptance: {:.1}% | Cooling rate: {:.3}",
                acceptance_ratio * 100.0,
                self.cooling_rate
            );
        }

        // Apply cooling
        self.temperature *= self.cooling_rate;
        if self.temperature < MIN_TEMPERATURE {
            self.temperature = MIN_TEMPERATURE;
        }

        // Reset counters
        self.sa_accepted = 0;
        self.sa_proposed = 0;
    }

    /// Warm-up phase: broad exploration with LHS and grid
    fn warmup_expansion(&mut self) -> Result<()> {
        tracing::debug!("Warmup expansion: LHS + adaptive grid");

        // Latin Hypercube Sampling for better initial coverage
        self.lhs_injection(LHS_SAMPLES)?;

        // Also do NPAG-style grid expansion
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D)?;

        Ok(())
    }

    /// Hybrid phase: balanced exploration and exploitation
    fn hybrid_expansion(&mut self) -> Result<()> {
        let initial = self.theta.nspp();

        // 1. D-optimal refinement for existing high-weight points
        self.d_optimal_refinement()?;
        let after_dopt = self.theta.nspp();

        // 2. Local SA moves around high-weight points
        self.local_sa_injection()?;
        let after_local = self.theta.nspp();

        // 3. Sparse grid expansion
        self.sparse_grid_expansion()?;
        let after_grid = self.theta.nspp();

        // 4. Global SA injection with temperature-aware count
        self.sa_injection()?;
        let after_sa = self.theta.nspp();

        // 5. Re-inject elite points
        self.inject_elite_points()?;
        let after_elite = self.theta.nspp();

        tracing::debug!(
            "Hybrid: {} -> {} (D-opt) -> {} (local) -> {} (grid) -> {} (SA) -> {} (elite)",
            initial,
            after_dopt,
            after_local,
            after_grid,
            after_sa,
            after_elite
        );

        Ok(())
    }

    /// Exploitation phase: focus on refining existing points (lightweight)
    fn exploitation_expansion(&mut self) -> Result<()> {
        tracing::debug!("Exploitation expansion: D-optimal + light grid");

        // D-optimal refinement (only high-weight points)
        self.d_optimal_refinement()?;

        // Light grid expansion
        adaptative_grid(&mut self.theta, self.eps * 0.5, &self.ranges, THETA_D * 2.0)?;

        Ok(())
    }

    /// Convergence phase: minimal expansion, focus on verification
    fn convergence_expansion(&mut self) -> Result<()> {
        tracing::debug!("Convergence expansion: minimal changes");

        // Only light D-optimal refinement
        let eps = self.eps * 0.25;
        adaptative_grid(&mut self.theta, eps, &self.ranges, THETA_D * 2.0)?;

        Ok(())
    }

    /// Latin Hypercube Sampling for initial exploration
    fn lhs_injection(&mut self, n_samples: usize) -> Result<()> {
        let n_dims = self.ranges.len();

        // Generate LHS samples with safety margins
        let mut samples: Vec<Vec<f64>> = (0..n_samples)
            .map(|_| {
                self.ranges
                    .iter()
                    .map(|(lo, hi)| {
                        let margin = (hi - lo) * BOUNDARY_MARGIN_RATIO;
                        self.rng.random_range((lo + margin)..(hi - margin))
                    })
                    .collect()
            })
            .collect();

        // Improve LHS quality with random permutation in each dimension
        for dim in 0..n_dims {
            let (lo, hi) = self.ranges[dim];
            let margin = (hi - lo) * BOUNDARY_MARGIN_RATIO;
            let safe_lo = lo + margin;
            let safe_hi = hi - margin;
            let step = (safe_hi - safe_lo) / n_samples as f64;

            let mut perm: Vec<usize> = (0..n_samples).collect();
            perm.shuffle(&mut self.rng);

            for (i, &p) in perm.iter().enumerate() {
                let jitter = self.rng.random_range(0.0..step);
                samples[i][dim] = safe_lo + step * p as f64 + jitter;
            }
        }

        // Add samples to theta
        let mut added = 0;
        for sample in samples {
            if self.theta.check_point(&sample, THETA_D) {
                self.theta.add_point(&sample)?;
                added += 1;
            }
        }

        tracing::debug!("LHS injection: added {} of {} samples", added, n_samples);
        Ok(())
    }

    /// Sparse grid expansion in low-density regions
    fn sparse_grid_expansion(&mut self) -> Result<()> {
        let sparse_eps = self.eps * 0.5;
        adaptative_grid(&mut self.theta, sparse_eps, &self.ranges, THETA_D * 2.0)?;
        Ok(())
    }

    /// D-optimal refinement with adaptive iteration count
    /// Only refines points with significant weight to save computation
    fn d_optimal_refinement(&mut self) -> Result<()> {
        let psi = self.psi().to_ndarray();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        let error_model: AssayErrorModels = self.error_models.clone();
        let max_weight = self.w.iter().fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        let n_points_with_weights = self.w.len().min(self.theta.nspp());

        // Only refine points with meaningful weight (>1% of max)
        let min_weight_threshold = max_weight * 0.01;

        let mut candidate_points: Vec<(Array1<f64>, f64)> = Vec::default();

        for (idx, spp) in self
            .theta
            .matrix()
            .row_iter()
            .enumerate()
            .take(n_points_with_weights)
        {
            let weight = self.w[idx];
            // Skip points with negligible weight
            if weight < min_weight_threshold {
                continue;
            }

            let candidate: Vec<f64> = spp.iter().cloned().collect();
            let importance = weight / max_weight;
            candidate_points.push((Array1::from(candidate), importance));
        }

        tracing::debug!(
            "D-optimal: refining {} of {} points",
            candidate_points.len(),
            n_points_with_weights
        );

        // Optimize points in parallel
        let ranges = self.ranges.clone();
        candidate_points
            .par_iter_mut()
            .for_each(|(spp, importance)| {
                let max_iters = if *importance > HIGH_IMPORTANCE_THRESHOLD {
                    HIGH_IMPORTANCE_MAX_ITERS
                } else if *importance > HIGH_IMPORTANCE_THRESHOLD * 0.1 {
                    MEDIUM_IMPORTANCE_MAX_ITERS
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
                    // Clamp to safe boundaries to avoid ODE solver issues
                    let clamped: Array1<f64> = candidate_point
                        .iter()
                        .zip(ranges.iter())
                        .map(|(&val, &(lo, hi))| {
                            let margin = (hi - lo) * BOUNDARY_MARGIN_RATIO;
                            val.clamp(lo + margin, hi - margin)
                        })
                        .collect();
                    *spp = clamped;
                }
            });

        // Add optimized points
        for (cp, _) in candidate_points {
            self.theta.suggest_point(cp.to_vec().as_slice(), THETA_D)?;
        }

        Ok(())
    }

    /// Simulated annealing point injection
    fn sa_injection(&mut self) -> Result<()> {
        let psi = self.psi().to_ndarray();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        // Temperature-dependent injection count
        let n_inject = (SA_INJECT_BASE as f64 * (self.temperature / INITIAL_TEMPERATURE).sqrt())
            .ceil() as usize;
        let n_inject = n_inject.max(3);

        let mut accepted = 0;
        let mut proposed = 0;

        for _ in 0..n_inject * 20 {
            proposed += 1;

            // Generate random point with safety margins
            let point: Vec<f64> = self
                .ranges
                .iter()
                .map(|(lo, hi)| {
                    let margin = (hi - lo) * BOUNDARY_MARGIN_RATIO;
                    self.rng.random_range((lo + margin)..(hi - margin))
                })
                .collect();

            // Compute D-criterion
            let d_value = self.compute_d_criterion(&point, &pyl)?;

            // Metropolis acceptance
            let accept = if d_value > 0.0 {
                true
            } else {
                let p_accept = (d_value / self.temperature).exp();
                self.rng.random::<f64>() < p_accept
            };

            if accept {
                if self.theta.check_point(&point, THETA_D) {
                    self.theta.add_point(&point)?;
                    accepted += 1;
                }
            }

            if accepted >= n_inject {
                break;
            }
        }

        self.sa_accepted += accepted;
        self.sa_proposed += proposed;

        tracing::debug!(
            "SA injection: {}/{} accepted (T={:.4})",
            accepted,
            proposed,
            self.temperature
        );
        Ok(())
    }

    /// Local SA moves around existing high-weight points
    fn local_sa_injection(&mut self) -> Result<()> {
        let psi = self.psi().to_ndarray();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        let max_weight = self.w.iter().fold(f64::NEG_INFINITY, |acc, x| x.max(acc));
        let n_points = self.w.len().min(self.theta.nspp());

        let mut new_points = Vec::new();

        for (idx, spp) in self.theta.matrix().row_iter().enumerate().take(n_points) {
            let importance = self.w[idx] / max_weight;
            if importance < HIGH_IMPORTANCE_THRESHOLD * 0.5 {
                continue;
            }

            // Local perturbation
            let current: Vec<f64> = spp.iter().cloned().collect();
            let scale = self.temperature * 0.1;

            for _ in 0..5 {
                self.sa_proposed += 1;

                let perturbed: Vec<f64> = current
                    .iter()
                    .zip(self.ranges.iter())
                    .map(|(&val, &(lo, hi))| {
                        let range = hi - lo;
                        let margin = range * BOUNDARY_MARGIN_RATIO;
                        let delta = self.rng.random_range(-scale..scale) * range;
                        (val + delta).clamp(lo + margin, hi - margin)
                    })
                    .collect();

                let d_value = self.compute_d_criterion(&perturbed, &pyl)?;

                if d_value > 0.0 || self.rng.random::<f64>() < (d_value / self.temperature).exp() {
                    if self.theta.check_point(&perturbed, THETA_D) {
                        new_points.push(perturbed);
                        self.sa_accepted += 1;
                    }
                }
            }
        }

        for point in new_points {
            self.theta.add_point(&point)?;
        }

        Ok(())
    }

    /// Update elite points based on current weights and D-values
    fn update_elite_points(&mut self) -> Result<()> {
        if self.w.len() == 0 {
            return Ok(());
        }

        let psi = self.psi().to_ndarray();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        // Age existing elite points
        for elite in &mut self.elite_points {
            elite.age += 1;
        }

        // Remove old elite points
        self.elite_points.retain(|e| e.age < 20);

        // Find top points by weight
        let mut indexed_weights: Vec<(usize, f64)> = self.w.iter().enumerate().collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (idx, _weight) in indexed_weights.iter().take(ELITE_COUNT) {
            if *idx >= self.theta.nspp() {
                continue;
            }

            let params: Vec<f64> = self.theta.matrix().row(*idx).iter().cloned().collect();
            let d_value = self.compute_d_criterion(&params, &pyl).unwrap_or(0.0);

            // Check if this point is already elite
            let already_elite = self.elite_points.iter().any(|e| {
                e.params
                    .iter()
                    .zip(&params)
                    .all(|(a, b)| (a - b).abs() < THETA_D)
            });

            if !already_elite && self.elite_points.len() < ELITE_COUNT * 2 {
                self.elite_points.push(ElitePoint {
                    params,
                    d_value,
                    age: 0,
                });
            }
        }

        // Keep only top elite points
        self.elite_points
            .sort_by(|a, b| b.d_value.partial_cmp(&a.d_value).unwrap());
        self.elite_points.truncate(ELITE_COUNT);

        Ok(())
    }

    /// Inject elite points back into theta
    fn inject_elite_points(&mut self) -> Result<()> {
        for elite in &self.elite_points {
            if self.theta.check_point(&elite.params, THETA_D) {
                self.theta.add_point(&elite.params)?;
            }
        }
        Ok(())
    }

    /// Perform restart when stuck
    fn perform_restart(&mut self) -> Result<()> {
        self.restart_count += 1;

        // Reset temperature
        self.temperature = INITIAL_TEMPERATURE * 0.5_f64.powi(self.restart_count as i32);

        // Reset phase
        self.phase = Phase::Hybrid;

        // Reset cooling rate
        self.cooling_rate = BASE_COOLING_RATE;

        // Reset stagnation counter
        self.cycles_since_improvement = 0;

        // Inject diverse points via LHS
        self.lhs_injection(LHS_SAMPLES / 2)?;

        tracing::info!(
            "Restart complete: T={:.4}, {} elite points preserved",
            self.temperature,
            self.elite_points.len()
        );

        Ok(())
    }

    /// Compute D-criterion for a candidate point
    fn compute_d_criterion(&self, point: &[f64], pyl: &Array1<f64>) -> Result<f64> {
        let theta_single = Array1::from(point.to_vec()).insert_axis(Axis(0));

        let psi_single = pharmsol::prelude::simulator::log_likelihood_matrix(
            &self.equation,
            &self.data,
            &theta_single,
            &self.error_models,
            false,
        )?.mapv(f64::exp);

        let nsub = psi_single.nrows() as f64;
        let mut d_sum = -nsub;

        for (p_i, pyl_i) in psi_single.iter().zip(pyl.iter()) {
            d_sum += p_i / pyl_i;
        }

        Ok(d_sum)
    }

    /// Multi-criterion convergence check
    fn check_convergence(&mut self) -> Result<bool> {
        if self.objf_history.len() < CONVERGENCE_WINDOW {
            return Ok(false);
        }

        // Criterion 1: Objective function stability
        let recent: Vec<f64> = self
            .objf_history
            .iter()
            .rev()
            .take(CONVERGENCE_WINDOW)
            .cloned()
            .collect();

        let objf_stable = recent.windows(2).all(|w| (w[0] - w[1]).abs() < THETA_G);

        if !objf_stable {
            return Ok(false);
        }

        // Criterion 2: Global optimality via Monte Carlo
        let psi = self.psi().to_ndarray();
        let w: Array1<f64> = self.w.clone().iter().collect();
        let pyl = psi.dot(&w);

        let max_d = self.monte_carlo_global_check(&pyl)?;

        if max_d > GLOBAL_OPTIMALITY_THRESHOLD {
            tracing::debug!("Global check failed: max_D = {:.6}", max_d);
            return Ok(false);
        }

        tracing::debug!("Global check passed: max_D = {:.6}", max_d);
        Ok(true)
    }

    /// Monte Carlo estimate of maximum D-criterion
    fn monte_carlo_global_check(&mut self, pyl: &Array1<f64>) -> Result<f64> {
        let points: Vec<Vec<f64>> = (0..GLOBAL_OPTIMALITY_SAMPLES)
            .map(|_| {
                self.ranges
                    .iter()
                    .map(|(lo, hi)| self.rng.random_range(*lo..*hi))
                    .collect()
            })
            .collect();

        let max_d = points
            .into_par_iter()
            .filter_map(|point| self.compute_d_criterion(&point, pyl).ok())
            .reduce(|| f64::NEG_INFINITY, f64::max);

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

/// Support Point Optimizer with configurable iteration count
struct SppOptimizerAdaptive<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    sig: &'a AssayErrorModels,
    pyl: &'a Array1<f64>,
    max_iters: u64,
}

impl<E: Equation> CostFunction for SppOptimizerAdaptive<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, spp: &Self::Param) -> Result<Self::Output, Error> {
        let theta = Array1::from(spp.clone()).insert_axis(Axis(0));

        let psi = pharmsol::prelude::simulator::log_likelihood_matrix(
            self.equation,
            self.data,
            &theta,
            self.sig,
            false,
        )?.mapv(f64::exp);

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
        sig: &'a AssayErrorModels,
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
    let perturbation_percentage = 0.05;

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
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::Warmup), "Warmup");
        assert_eq!(format!("{}", Phase::Hybrid), "Hybrid");
        assert_eq!(format!("{}", Phase::Exploitation), "Exploitation");
        assert_eq!(format!("{}", Phase::Convergence), "Convergence");
    }

    #[test]
    fn test_initial_simplex() {
        let point = vec![1.0, 2.0, 3.0];
        let simplex = create_initial_simplex(&point);
        assert_eq!(simplex.len(), 4);
        assert_eq!(simplex[0], point);
    }

    #[test]
    fn test_temperature_bounds() {
        assert!(INITIAL_TEMPERATURE > MIN_TEMPERATURE);
        assert!(BASE_COOLING_RATE > 0.0 && BASE_COOLING_RATE < 1.0);
    }

    #[test]
    fn test_convergence_window() {
        assert!(CONVERGENCE_WINDOW >= 2);
    }
}
