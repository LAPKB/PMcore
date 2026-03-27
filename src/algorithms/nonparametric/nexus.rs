//! # NEXUS: Non-parametric EXploration via Unified Subject-driven Search
//!
//! A state-of-the-art hybrid algorithm combining the best of multiple approaches
//! with novel innovations in cross-entropy optimization and adaptive exploration.
//!
//! ## Key Innovations
//!
//! NEXUS combines:
//! 1. **Cross-Entropy Method (CE)** with GMM for learning the distribution of good solutions
//! 2. **Subject-guided exploration** for targeted mode discovery
//! 3. **Adaptive simulated annealing** with temperature feedback
//! 4. **D-optimal refinement** with hierarchical iteration allocation
//! 5. **Multi-scale Sobol global verification** for convergence certificates
//!
//! ## The Cross-Entropy Insight
//!
//! Unlike SA which samples blindly, CE maintains a Gaussian Mixture Model (GMM)
//! that learns where good solutions tend to be. Each cycle:
//! 1. Sample candidates from GMM
//! 2. Evaluate D-criterion for all candidates
//! 3. Select elite points (top 10%)
//! 4. Update GMM to fit elite distribution
//!
//! This converges faster than SA because it learns problem structure.
//!
//! ## The Subject-Guided Insight
//!
//! The D-criterion D(θ*) = Σᵢ P(yᵢ|θ*) / P(yᵢ|G) - N is large when:
//! - P(yᵢ|θ*) is high: θ* explains subject i well
//! - P(yᵢ|G) is low: current mixture explains subject i poorly
//!
//! **Insight**: Find parameters for poorly-fit subjects, targeting modes the
//! mixture is missing.
//!
//! ## Algorithm Phases
//!
//! ### Phase 1: Warmup
//! - Stratified Sobol initialization for space-filling coverage
//! - Adaptive grid expansion to build parameter space scaffold
//! - GMM initialization from initial support points
//!
//! ### Phase 2: Hybrid Expansion  
//! - **Cross-entropy sampling** from adaptive GMM
//! - **Subject-guided search** from poorly-fit subjects
//! - **Adaptive SA** with feedback-controlled temperature
//! - **D-optimal refinement** with hierarchical iteration counts
//! - **Elite preservation** to prevent loss of good solutions
//!
//! ### Phase 3: Convergence Verification
//! - Multi-scale Sobol global optimality check (64 → 256 → 1024 samples)
//! - Final polishing of all support points
//! - Convergence certificate when all scales pass
//!
//! ## Convergence Guarantees
//!
//! NEXUS provides multiple convergence criteria:
//! 1. Objective function stability (THETA_G)
//! 2. Weight stability (THETA_W)
//! 3. P(Y|L) criterion (THETA_F)
//! 4. Multi-scale global D-criterion < threshold

use crate::algorithms::{NativeNonparametricConfig, NonparametricAlgorithmInput, Status, StopReason};
use crate::estimation::nonparametric::{calculate_psi, CycleLog, NonparametricWorkspace, NPCycle, Psi, Theta, Weights};
use crate::prelude::algorithms::Algorithms;
use crate::estimation::nonparametric::adaptative_grid;
use crate::estimation::nonparametric::ipm::burke;
use crate::estimation::nonparametric::qr;
use crate::estimation::nonparametric::sample_space_for_parameters;

use anyhow::{bail, Result};
use ndarray::parallel::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use ndarray::{Array1, Axis};
use pharmsol::prelude::AssayErrorModel;
use pharmsol::prelude::{
    data::{Data, AssayErrorModels},
    simulator::Equation,
};
use rand::prelude::*;
use sobol_burley::sample;

use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};

// ============================================================================
// ALGORITHM CONSTANTS
// ============================================================================

/// Objective function convergence threshold
const THETA_G: f64 = 1e-4;
/// P(Y|L) convergence criterion
const THETA_F: f64 = 1e-2;
/// Minimum distance between support points (normalized)
const THETA_D: f64 = 1e-4;
/// Weight stability threshold
const THETA_W: f64 = 1e-3;
/// Grid spacing for adaptive expansion
const INITIAL_EPS: f64 = 0.2;
/// Minimum grid spacing before reset
const MIN_EPS: f64 = 1e-4;

/// Number of warm-up cycles using grid expansion
const WARMUP_CYCLES: usize = 5;

// === Cross-Entropy Method Parameters ===
/// Number of samples to generate from GMM per cycle
const CE_SAMPLE_SIZE: usize = 50;
/// Fraction of samples considered "elite" (top performers)
const CE_ELITE_FRACTION: f64 = 0.10;
/// Number of GMM components for multimodal handling
const CE_GMM_COMPONENTS: usize = 3;
/// Minimum variance for GMM (prevents collapse)
const CE_MIN_VARIANCE: f64 = 1e-6;
/// Smoothing factor for GMM updates (0 = no smoothing, 1 = no update)
const CE_SMOOTHING: f64 = 0.3;
/// Decay rate for sample size (reduces over cycles)
const CE_SAMPLE_DECAY: f64 = 0.95;

// === Subject-guided parameters ===
/// Fraction of subjects considered "poorly fit" (bottom percentile)
const RESIDUAL_SUBJECT_FRACTION: f64 = 0.3;
/// Minimum number of residual subjects to process
const MIN_RESIDUAL_SUBJECTS: usize = 3;
/// Maximum Nelder-Mead iterations for subject MAP estimation
const SUBJECT_MAP_MAX_ITERS: u64 = 30;

// === D-optimal refinement parameters ===
/// Maximum Nelder-Mead iterations for high-weight D-optimal refinement
const DOPT_HIGH_WEIGHT_ITERS: u64 = 100;
/// Maximum Nelder-Mead iterations for medium-weight D-optimal refinement
const DOPT_MED_WEIGHT_ITERS: u64 = 40;
/// Maximum Nelder-Mead iterations for low-weight D-optimal refinement  
const DOPT_LOW_WEIGHT_ITERS: u64 = 15;
/// Weight threshold for "high importance" (fraction of max weight)
const HIGH_WEIGHT_THRESHOLD: f64 = 0.10;
/// Weight threshold for "medium importance"
const MED_WEIGHT_THRESHOLD: f64 = 0.01;

// === Adaptive Simulated Annealing Parameters ===
/// Initial temperature for SA
const INITIAL_TEMPERATURE: f64 = 5.0;
/// Base cooling rate (adapted based on acceptance)
const BASE_COOLING_RATE: f64 = 0.85;
/// Number of SA samples per injection cycle
const SA_INJECT_COUNT: usize = 50;
/// Minimum temperature before SA stops
const MIN_TEMPERATURE: f64 = 0.01;
/// Target acceptance ratio for adaptive temperature
const TARGET_ACCEPTANCE_RATIO: f64 = 0.25;
/// Reheat factor when acceptance is too low
const REHEAT_FACTOR: f64 = 1.2;

// === Elite Preservation ===
/// Number of elite points to preserve across cycles
const ELITE_COUNT: usize = 5;
/// Maximum age of elite point before removal
const ELITE_MAX_AGE: usize = 20;

// === Multi-Scale Global Optimality ===
/// Sobol samples at each scale level
const GLOBAL_CHECK_SCALES: [usize; 3] = [64, 256, 1024];
/// D-criterion threshold for global optimality
const GLOBAL_D_THRESHOLD: f64 = 0.005;
/// Seed for reproducible Sobol sequence
const SOBOL_SEED: u32 = 0;

/// Consecutive stable cycles needed for convergence
const CONVERGENCE_WINDOW: usize = 3;

/// Boundary margin to prevent numerical issues (fraction of range)
const BOUNDARY_MARGIN: f64 = 0.005;

// ============================================================================
// GAUSSIAN MIXTURE MODEL FOR CROSS-ENTROPY
// ============================================================================

/// A single Gaussian component in the mixture
#[derive(Debug, Clone)]
struct GaussianComponent {
    /// Mean vector (center of component)
    mean: Vec<f64>,
    /// Diagonal covariance (variance per dimension)
    variance: Vec<f64>,
    /// Mixture weight (probability of selecting this component)
    weight: f64,
}

impl GaussianComponent {
    #[allow(dead_code)]
    fn new(_n_dims: usize, ranges: &[(f64, f64)]) -> Self {
        // Initialize at center with variance = (range/4)^2
        let mean: Vec<f64> = ranges.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();
        let variance: Vec<f64> = ranges
            .iter()
            .map(|(lo, hi)| ((hi - lo) / 4.0).powi(2))
            .collect();
        Self {
            mean,
            variance,
            weight: 1.0,
        }
    }

    /// Sample a point from this Gaussian
    fn sample(&self, rng: &mut StdRng, ranges: &[(f64, f64)]) -> Vec<f64> {
        self.mean
            .iter()
            .zip(self.variance.iter())
            .zip(ranges.iter())
            .map(|((&m, &v), (lo, hi))| {
                let std = v.sqrt();
                let margin = (hi - lo) * BOUNDARY_MARGIN;
                // Sample from N(m, std^2) and clamp to bounds
                let sample = m + std * sample_standard_normal(rng);
                sample.clamp(lo + margin, hi - margin)
            })
            .collect()
    }

    /// Compute log probability density of a point under this Gaussian
    fn log_pdf(&self, point: &[f64]) -> f64 {
        let n = point.len() as f64;
        let mut log_p = -0.5 * n * (2.0 * std::f64::consts::PI).ln();

        for ((&x, &m), &v) in point.iter().zip(self.mean.iter()).zip(self.variance.iter()) {
            let safe_v = v.max(CE_MIN_VARIANCE);
            log_p -= 0.5 * safe_v.ln();
            log_p -= 0.5 * (x - m).powi(2) / safe_v;
        }

        log_p
    }
}

/// Gaussian Mixture Model for Cross-Entropy sampling
#[derive(Debug, Clone)]
struct GMM {
    components: Vec<GaussianComponent>,
    n_dims: usize,
}

impl GMM {
    /// Create a new GMM with k components initialized across the parameter space
    fn new(n_components: usize, n_dims: usize, ranges: &[(f64, f64)], rng: &mut StdRng) -> Self {
        let components: Vec<GaussianComponent> = (0..n_components)
            .map(|i| {
                // Spread initial means across the space
                let mean: Vec<f64> = ranges
                    .iter()
                    .map(|(lo, hi)| {
                        let frac = (i as f64 + 0.5) / n_components as f64;
                        lo + frac * (hi - lo) + rng.random_range(-0.1..0.1) * (hi - lo)
                    })
                    .collect();
                let variance: Vec<f64> = ranges
                    .iter()
                    .map(|(lo, hi)| ((hi - lo) / (n_components as f64 + 1.0)).powi(2))
                    .collect();
                GaussianComponent {
                    mean,
                    variance,
                    weight: 1.0 / n_components as f64,
                }
            })
            .collect();

        Self { components, n_dims }
    }

    /// Initialize GMM from existing support points and weights
    fn from_theta(
        theta: &Theta,
        weights: &Weights,
        ranges: &[(f64, f64)],
        _rng: &mut StdRng,
    ) -> Self {
        let n_dims = ranges.len();
        let n_spp = theta.nspp().min(weights.len());

        if n_spp == 0 {
            // Empty theta - use default initialization
            let mut rng = StdRng::seed_from_u64(42);
            return Self::new(CE_GMM_COMPONENTS, n_dims, ranges, &mut rng);
        }

        // Use K-means-like initialization: pick top-weighted points as centers
        let mut indexed: Vec<(usize, f64)> = weights.iter().enumerate().take(n_spp).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let n_components = CE_GMM_COMPONENTS.min(n_spp);
        let components: Vec<GaussianComponent> = indexed
            .iter()
            .take(n_components)
            .map(|(idx, w)| {
                let mean: Vec<f64> = theta.matrix().row(*idx).iter().cloned().collect();
                let variance: Vec<f64> = ranges
                    .iter()
                    .map(|(lo, hi)| ((hi - lo) / 6.0).powi(2)) // Start with moderate variance
                    .collect();
                GaussianComponent {
                    mean,
                    variance,
                    weight: *w,
                }
            })
            .collect();

        // Normalize weights
        let total_weight: f64 = components.iter().map(|c| c.weight).sum();
        let mut gmm = Self { components, n_dims };
        if total_weight > 0.0 {
            for c in &mut gmm.components {
                c.weight /= total_weight;
            }
        }

        gmm
    }

    /// Sample n points from the GMM
    fn sample(&self, n: usize, rng: &mut StdRng, ranges: &[(f64, f64)]) -> Vec<Vec<f64>> {
        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            // Select component based on weights
            let u: f64 = rng.random();
            let mut cumsum = 0.0;
            let mut selected = 0;
            for (i, c) in self.components.iter().enumerate() {
                cumsum += c.weight;
                if u <= cumsum {
                    selected = i;
                    break;
                }
            }

            samples.push(self.components[selected].sample(rng, ranges));
        }

        samples
    }

    /// Update GMM from elite points using weighted MLE
    fn update_from_elite(&mut self, elite_points: &[(Vec<f64>, f64)], ranges: &[(f64, f64)]) {
        if elite_points.is_empty() {
            return;
        }

        // Soft assignment of elite points to components
        let n_elite = elite_points.len();
        let n_components = self.components.len();

        // E-step: compute responsibilities
        let mut responsibilities: Vec<Vec<f64>> = vec![vec![0.0; n_components]; n_elite];

        for (i, (point, _)) in elite_points.iter().enumerate() {
            let mut log_probs: Vec<f64> = self
                .components
                .iter()
                .map(|c| c.weight.ln() + c.log_pdf(point))
                .collect();

            // Log-sum-exp for numerical stability
            let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum: f64 = log_probs
                .iter()
                .map(|&lp| (lp - max_log).exp())
                .sum::<f64>()
                .ln()
                + max_log;

            for (j, lp) in log_probs.iter_mut().enumerate() {
                responsibilities[i][j] = (*lp - log_sum).exp();
            }
        }

        // M-step: update parameters with smoothing
        for (k, component) in self.components.iter_mut().enumerate() {
            let mut total_resp = 0.0;
            let mut new_mean = vec![0.0; self.n_dims];
            let mut new_var = vec![0.0; self.n_dims];

            // Weight responsibilities by D-criterion values
            for (i, (point, d_val)) in elite_points.iter().enumerate() {
                let resp = responsibilities[i][k] * d_val.max(0.0);
                total_resp += resp;
                for (j, &x) in point.iter().enumerate() {
                    new_mean[j] += resp * x;
                }
            }

            if total_resp > 1e-10 {
                for j in 0..self.n_dims {
                    new_mean[j] /= total_resp;
                }

                // Compute variance
                for (i, (point, d_val)) in elite_points.iter().enumerate() {
                    let resp = responsibilities[i][k] * d_val.max(0.0);
                    for (j, &x) in point.iter().enumerate() {
                        new_var[j] += resp * (x - new_mean[j]).powi(2);
                    }
                }

                for j in 0..self.n_dims {
                    new_var[j] = (new_var[j] / total_resp).max(CE_MIN_VARIANCE);
                    // Bound variance to reasonable range
                    let (lo, hi) = ranges[j];
                    let max_var = ((hi - lo) / 2.0).powi(2);
                    new_var[j] = new_var[j].min(max_var);
                }

                // Apply smoothing to prevent sudden changes
                for j in 0..self.n_dims {
                    component.mean[j] =
                        CE_SMOOTHING * component.mean[j] + (1.0 - CE_SMOOTHING) * new_mean[j];
                    component.variance[j] =
                        CE_SMOOTHING * component.variance[j] + (1.0 - CE_SMOOTHING) * new_var[j];
                }

                // Update weight
                let new_weight =
                    total_resp / elite_points.iter().map(|(_, d)| d.max(0.0)).sum::<f64>();
                component.weight =
                    CE_SMOOTHING * component.weight + (1.0 - CE_SMOOTHING) * new_weight;
            }
        }

        // Normalize weights
        let total_weight: f64 = self.components.iter().map(|c| c.weight).sum();
        if total_weight > 0.0 {
            for c in &mut self.components {
                c.weight /= total_weight;
            }
        }
    }
}

/// Sample from standard normal distribution using Box-Muller
fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ============================================================================
// ELITE POINT TRACKING
// ============================================================================

/// An elite point with metadata for preservation across cycles
#[derive(Debug, Clone)]
struct ElitePoint {
    params: Vec<f64>,
    d_value: f64,
    age: usize,
}

// ============================================================================
// CONVERGENCE STATE
// ============================================================================

/// Algorithm phase
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    /// Initial grid-based coverage
    Warmup,
    /// Subject-guided expansion + D-optimal refinement
    Expansion,
    /// Final convergence verification
    Convergence,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Phase::Warmup => write!(f, "Warmup"),
            Phase::Expansion => write!(f, "Expansion"),
            Phase::Convergence => write!(f, "Convergence"),
        }
    }
}

// ============================================================================
// NEXUS STRUCT
// ============================================================================

/// NEXUS: Non-parametric EXploration via Unified Subject-driven Search
#[derive(Debug)]
pub struct NEXUS<E: Equation + Send + 'static> {
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
    /// Current grid spacing (for warm-up phase)
    eps: f64,
    /// Previous objective function value
    last_objf: f64,
    /// Current objective function value
    objf: f64,
    /// Best objective function seen
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
    /// Unified runtime/model-derived configuration
    config: NativeNonparametricConfig,

    // NEXUS-specific fields
    /// Current algorithm phase
    phase: Phase,
    /// History of objective function values
    objf_history: Vec<f64>,
    /// Sobol sequence index for reproducible sampling
    sobol_index: u32,
    /// Maximum D found in last global check
    last_global_d_max: f64,
    /// Count of stable cycles for convergence
    stability_counter: usize,
    /// Current global check scale level
    current_global_scale: usize,

    // Cross-Entropy fields
    /// Gaussian Mixture Model for CE sampling
    gmm: Option<GMM>,
    /// Current CE sample size (decays over cycles)
    ce_sample_size: f64,

    // Adaptive SA fields
    /// SA temperature for global exploration
    temperature: f64,
    /// Effective cooling rate (adaptive)
    cooling_rate: f64,
    /// SA accepted count this cycle
    sa_accepted: usize,
    /// SA proposed count this cycle  
    sa_proposed: usize,

    // Elite preservation
    /// Elite points preserved across cycles
    elite_points: Vec<ElitePoint>,

    /// Random number generator
    rng: StdRng,
}

// ============================================================================
// ALGORITHMS TRAIT IMPLEMENTATION
// ============================================================================

impl<E: Equation + Send + 'static> Algorithms<E> for NEXUS<E> {
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
        sample_space_for_parameters(&self.config.parameter_space, &self.config.prior)
            .unwrap()
    }

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn increment_cycle(&mut self) -> usize {
        self.cycle += 1;

        // Phase transition: Warmup → Expansion
        if self.cycle > WARMUP_CYCLES && self.phase == Phase::Warmup {
            self.phase = Phase::Expansion;

            // Initialize GMM from current theta
            self.gmm = Some(GMM::from_theta(
                &self.theta,
                &self.w,
                &self.ranges,
                &mut self.rng,
            ));

            tracing::info!(
                "NEXUS: Warmup → Expansion (cycle {}, {} support points, GMM initialized)",
                self.cycle,
                self.theta.nspp()
            );
        }

        // Adaptive temperature adjustment based on acceptance ratio
        self.adapt_temperature();

        // Decay CE sample size
        self.ce_sample_size = (self.ce_sample_size * CE_SAMPLE_DECAY).max(10.0);

        // Track best objective
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

    fn evaluation(&mut self) -> Result<Status> {
        tracing::info!("Objective function = {:.4}", -2.0 * self.objf);
        tracing::debug!(
            "Support points: {} | Phase: {} | T: {:.3} | CE samples: {:.0}",
            self.theta.nspp(),
            self.phase,
            self.temperature,
            self.ce_sample_size
        );

        // Log error models
        self.error_models.iter().for_each(|(outeq, em)| {
            if AssayErrorModel::None != *em {
                tracing::debug!(
                    "Error model outeq {}: {:.4}",
                    outeq,
                    em.factor().unwrap_or_default()
                );
            }
        });

        // Track history
        self.objf_history.push(self.objf);

        // Warn on decrease
        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective function decreased: {:.4} → {:.4}",
                -2.0 * self.last_objf,
                -2.0 * self.objf
            );
        }

        // Check convergence
        let converged = self.check_convergence()?;
        if converged {
            tracing::info!("NEXUS converged after {} cycles", self.cycle);
            self.set_status(Status::Stop(StopReason::Converged));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // NPAG-style eps convergence (during warmup and expansion)
        if self.phase != Phase::Convergence {
            if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > MIN_EPS {
                self.eps /= 2.0;
                tracing::debug!("Halving eps to {:.6}", self.eps);

                if self.eps <= MIN_EPS {
                    let pyl = self.psi.matrix() * self.w.weights();
                    self.f1 = pyl.iter().map(|x| x.ln()).sum();
                    if (self.f1 - self.f0).abs() <= THETA_F {
                        // Transition to convergence verification
                        self.phase = Phase::Convergence;
                        tracing::info!(
                            "NEXUS: Expansion → Convergence (cycle {}, verifying global optimality)",
                            self.cycle
                        );
                    } else {
                        self.f0 = self.f1;
                        self.eps = INITIAL_EPS;
                    }
                }
            }
        }

        // Check maximum cycles
        if self.cycle >= self.config.max_cycles {
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
        // Store previous weights for stability checking
        self.w_prev = self.w.clone();

        // Lambda filter: remove low-weight points
        let max_lambda = self
            .lambda
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        let threshold = max_lambda / 10000.0;
        let keep: Vec<usize> = self
            .lambda
            .iter()
            .enumerate()
            .filter(|(_, lam)| *lam > threshold)
            .map(|(i, _)| i)
            .collect();

        let dropped = self.psi.matrix().ncols() - keep.len();
        if dropped > 0 {
            tracing::debug!("Lambda filter dropped {} point(s)", dropped);
        }

        self.theta.filter_indices(keep.as_slice());
        self.psi.filter_column_indices(keep.as_slice());

        // QR rank-revealing factorization
        let (r, perm) = qr::qrd(&self.psi)?;
        let keep_n = self.psi.matrix().ncols().min(self.psi.matrix().nrows());
        let keep: Vec<usize> = (0..keep_n)
            .filter(|&i| {
                let test = r.col(i).norm_l2();
                let r_diag = r.get(i, i);
                (r_diag / test).abs() >= 1e-8
            })
            .map(|i| *perm.get(i).unwrap())
            .collect();

        let dropped = self.psi.matrix().ncols() - keep.len();
        if dropped > 0 {
            tracing::debug!("QR dropped {} point(s)", dropped);
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
                    Err(err) => bail!("Error in IPM during optim: {:?}", err),
                };
                let (lambda_down, objf_down) = match burke(&psi_down) {
                    Ok((lambda, objf)) => (lambda, objf),
                    Err(err) => bail!("Error in IPM during optim: {:?}", err),
                };

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
                Ok(())
            })?;

        Ok(())
    }

    fn expansion(&mut self) -> Result<()> {
        match self.phase {
            Phase::Warmup => {
                // Use adaptive grid expansion for initial coverage
                self.grid_expansion()?;
            }
            Phase::Expansion => {
                // Hybrid expansion: combining all strategies
                let initial_spp = self.theta.nspp();

                // Step 1: D-optimal refinement FIRST (like NPSAH)
                self.d_optimal_refinement()?;
                let after_dopt = self.theta.nspp();

                // Step 2: Cross-Entropy sampling from GMM (new!)
                self.cross_entropy_expansion()?;
                let after_ce = self.theta.nspp();

                // Step 3: Sparse grid expansion (reduced rate in hybrid phase)
                self.sparse_grid_expansion()?;
                let after_grid = self.theta.nspp();

                // Step 4: Adaptive SA injection for global exploration
                if self.temperature > MIN_TEMPERATURE {
                    self.sa_injection()?;
                }
                let after_sa = self.theta.nspp();

                // Step 5: Subject-guided exploration
                self.subject_guided_expansion()?;
                let after_subject = self.theta.nspp();

                // Step 6: Re-inject elite points
                self.inject_elite_points()?;
                let after_elite = self.theta.nspp();

                tracing::debug!(
                    "Expansion: {} → {} (D-opt) → {} (CE) → {} (grid) → {} (SA) → {} (subject) → {} (elite)",
                    initial_spp, after_dopt, after_ce, after_grid, after_sa, after_subject, after_elite
                );
            }
            Phase::Convergence => {
                // Multi-scale global verification with injection
                self.multi_scale_global_check()?;
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
// NEXUS-SPECIFIC METHODS
// ============================================================================

impl<E: Equation + Send + 'static> NEXUS<E> {
    pub(crate) fn from_input(input: NonparametricAlgorithmInput<E>) -> Result<Box<Self>> {
        let config = input.native_config()?;
        let seed = config.prior.seed().unwrap_or(42);
        let ranges = config.ranges.clone();
        let error_models = input.error_models().clone();

        Ok(Box::new(Self {
            equation: input.equation,
            ranges: ranges.clone(),
            psi: Psi::new(),
            theta: Theta::new(),
            lambda: Weights::default(),
            w: Weights::default(),
            w_prev: Weights::default(),
            eps: INITIAL_EPS,
            last_objf: -1e30,
            objf: f64::NEG_INFINITY,
            best_objf: f64::NEG_INFINITY,
            f0: -1e30,
            f1: f64::default(),
            cycle: 0,
            gamma_delta: vec![0.1; error_models.len()],
            error_models,
            status: Status::Continue,
            cycle_log: CycleLog::new(),
            data: input.data,
            config,
            phase: Phase::Warmup,
            objf_history: Vec::with_capacity(100),
            sobol_index: seed as u32,
            last_global_d_max: f64::INFINITY,
            stability_counter: 0,
            current_global_scale: 0,
            gmm: None,
            ce_sample_size: CE_SAMPLE_SIZE as f64,
            temperature: INITIAL_TEMPERATURE,
            cooling_rate: BASE_COOLING_RATE,
            sa_accepted: 0,
            sa_proposed: 0,
            elite_points: Vec::with_capacity(ELITE_COUNT * 2),
            rng: StdRng::seed_from_u64(seed as u64),
        }))
    }

    /// Compute P(Y|G) = Psi * w for all subjects
    fn compute_pyl(&self) -> Array1<f64> {
        let psi = self.psi().to_ndarray();
        let w: Array1<f64> = self.w.clone().iter().collect();
        psi.dot(&w)
    }

    /// Compute D-criterion for a candidate point
    fn compute_d(&self, point: &[f64], pyl: &Array1<f64>) -> Result<f64> {
        let theta_single = ndarray::Array1::from(point.to_vec()).insert_axis(Axis(0));

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
            if *pyl_i > 0.0 {
                d_sum += p_i / pyl_i;
            }
        }

        Ok(d_sum)
    }

    /// Check multi-criterion convergence
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
            self.stability_counter = 0;
            return Ok(false);
        }

        // Criterion 2: Weight stability
        if !self.weights_stable() {
            self.stability_counter = 0;
            return Ok(false);
        }

        self.stability_counter += 1;

        // Criterion 3: Multi-scale global optimality (only in Convergence phase)
        if self.phase == Phase::Convergence && self.stability_counter >= CONVERGENCE_WINDOW {
            // Progressive multi-scale check
            if self.current_global_scale < GLOBAL_CHECK_SCALES.len() {
                let n_samples = GLOBAL_CHECK_SCALES[self.current_global_scale];
                let pyl = self.compute_pyl();
                let max_d = self.sobol_global_check_n(&pyl, n_samples)?;

                if max_d > GLOBAL_D_THRESHOLD {
                    tracing::debug!(
                        "Global check scale {} failed: max_D = {:.4} > {:.4}",
                        self.current_global_scale,
                        max_d,
                        GLOBAL_D_THRESHOLD
                    );
                    // Reset to expansion phase if we fail
                    self.phase = Phase::Expansion;
                    self.stability_counter = 0;
                    self.current_global_scale = 0;
                    return Ok(false);
                }

                tracing::info!(
                    "Global check scale {} ({} samples) passed: max_D = {:.4}",
                    self.current_global_scale,
                    n_samples,
                    max_d
                );
                self.current_global_scale += 1;

                // Not converged until all scales pass
                if self.current_global_scale < GLOBAL_CHECK_SCALES.len() {
                    return Ok(false);
                }
            }

            // All scales passed!
            tracing::info!("All global optimality scales passed - convergence verified");
            return Ok(true);
        }

        Ok(false)
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

    // ════════════════════════════════════════════════════════════════════
    // PHASE 1: GRID EXPANSION (Warmup)
    // ════════════════════════════════════════════════════════════════════

    /// Adaptive grid expansion for initial coverage
    fn grid_expansion(&mut self) -> Result<()> {
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D)?;
        Ok(())
    }

    /// Sparse grid expansion (reduced epsilon)
    fn sparse_grid_expansion(&mut self) -> Result<()> {
        let sparse_eps = self.eps * 0.5;
        adaptative_grid(&mut self.theta, sparse_eps, &self.ranges, THETA_D * 2.0)?;
        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    // CROSS-ENTROPY EXPANSION (Novel)
    // ════════════════════════════════════════════════════════════════════

    /// Cross-Entropy Method expansion using GMM
    ///
    /// Unlike SA which samples blindly, CE learns the distribution of good solutions.
    /// Each cycle: sample from GMM → evaluate → select elite → update GMM
    fn cross_entropy_expansion(&mut self) -> Result<()> {
        // Initialize GMM if not present
        if self.gmm.is_none() {
            self.gmm = Some(GMM::from_theta(
                &self.theta,
                &self.w,
                &self.ranges,
                &mut self.rng,
            ));
        }

        let gmm = self.gmm.as_ref().unwrap();
        let pyl = self.compute_pyl();
        let n_samples = self.ce_sample_size.ceil() as usize;

        // Sample candidates from GMM
        let candidates = gmm.sample(n_samples, &mut self.rng, &self.ranges);

        // Evaluate D-criterion for all candidates
        let mut evaluated: Vec<(Vec<f64>, f64)> = candidates
            .into_iter()
            .filter_map(|point| self.compute_d(&point, &pyl).ok().map(|d| (point, d)))
            .collect();

        // Sort by D descending (best first)
        evaluated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select elite (top fraction)
        let n_elite = (evaluated.len() as f64 * CE_ELITE_FRACTION).ceil() as usize;
        let elite: Vec<(Vec<f64>, f64)> = evaluated.into_iter().take(n_elite.max(1)).collect();

        // Add elite points that improve the mixture
        let mut added = 0;
        for (point, d) in &elite {
            if *d > 0.0 && self.theta.check_point(point, THETA_D) {
                self.theta.add_point(point)?;
                added += 1;
            }
        }

        // Update GMM from elite points
        if !elite.is_empty() {
            if let Some(ref mut gmm) = self.gmm {
                gmm.update_from_elite(&elite, &self.ranges);
            }
        }

        tracing::debug!(
            "CE expansion: sampled {}, elite {}, added {}",
            n_samples,
            elite.len(),
            added
        );

        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    // ADAPTIVE TEMPERATURE CONTROL
    // ════════════════════════════════════════════════════════════════════

    /// Adapt SA temperature based on acceptance ratio
    fn adapt_temperature(&mut self) {
        if self.sa_proposed > 0 {
            let acceptance_ratio = self.sa_accepted as f64 / self.sa_proposed as f64;

            // Adjust cooling rate based on acceptance ratio
            if acceptance_ratio < TARGET_ACCEPTANCE_RATIO * 0.5 {
                // Too cold - slow down cooling and possibly reheat
                self.cooling_rate = (self.cooling_rate + 0.02).min(0.98);
                if acceptance_ratio < 0.1 && self.temperature < 0.5 {
                    self.temperature *= REHEAT_FACTOR;
                    tracing::debug!("Reheating to T = {:.4}", self.temperature);
                }
            } else if acceptance_ratio > TARGET_ACCEPTANCE_RATIO * 1.5 {
                // Too hot - speed up cooling
                self.cooling_rate = (self.cooling_rate - 0.02).max(0.80);
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

    // ════════════════════════════════════════════════════════════════════
    // ELITE POINT MANAGEMENT
    // ════════════════════════════════════════════════════════════════════

    /// Update elite points based on current weights and D-values
    fn update_elite_points(&mut self) -> Result<()> {
        if self.w.len() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();

        // Age existing elite points
        for elite in &mut self.elite_points {
            elite.age += 1;
        }

        // Remove old elite points
        self.elite_points.retain(|e| e.age < ELITE_MAX_AGE);

        // Find top points by weight
        let n_spp = self.theta.nspp().min(self.w.len());
        let mut indexed_weights: Vec<(usize, f64)> =
            self.w.iter().enumerate().take(n_spp).collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (idx, _weight) in indexed_weights.iter().take(ELITE_COUNT) {
            if *idx >= self.theta.nspp() {
                continue;
            }

            let params: Vec<f64> = self.theta.matrix().row(*idx).iter().cloned().collect();
            let d_value = self.compute_d(&params, &pyl).unwrap_or(0.0);

            // Check if already elite
            let already_elite = self.elite_points.iter().any(|e| {
                e.params
                    .iter()
                    .zip(&params)
                    .all(|(a, b)| (a - b).abs() < THETA_D * 10.0)
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
        let mut injected = 0;
        for elite in &self.elite_points {
            if self.theta.check_point(&elite.params, THETA_D) {
                self.theta.add_point(&elite.params)?;
                injected += 1;
            }
        }

        if injected > 0 {
            tracing::debug!("Injected {} elite points", injected);
        }

        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    // SA INJECTION: Global Exploration via Simulated Annealing
    // ════════════════════════════════════════════════════════════════════

    /// Simulated annealing point injection for global mode discovery
    ///
    /// Uses Metropolis acceptance criterion with adaptive temperature control.
    fn sa_injection(&mut self) -> Result<()> {
        let pyl = self.compute_pyl();

        // Number of points to try scales with temperature
        let n_inject = (SA_INJECT_COUNT as f64 * (self.temperature / INITIAL_TEMPERATURE).sqrt())
            .ceil() as usize;
        let n_inject = n_inject.max(5);

        let mut accepted_points = 0;
        let mut max_d_found = f64::NEG_INFINITY;

        for _ in 0..n_inject * 10 {
            self.sa_proposed += 1;

            // Generate random point with boundary margin
            let point: Vec<f64> = self
                .ranges
                .iter()
                .map(|(lo, hi)| {
                    let margin = (hi - lo) * BOUNDARY_MARGIN;
                    self.rng.random_range((lo + margin)..(hi - margin))
                })
                .collect();

            // Compute D-criterion
            let d_value = match self.compute_d(&point, &pyl) {
                Ok(d) => d,
                Err(_) => continue,
            };
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
                    self.sa_accepted += 1;
                }
            }

            if accepted_points >= n_inject {
                break;
            }
        }

        tracing::debug!(
            "SA injection: {}/{} accepted, max_D = {:.4}, T = {:.4}",
            accepted_points,
            self.sa_proposed,
            max_d_found,
            self.temperature
        );

        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    // PHASE 2: SUBJECT-GUIDED EXPANSION (The Core Innovation)
    // ════════════════════════════════════════════════════════════════════

    /// Subject-Residual Driven Exploration
    ///
    /// Find subjects that are poorly explained by current mixture,
    /// then find parameters that would explain each one well.
    fn subject_guided_expansion(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();
        let n_subjects = pyl.len();

        // Identify "residual subjects" - those with low P(y|G)
        let mut indexed_pyl: Vec<(usize, f64)> = pyl.iter().cloned().enumerate().collect();
        indexed_pyl.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // Sort ascending

        // Take bottom fraction (poorly fit subjects)
        let n_residual = ((n_subjects as f64) * RESIDUAL_SUBJECT_FRACTION)
            .ceil()
            .max(MIN_RESIDUAL_SUBJECTS as f64) as usize;
        let n_residual = n_residual.min(n_subjects);

        let residual_subjects: Vec<usize> = indexed_pyl
            .iter()
            .take(n_residual)
            .map(|(idx, _)| *idx)
            .collect();

        tracing::debug!(
            "Subject-guided: {} residual subjects (of {}), P(y|G) range: {:.2e} to {:.2e}",
            residual_subjects.len(),
            n_subjects,
            indexed_pyl.first().map(|(_, p)| *p).unwrap_or(0.0),
            indexed_pyl
                .get(n_residual.saturating_sub(1))
                .map(|(_, p)| *p)
                .unwrap_or(0.0)
        );

        // For each residual subject, find MAP estimate: argmax P(y_i|θ)
        let subjects = self.data.subjects();
        let error_models = self.error_models.clone();

        let mut subject_map_points: Vec<(Vec<f64>, f64)> = Vec::new();

        for &subj_idx in &residual_subjects {
            // Create single-subject data for optimization
            let subject = &subjects[subj_idx];

            // Use centroid of current support points as starting guess
            let start = self.compute_weighted_centroid();

            // Find θ that maximizes P(y_i|θ) for this subject
            if let Ok(map_point) = self.find_subject_map(subject, &start, &error_models) {
                // Compute D-criterion for this point
                if let Ok(d) = self.compute_d(&map_point, &pyl) {
                    subject_map_points.push((map_point, d));
                }
            }
        }

        // Sort by D descending and add points that improve mixture
        subject_map_points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut added = 0;
        for (point, d) in subject_map_points {
            if d <= 0.0 {
                break; // No more improvements
            }

            if self.theta.check_point(&point, THETA_D) {
                self.theta.add_point(&point)?;
                added += 1;
            }
        }

        tracing::debug!("Subject-guided: added {} candidate points", added);

        Ok(())
    }

    /// Compute weighted centroid of current support points
    fn compute_weighted_centroid(&self) -> Vec<f64> {
        let n_params = self.ranges.len();
        let mut centroid = vec![0.0; n_params];
        let mut total_weight = 0.0;

        for (i, spp) in self.theta.matrix().row_iter().enumerate() {
            let weight = if i < self.w.len() { self.w[i] } else { 0.0 };
            total_weight += weight;
            for (j, val) in spp.iter().enumerate() {
                centroid[j] += weight * val;
            }
        }

        if total_weight > 0.0 {
            for c in &mut centroid {
                *c /= total_weight;
            }
        } else {
            // Fallback to center of ranges
            for (j, (lo, hi)) in self.ranges.iter().enumerate() {
                centroid[j] = (lo + hi) / 2.0;
            }
        }

        centroid
    }

    /// Find MAP estimate for a single subject: argmax P(y_i|θ)
    fn find_subject_map(
        &self,
        subject: &pharmsol::Subject,
        start: &[f64],
        error_models: &AssayErrorModels,
    ) -> Result<Vec<f64>, argmin::core::Error> {
        let optimizer = SubjectMapOptimizer {
            equation: &self.equation,
            subject,
            error_models,
            ranges: &self.ranges,
        };

        let simplex = create_initial_simplex(start, &self.ranges);
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(simplex).with_sd_tolerance(1e-3)?;

        let res = Executor::new(optimizer, solver)
            .configure(|state| state.max_iters(SUBJECT_MAP_MAX_ITERS))
            .run()?;

        Ok(res.state.best_param.unwrap())
    }

    // ════════════════════════════════════════════════════════════════════
    // PHASE 3: D-OPTIMAL REFINEMENT
    // ════════════════════════════════════════════════════════════════════

    /// D-optimal refinement with hierarchical iteration allocation
    fn d_optimal_refinement(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();
        let error_models = self.error_models.clone();
        let max_weight = self.w.iter().fold(f64::NEG_INFINITY, |a, b| a.max(b));

        // Collect points with weights - only refine points with meaningful weight
        let n_points = self.theta.nspp().min(self.w.len());
        let min_weight_threshold = max_weight * 0.001; // Skip points with < 0.1% of max weight

        let mut candidate_points: Vec<(Array1<f64>, f64)> = self
            .theta
            .matrix()
            .row_iter()
            .take(n_points)
            .enumerate()
            .filter(|(i, _)| self.w[*i] >= min_weight_threshold)
            .map(|(i, spp)| {
                let point: Vec<f64> = spp.iter().cloned().collect();
                let weight = self.w[i];
                (Array1::from(point), weight)
            })
            .collect();

        let ranges = self.ranges.clone();

        // Optimize with hierarchical iterations based on importance
        candidate_points.par_iter_mut().for_each(|(spp, weight)| {
            let importance = *weight / max_weight;
            let max_iters = if importance > HIGH_WEIGHT_THRESHOLD {
                DOPT_HIGH_WEIGHT_ITERS
            } else if importance > MED_WEIGHT_THRESHOLD {
                DOPT_MED_WEIGHT_ITERS
            } else {
                DOPT_LOW_WEIGHT_ITERS
            };

            let optimizer = DOptimalOptimizer {
                equation: &self.equation,
                data: &self.data,
                error_models: &error_models,
                pyl: &pyl,
            };

            if let Ok(refined) = optimizer.optimize(spp.to_vec(), max_iters) {
                // Clamp to safe boundaries
                let clamped: Array1<f64> = refined
                    .iter()
                    .zip(ranges.iter())
                    .map(|(&val, &(lo, hi))| {
                        let margin = (hi - lo) * BOUNDARY_MARGIN;
                        val.clamp(lo + margin, hi - margin)
                    })
                    .collect();
                *spp = clamped;
            }
        });

        // Add refined points
        for (cp, _) in candidate_points {
            self.theta.suggest_point(cp.to_vec().as_slice(), THETA_D)?;
        }

        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    // MULTI-SCALE GLOBAL OPTIMALITY
    // ════════════════════════════════════════════════════════════════════

    /// Sobol-based global optimality check with n samples
    fn sobol_global_check_n(&mut self, pyl: &Array1<f64>, n_samples: usize) -> Result<f64> {
        let n_dims = self.ranges.len();
        let mut max_d = f64::NEG_INFINITY;

        for i in 0..n_samples {
            let idx = self.sobol_index + i as u32;
            let mut point = Vec::with_capacity(n_dims);

            for dim in 0..n_dims {
                let sobol_val = sample(idx, dim as u32, SOBOL_SEED);
                let (lo, hi) = self.ranges[dim];
                let margin = (hi - lo) * BOUNDARY_MARGIN;
                point.push(lo + margin + sobol_val as f64 * (hi - lo - 2.0 * margin));
            }

            if let Ok(d) = self.compute_d(&point, pyl) {
                max_d = max_d.max(d);
            }
        }

        self.sobol_index += n_samples as u32;
        self.last_global_d_max = max_d;

        Ok(max_d)
    }

    /// Multi-scale global optimality check with injection of violating points
    fn multi_scale_global_check(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();
        let n_dims = self.ranges.len();
        let n_samples = GLOBAL_CHECK_SCALES[0]; // Use smallest scale for injection check

        let mut max_d = f64::NEG_INFINITY;
        let mut max_d_point = vec![0.0; n_dims];

        for i in 0..n_samples {
            let idx = self.sobol_index + i as u32;
            let mut point = Vec::with_capacity(n_dims);

            for dim in 0..n_dims {
                let sobol_val = sample(idx, dim as u32, SOBOL_SEED);
                let (lo, hi) = self.ranges[dim];
                let margin = (hi - lo) * BOUNDARY_MARGIN;
                point.push(lo + margin + sobol_val as f64 * (hi - lo - 2.0 * margin));
            }

            if let Ok(d) = self.compute_d(&point, &pyl) {
                if d > max_d {
                    max_d = d;
                    max_d_point = point;
                }
            }
        }

        self.sobol_index += n_samples as u32;

        // If we found a point with D > threshold, refine and inject it
        if max_d > GLOBAL_D_THRESHOLD {
            // Refine the point
            let optimizer = DOptimalOptimizer {
                equation: &self.equation,
                data: &self.data,
                error_models: &self.error_models,
                pyl: &pyl,
            };

            if let Ok(refined) = optimizer.optimize(max_d_point.clone(), 30) {
                // Clamp to safe bounds
                let clamped: Vec<f64> = refined
                    .iter()
                    .zip(self.ranges.iter())
                    .map(|(&val, &(lo, hi))| {
                        let margin = (hi - lo) * BOUNDARY_MARGIN;
                        val.clamp(lo + margin, hi - margin)
                    })
                    .collect();

                if let Ok(d_refined) = self.compute_d(&clamped, &pyl) {
                    if d_refined > GLOBAL_D_THRESHOLD * 0.5
                        && self.theta.check_point(&clamped, THETA_D)
                    {
                        self.theta.add_point(&clamped)?;
                        tracing::info!(
                            "Global check injected point with D = {:.4} (refined from {:.4})",
                            d_refined,
                            max_d
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// SUBJECT MAP OPTIMIZER
// ============================================================================

/// Optimizer for finding MAP estimate of single subject
struct SubjectMapOptimizer<'a, E: Equation> {
    equation: &'a E,
    subject: &'a pharmsol::Subject,
    error_models: &'a AssayErrorModels,
    ranges: &'a [(f64, f64)],
}

impl<E: Equation> CostFunction for SubjectMapOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        // Clamp to bounds
        let clamped: Vec<f64> = params
            .iter()
            .zip(self.ranges.iter())
            .map(|(v, (lo, hi))| v.clamp(*lo, *hi))
            .collect();

        // Create single-subject data
        let single_data = Data::new(vec![self.subject.clone()]);
        let theta = ndarray::Array1::from(clamped).insert_axis(Axis(0));

        let psi = pharmsol::prelude::simulator::log_likelihood_matrix(
            self.equation,
            &single_data,
            &theta,
            self.error_models,
            false,
        )?.mapv(f64::exp);

        // We want to MAXIMIZE P(y|θ), so minimize -P(y|θ)
        // Take log for numerical stability: minimize -log P(y|θ)
        let p = psi.iter().next().unwrap_or(&1e-300);
        let log_p = if *p > 0.0 { p.ln() } else { -700.0 }; // ln(1e-300) ≈ -690

        Ok(-log_p) // Minimize negative log-likelihood
    }
}

// ============================================================================
// D-OPTIMAL OPTIMIZER
// ============================================================================

/// Optimizer for D-criterion maximization
struct DOptimalOptimizer<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    error_models: &'a AssayErrorModels,
    pyl: &'a Array1<f64>,
}

impl<E: Equation> CostFunction for DOptimalOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, spp: &Self::Param) -> Result<Self::Output, Error> {
        let theta = Array1::from(spp.clone()).insert_axis(Axis(0));

        let psi = pharmsol::prelude::simulator::log_likelihood_matrix(
            self.equation,
            self.data,
            &theta,
            self.error_models,
            false,
        )?.mapv(f64::exp);

        let nsub = psi.nrows() as f64;
        let mut d_sum = -nsub;
        for (p_i, pyl_i) in psi.iter().zip(self.pyl.iter()) {
            if *pyl_i > 0.0 {
                d_sum += p_i / pyl_i;
            }
        }

        Ok(-d_sum) // Minimize -D = Maximize D
    }
}

impl<'a, E: Equation> DOptimalOptimizer<'a, E> {
    fn optimize(self, start: Vec<f64>, max_iters: u64) -> Result<Vec<f64>, Error> {
        let simplex = create_initial_simplex_simple(&start);
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(simplex).with_sd_tolerance(1e-3)?;

        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(max_iters))
            .run()?;

        Ok(res.state.best_param.unwrap())
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Create initial simplex with range-aware perturbation
fn create_initial_simplex(initial_point: &[f64], ranges: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let num_dims = initial_point.len();
    let perturbation_frac = 0.05; // 5% of range

    let mut vertices = Vec::new();
    vertices.push(initial_point.to_vec());

    for i in 0..num_dims {
        let (lo, hi) = ranges[i];
        let range = hi - lo;
        let perturbation = perturbation_frac * range;

        let mut perturbed = initial_point.to_vec();
        perturbed[i] = (perturbed[i] + perturbation).min(hi);
        vertices.push(perturbed);
    }

    vertices
}

/// Create initial simplex (simple version for D-optimal)
fn create_initial_simplex_simple(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let num_dims = initial_point.len();
    let perturbation_pct = 0.008;

    let mut vertices = Vec::new();
    vertices.push(initial_point.to_vec());

    for i in 0..num_dims {
        let perturbation = if initial_point[i] == 0.0 {
            0.00025
        } else {
            perturbation_pct * initial_point[i].abs()
        };

        let mut perturbed = initial_point.to_vec();
        perturbed[i] += perturbation;
        vertices.push(perturbed);
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
        assert_eq!(format!("{}", Phase::Expansion), "Expansion");
        assert_eq!(format!("{}", Phase::Convergence), "Convergence");
    }

    #[test]
    fn test_simplex_creation() {
        let point = vec![1.0, 2.0, 3.0];
        let ranges = vec![(0.0, 2.0), (0.0, 4.0), (0.0, 6.0)];
        let simplex = create_initial_simplex(&point, &ranges);

        assert_eq!(simplex.len(), 4); // n+1 vertices
        assert_eq!(simplex[0], point);
    }

    #[test]
    fn test_constants() {
        assert!(WARMUP_CYCLES > 0);
        assert!(RESIDUAL_SUBJECT_FRACTION > 0.0 && RESIDUAL_SUBJECT_FRACTION < 1.0);
        assert!(GLOBAL_D_THRESHOLD > 0.0);
        assert!(CONVERGENCE_WINDOW > 0);
        assert!(CE_ELITE_FRACTION > 0.0 && CE_ELITE_FRACTION < 1.0);
        assert!(CE_GMM_COMPONENTS >= 1);
    }

    #[test]
    fn test_gmm_component() {
        let ranges = vec![(0.0, 10.0), (0.0, 20.0)];
        let component = GaussianComponent::new(2, &ranges);

        assert_eq!(component.mean.len(), 2);
        assert_eq!(component.variance.len(), 2);
        assert!((component.mean[0] - 5.0).abs() < 0.01);
        assert!((component.mean[1] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_gmm_sampling() {
        let ranges = vec![(0.0, 10.0), (0.0, 20.0)];
        let mut rng = StdRng::seed_from_u64(42);
        let gmm = GMM::new(2, 2, &ranges, &mut rng);

        let samples = gmm.sample(100, &mut rng, &ranges);
        assert_eq!(samples.len(), 100);

        // Check samples are within bounds
        for sample in &samples {
            assert!(sample[0] >= 0.0 && sample[0] <= 10.0);
            assert!(sample[1] >= 0.0 && sample[1] <= 20.0);
        }
    }

    #[test]
    fn test_standard_normal() {
        let mut rng = StdRng::seed_from_u64(42);
        let samples: Vec<f64> = (0..1000)
            .map(|_| sample_standard_normal(&mut rng))
            .collect();

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        // Should be approximately N(0, 1)
        assert!(mean.abs() < 0.1);
        assert!((variance - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_global_scales() {
        // Verify scales are increasing
        for i in 1..GLOBAL_CHECK_SCALES.len() {
            assert!(GLOBAL_CHECK_SCALES[i] > GLOBAL_CHECK_SCALES[i - 1]);
        }
    }

    #[test]
    fn test_temperature_bounds() {
        assert!(INITIAL_TEMPERATURE > MIN_TEMPERATURE);
        assert!(BASE_COOLING_RATE > 0.0 && BASE_COOLING_RATE < 1.0);
        assert!(REHEAT_FACTOR > 1.0);
    }
}
