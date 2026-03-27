//! # NPOPT: Non-Parametric OPTimal Trajectory Algorithm
//!
//! A state-of-the-art hybrid algorithm combining the best elements from NPSAH, NPCAT, and NEXUS.
//!
//! ## Design Principles
//! 1. **Keep what works**: D-optimal refinement + Global optimality checks
//! 2. **Adaptive SA with reheat**: Prevents premature cooling, enables escape from local optima
//! 3. **Fisher-guided exploration**: Principled exploration in high-uncertainty directions
//! 4. **Simplified subject residual injection**: Targets missing modes directly
//! 5. **Elite preservation**: Prevents loss of good solutions during exploration
//!
//! ## Three-Phase Architecture
//!
//! ### Phase 1: Exploration (cycles 1-3)
//! - Stratified Sobol initialization for space-filling coverage
//! - Sparse adaptive grid expansion
//! - Track Fisher Information estimates
//!
//! ### Phase 2: Refinement (cycles 4+)
//! - Parallel D-optimal refinement (hierarchical iterations)
//! - Adaptive SA injection (with reheat mechanism)
//! - Fisher-guided expansion (high-variance directions only)
//! - Subject residual injection (top 3 poorly-fit subjects)
//! - Elite preservation (top 5 points)
//! - Periodic Sobol global check (every 3 cycles)
//!
//! ### Phase 3: Polishing (when global check passes)
//! - Full D-optimal refinement of all points
//! - No expansion
//! - Convergence when weights stable + P(Y|L) criterion met

mod constants;
mod convergence;
mod expansion;
mod optimizers;

pub use constants::*;

use crate::algorithms::{
    NativeNonparametricConfig, NonparametricAlgorithmInput, Status, StopReason,
};
use crate::estimation::nonparametric::ipm::burke;
use crate::estimation::nonparametric::qr;
use crate::estimation::nonparametric::sample_space_for_parameters;
use crate::estimation::nonparametric::{
    calculate_psi, CycleLog, NPCycle, NonparametricWorkspace, Psi, Theta, Weights,
};
use crate::prelude::algorithms::Algorithms;

use anyhow::{bail, Result};
use ndarray::Array1;
use pharmsol::prelude::AssayErrorModel;
use pharmsol::prelude::{
    data::{AssayErrorModels, Data},
    simulator::Equation,
};
use rand::prelude::*;
use std::collections::VecDeque;

// ============================================================================
// PHASE ENUM
// ============================================================================

/// Algorithm phase for NPOPT
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    /// Initial exploration with Sobol + grid
    Exploration,
    /// Balanced refinement with D-optimal, SA, Fisher
    Refinement,
    /// Final polishing, no expansion
    Polishing,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Phase::Exploration => write!(f, "Exploration"),
            Phase::Refinement => write!(f, "Refinement"),
            Phase::Polishing => write!(f, "Polishing"),
        }
    }
}

// ============================================================================
// ELITE POINT
// ============================================================================

/// An elite point preserved across cycles
#[derive(Debug, Clone)]
pub struct ElitePoint {
    pub params: Vec<f64>,
    pub d_value: f64,
    pub cycle_added: usize,
}

// ============================================================================
// NPOPT STRUCT
// ============================================================================

/// NPOPT: Non-Parametric OPTimal Trajectory Algorithm
#[derive(Debug)]
pub struct NPOPT<E: Equation + Send + 'static> {
    /// The pharmacometric equation/model
    pub(crate) equation: E,
    /// Parameter ranges for each dimension
    pub(crate) ranges: Vec<(f64, f64)>,
    /// Probability matrix: P(y_i | θ_j)
    pub(crate) psi: Psi,
    /// Support points (parameter values)
    pub(crate) theta: Theta,
    /// Weights from IPM before condensation
    pub(crate) lambda: Weights,
    /// Final weights after condensation
    pub(crate) w: Weights,
    /// Previous weights for stability check
    pub(crate) w_prev: Weights,
    /// Current grid spacing
    pub(crate) eps: f64,
    /// Previous objective function value
    pub(crate) last_objf: f64,
    /// Current objective function value
    pub(crate) objf: f64,
    /// Best objective function seen
    pub(crate) best_objf: f64,
    /// P(Y|L) values for convergence checking
    pub(crate) f0: f64,
    pub(crate) f1: f64,
    /// Current cycle number
    pub(crate) cycle: usize,
    /// Step sizes for error model optimization
    pub(crate) gamma_delta: Vec<f64>,
    /// Error models for observations
    pub(crate) error_models: AssayErrorModels,
    /// Algorithm status
    pub(crate) status: Status,
    /// Cycle log for tracking progress
    pub(crate) cycle_log: CycleLog,
    /// Subject data
    pub(crate) data: Data,
    /// Unified runtime/model-derived configuration
    pub(crate) config: NativeNonparametricConfig,

    // NPOPT specific fields
    /// Current algorithm phase
    pub(crate) phase: Phase,
    /// History of objective function values
    pub(crate) objf_history: Vec<f64>,
    /// Sobol sequence index
    pub(crate) sobol_index: u32,

    // Adaptive SA fields
    /// SA temperature
    pub(crate) temperature: f64,
    /// Effective cooling rate (adaptive)
    pub(crate) cooling_rate: f64,
    /// Rolling window of acceptance ratios
    pub(crate) sa_acceptance_history: VecDeque<f64>,
    /// SA accepted count this cycle
    pub(crate) sa_accepted: usize,
    /// SA proposed count this cycle
    pub(crate) sa_proposed: usize,

    // Fisher Information
    /// Diagonal approximation of Fisher Information
    pub(crate) fisher_diagonal: Vec<f64>,

    // Elite preservation
    /// Elite points preserved across cycles
    pub(crate) elite_points: Vec<ElitePoint>,

    // Convergence tracking
    /// Count of consecutive global check passes
    pub(crate) global_check_passes: usize,
    /// Last global check max D value
    pub(crate) last_global_d_max: f64,

    /// Random number generator
    pub(crate) rng: StdRng,
}

// ============================================================================
// ALGORITHMS TRAIT IMPLEMENTATION
// ============================================================================

impl<E: Equation + Send + 'static> Algorithms<E> for NPOPT<E> {
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

        // Phase transitions
        if self.cycle > EXPLORATION_CYCLES && self.phase == Phase::Exploration {
            self.phase = Phase::Refinement;
            tracing::info!(
                "NPOPT: Exploration → Refinement (cycle {}, {} SPPs)",
                self.cycle,
                self.theta.nspp()
            );
        }

        // Adapt temperature
        self.adapt_temperature();

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
            "Support points: {} | Phase: {} | T: {:.4}",
            self.theta.nspp(),
            self.phase,
            self.temperature
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
            tracing::info!("NPOPT converged after {} cycles", self.cycle);
            self.set_status(Status::Stop(StopReason::Converged));
            self.log_cycle_state();
            return Ok(self.status().clone());
        }

        // NPAG-style eps convergence
        if self.phase != Phase::Polishing {
            if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > MIN_EPS {
                self.eps /= 2.0;
                tracing::debug!("Halving eps to {:.6}", self.eps);

                if self.eps <= MIN_EPS {
                    let pyl = self.psi.matrix() * self.w.weights();
                    self.f1 = pyl.iter().map(|x| x.ln()).sum();
                    if (self.f1 - self.f0).abs() <= THETA_F {
                        // Transition to polishing
                        self.phase = Phase::Polishing;
                        tracing::info!("NPOPT: Refinement → Polishing (cycle {})", self.cycle);
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
        // Store previous weights
        self.w_prev = self.w.clone();

        // Lambda filter
        let max_lambda = self
            .lambda
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        let threshold = max_lambda / LAMBDA_FILTER_DIVISOR;
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

        // Update Fisher Information and elite points
        self.update_fisher_information();
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
            Phase::Exploration => self.exploration_expansion()?,
            Phase::Refinement => self.refinement_expansion()?,
            Phase::Polishing => self.polishing_expansion()?,
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
// HELPER METHODS
// ============================================================================

impl<E: Equation + Send + 'static> NPOPT<E> {
    pub(crate) fn from_input(input: NonparametricAlgorithmInput<E>) -> Result<Box<Self>> {
        let config = input.native_config()?;
        let seed = config.prior.seed().unwrap_or(42);
        let n_params = config.ranges.len();
        let error_models = input.error_models().clone();

        Ok(Box::new(Self {
            equation: input.equation,
            ranges: config.ranges.clone(),
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
            phase: Phase::Exploration,
            objf_history: Vec::with_capacity(500),
            sobol_index: seed as u32,
            temperature: INITIAL_TEMPERATURE,
            cooling_rate: BASE_COOLING_RATE,
            sa_acceptance_history: VecDeque::with_capacity(SA_HISTORY_WINDOW),
            sa_accepted: 0,
            sa_proposed: 0,
            fisher_diagonal: vec![1.0; n_params],
            elite_points: Vec::with_capacity(ELITE_COUNT * 2),
            global_check_passes: 0,
            last_global_d_max: f64::INFINITY,
            rng: StdRng::seed_from_u64(seed as u64),
        }))
    }

    /// Compute P(Y|G) = Psi * w
    pub(crate) fn compute_pyl(&self) -> Array1<f64> {
        let psi = self.psi.to_ndarray();
        let w: Array1<f64> = self.w.clone().iter().collect();
        psi.dot(&w)
    }

    /// Compute D-criterion for a candidate point
    pub(crate) fn compute_d(&self, point: &[f64], pyl: &Array1<f64>) -> Result<f64> {
        let theta_single = ndarray::Array1::from(point.to_vec()).insert_axis(ndarray::Axis(0));

        let psi_single = pharmsol::prelude::simulator::log_likelihood_matrix(
            &self.equation,
            &self.data,
            &theta_single,
            &self.error_models,
            false,
        )?
        .mapv(f64::exp);

        let nsub = psi_single.nrows() as f64;
        let mut d_sum = -nsub;

        for (p_i, pyl_i) in psi_single.iter().zip(pyl.iter()) {
            if *pyl_i > 0.0 {
                d_sum += p_i / pyl_i;
            }
        }

        Ok(d_sum)
    }

    /// Compute weighted centroid of support points
    pub(crate) fn compute_weighted_centroid(&self) -> Vec<f64> {
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
            for (j, (lo, hi)) in self.ranges.iter().enumerate() {
                centroid[j] = (lo + hi) / 2.0;
            }
        }

        centroid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::Exploration), "Exploration");
        assert_eq!(format!("{}", Phase::Refinement), "Refinement");
        assert_eq!(format!("{}", Phase::Polishing), "Polishing");
    }

    #[test]
    fn test_constants() {
        assert!(EXPLORATION_CYCLES > 0);
        assert!(INITIAL_TEMPERATURE > MIN_TEMPERATURE);
        assert!(BASE_COOLING_RATE > 0.0 && BASE_COOLING_RATE < 1.0);
        assert!(ELITE_COUNT > 0);
    }
}
