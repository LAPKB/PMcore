//! # NPPSO: Non-Parametric Particle Swarm Optimization
//!
//! A true PSO-based algorithm for non-parametric population modeling.
//!
//! ## Key Innovation: D-Criterion Guided Swarm + Subject Targeting
//!
//! Unlike standard PSO which optimizes a single objective, NPPSO particles
//! search for regions of parameter space that maximize the D-optimality criterion.
//! Additionally, we target poorly-fit subjects with MAP estimation to ensure
//! all subjects are well-represented.
//!
//! ## Why This Works
//!
//! 1. **Momentum escapes local optima**: Velocity-based movement allows particles
//!    to overshoot and explore beyond current best positions
//! 2. **Collective learning**: The swarm shares information about high-D regions
//! 3. **Subject targeting**: MAP injection for poorly-fit subjects ensures coverage
//! 4. **D-optimal refinement**: COBYLA polishes support point positions
//! 5. **Elite preservation**: Best points are preserved across cycles
//!
//! ## Algorithm Structure
//!
//! - **Warm-up (cycles 1-3)**: NPAG-style grid expansion for broad coverage
//! - **PSO Phase**: Particles search for high D-criterion regions
//! - **Subject MAP**: Inject points for poorly-fit subjects
//! - **D-refinement**: Polish existing support points with COBYLA
//! - **Elite preservation**: Maintain best points across cycles

mod constants;
mod optimizers;
mod swarm;

pub use constants::*;
use optimizers::{optimize_subject_map, refine_d_optimal, ElitePoint};

use crate::algorithms::StopReason;
use crate::routines::expansion::adaptative_grid::adaptative_grid;
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
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::{Array, Array1, ArrayBase, Axis, Dim, OwnedRepr};
use pharmsol::prelude::data::Data;
use pharmsol::prelude::simulator::Equation;
use pharmsol::{prelude::ErrorModel, ErrorModels, Subject};
use rand::prelude::*;
use rand::SeedableRng;
use swarm::Swarm;

// ============================================================================
// NPPSO STRUCT
// ============================================================================

pub struct NPPSO<E: Equation + Send + 'static> {
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
    error_models: ErrorModels,
    status: Status,
    cycle_log: CycleLog,
    data: Data,
    settings: Settings,

    // PSO specific
    swarm: Swarm,
    objf_history: Vec<f64>,
    rng: StdRng,
    /// Cached pyl vector for D-criterion evaluation
    pyl: Array1<f64>,
    /// Phase: true = warm-up (grid expansion), false = PSO-driven
    in_warmup: bool,
    /// Elite points preserved across cycles
    elite_points: Vec<ElitePoint>,
    /// SA temperature for escaping local optima
    sa_temperature: f64,
    /// Track stagnation for SA reheat
    stagnation_count: usize,
}

// ============================================================================
// ALGORITHMS TRAIT
// ============================================================================

impl<E: Equation + Send + 'static> Algorithms<E> for NPPSO<E> {
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>> {
        let seed = settings.prior().seed().unwrap_or(42) as u64;
        let ranges = settings.parameters().ranges();
        let n_dims = ranges.len();
        let n_subjects = data.len();

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
            data,
            settings,
            // PSO
            swarm: Swarm::new(n_dims, &ranges, seed),
            objf_history: Vec::with_capacity(500),
            rng: StdRng::seed_from_u64(seed),
            pyl: Array1::ones(n_subjects), // Initialize with uniform
            in_warmup: true,
            elite_points: Vec::with_capacity(ELITE_COUNT),
            sa_temperature: SA_INITIAL_TEMP,
            stagnation_count: 0,
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

        // Exit warm-up after WARMUP_CYCLES
        if self.cycle > WARMUP_CYCLES && self.in_warmup {
            self.in_warmup = false;
            tracing::info!("NPPSO: Warm-up complete, entering PSO-driven expansion");
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
        tracing::debug!(
            "Support points: {} | Phase: {} | EPS: {:.4}",
            self.theta.nspp(),
            if self.in_warmup { "Warm-up" } else { "PSO" },
            self.eps
        );

        self.error_models.iter().for_each(|(outeq, em)| {
            if ErrorModel::None != *em {
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

        // Track stagnation for SA reheat
        if (self.last_objf - self.objf).abs() < 1e-3 {
            self.stagnation_count += 1;
            // Reheat SA if stagnating and temperature is low
            if self.stagnation_count > 5 && self.sa_temperature < SA_INITIAL_TEMP * 0.3 {
                self.sa_temperature = (self.sa_temperature * SA_REHEAT_FACTOR).min(SA_INITIAL_TEMP);
                self.stagnation_count = 0;
                tracing::debug!("SA reheat to T={:.3}", self.sa_temperature);
            }
        } else {
            self.stagnation_count = 0;
        }

        // NPAG-style convergence with eps halving
        let psi = self.psi.matrix();
        let w = &self.w;

        if (self.last_objf - self.objf).abs() <= THETA_G && self.eps > THETA_E {
            self.eps /= 2.0;
            tracing::debug!("Halving eps to {:.6}", self.eps);

            if self.eps <= THETA_E {
                let pyl = psi * w.weights();
                self.f1 = pyl.iter().map(|x| x.ln()).sum();

                if (self.f1 - self.f0).abs() <= THETA_F {
                    // Also check global optimality via swarm
                    let global_check = self.global_optimality_check()?;
                    if global_check {
                        tracing::info!("NPPSO converged after {} cycles", self.cycle);
                        self.status = Status::Stop(StopReason::Converged);
                        self.log_cycle_state();
                        return Ok(self.status.clone());
                    } else {
                        tracing::debug!("P(Y|L) criterion met but global check failed, continuing");
                        self.f0 = self.f1;
                        self.eps = 0.2;
                    }
                } else {
                    self.f0 = self.f1;
                    self.eps = 0.2;
                }
            }
        }

        // Max cycles check
        if self.cycle >= self.settings.config().cycles {
            tracing::warn!("Maximum cycles reached");
            self.status = Status::Stop(StopReason::MaxCycles);
            self.log_cycle_state();
            return Ok(self.status.clone());
        }

        // Stop file check
        if std::path::Path::new("stop").exists() {
            tracing::warn!("Stop file detected");
            self.status = Status::Stop(StopReason::Stopped);
            self.log_cycle_state();
            return Ok(self.status.clone());
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

        let (lambda, _) = burke(&self.psi)?;
        self.lambda = lambda;

        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        // Lambda threshold pruning (more aggressive: 1/10000)
        let max_lambda = self
            .lambda
            .iter()
            .fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

        let threshold = max_lambda / 10000.0;
        let mut keep: Vec<usize> = self
            .lambda
            .iter()
            .enumerate()
            .filter(|(_, lam)| *lam > threshold)
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

        // QR rank-revealing factorization
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

        self.validate_psi()?;

        let (lambda, objf) = burke(&self.psi)?;
        self.lambda = lambda;
        self.objf = objf;
        self.w = self.lambda.clone();

        // Update pyl for D-criterion calculations
        let psi = self.psi.matrix().as_ref().into_ndarray();
        let w_arr: Array1<f64> = self.w.iter().collect();
        self.pyl = psi.dot(&w_arr);

        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        // Update swarm with D-criterion fitness
        self.update_swarm_fitness()?;

        // Standard error model optimization
        self.optimize_error_models()?;

        Ok(())
    }

    fn expansion(&mut self) -> Result<()> {
        if self.in_warmup {
            // Warm-up: NPAG-style grid expansion for broad coverage
            self.warmup_expansion()?;
        } else {
            // PSO-driven expansion
            self.pso_expansion()?;
        }

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
// NPPSO SPECIFIC METHODS
// ============================================================================

impl<E: Equation + Send + 'static> NPPSO<E> {
    /// Warm-up expansion using adaptive grid (like NPAG)
    fn warmup_expansion(&mut self) -> Result<()> {
        tracing::debug!("NPPSO warm-up: adaptive grid expansion");
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D)?;
        Ok(())
    }

    /// PSO-driven expansion: particles search for high-D regions
    fn pso_expansion(&mut self) -> Result<()> {
        let initial_points = self.theta.nspp();

        // 1. Evaluate D-criterion for all particles (in parallel)
        let particle_fitness = self.evaluate_particle_fitness()?;

        // 2. Update particle personal bests
        self.swarm.update_personal_bests(&particle_fitness);

        // 3. PSO velocity/position update
        let inertia = self.adaptive_inertia();
        self.swarm.update_all(
            inertia,
            COGNITIVE_WEIGHT,
            SOCIAL_WEIGHT,
            &self.ranges,
            &mut self.rng,
        );

        // 4. Add high-fitness particles as support point candidates
        let max_fitness = particle_fitness
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_fitness * D_THRESHOLD_FRACTION;

        let mut added = 0;
        for (i, particle) in self.swarm.particles().iter().enumerate() {
            if particle_fitness[i] > threshold.max(0.0) {
                if self.theta.check_point(&particle.position, THETA_D) {
                    self.theta.add_point(&particle.position)?;
                    added += 1;
                }
            }
        }

        // 5. Also add from personal bests (memory of good regions)
        for particle in self.swarm.particles() {
            if particle.pbest_fitness > threshold.max(0.0) {
                if self.theta.check_point(&particle.pbest_position, THETA_D) {
                    self.theta.add_point(&particle.pbest_position)?;
                    added += 1;
                }
            }
        }

        // 6. Simulated Annealing injection (key for escaping local optima!)
        self.sa_injection()?;

        // 7. Subject MAP injection for poorly-fit subjects
        self.inject_subject_maps()?;

        // 8. D-optimal refinement (periodically)
        if self.cycle % DOPT_REFINE_INTERVAL == 0 && self.cycle > WARMUP_CYCLES {
            self.d_optimal_refinement()?;
        }

        // 9. Inject elite points
        self.inject_elite_points()?;

        // 10. Sparse grid expansion to fill gaps every few cycles
        if self.cycle % 3 == 0 {
            let sparse_eps = self.eps * 0.5;
            adaptative_grid(&mut self.theta, sparse_eps, &self.ranges, THETA_D * 2.0)?;
        }

        // 11. Reinject diversity if swarm is converging
        if self.swarm_convergence_ratio() > CONVERGENCE_THRESHOLD {
            let n_reinject = (SWARM_SIZE as f64 * REINJECT_FRACTION) as usize;
            self.swarm
                .reinject_random(&self.ranges, &mut self.rng, n_reinject);
            tracing::debug!("Swarm converging, reinjected {} particles", n_reinject);
        }

        // 12. Cool SA temperature
        self.sa_temperature = (self.sa_temperature * SA_COOLING_RATE).max(SA_MIN_TEMP);

        tracing::debug!(
            "PSO expansion: {} → {} (added {}), SA temp={:.3}",
            initial_points,
            self.theta.nspp(),
            added,
            self.sa_temperature
        );

        Ok(())
    }

    // ========================================================================
    // SIMULATED ANNEALING INJECTION
    // ========================================================================

    /// SA-style random injection with Metropolis acceptance
    /// This is KEY for escaping local optima that PSO can't escape
    fn sa_injection(&mut self) -> Result<()> {
        if self.sa_temperature < SA_MIN_TEMP {
            return Ok(());
        }

        let mut accepted = 0;
        let mut proposed = 0;

        for _ in 0..SA_INJECT_COUNT * 3 {
            proposed += 1;

            // Generate random point with margin from boundaries
            let point: Vec<f64> = self
                .ranges
                .iter()
                .map(|(lo, hi)| {
                    let margin = (hi - lo) * BOUNDARY_MARGIN;
                    self.rng.random_range((lo + margin)..(hi - margin))
                })
                .collect();

            // Compute D-criterion for this point
            let d = match self.compute_d_criterion(&point) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Metropolis acceptance: always accept if D > 0,
            // probabilistically accept if D < 0 (allows exploration)
            let accept = if d > 0.0 {
                true
            } else {
                let p_accept = (d / self.sa_temperature).exp();
                self.rng.random::<f64>() < p_accept
            };

            if accept && self.theta.check_point(&point, THETA_D) {
                self.theta.add_point(&point)?;
                accepted += 1;
            }

            if accepted >= SA_INJECT_COUNT {
                break;
            }
        }

        if accepted > 0 {
            tracing::debug!(
                "SA injection: {}/{} accepted (T={:.3})",
                accepted,
                proposed,
                self.sa_temperature
            );
        }

        Ok(())
    }

    // ========================================================================
    // SUBJECT MAP INJECTION
    // ========================================================================

    /// Inject support points for poorly-fit subjects
    fn inject_subject_maps(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let n_subjects = self.pyl.len();

        // Find worst-fit subjects (lowest P(y|G))
        let mut indexed_pyl: Vec<(usize, f64)> = self.pyl.iter().cloned().enumerate().collect();
        indexed_pyl.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n_residual = RESIDUAL_SUBJECTS.min(n_subjects);
        let subjects = self.data.subjects();

        let mut added = 0;

        for (subj_idx, pyl_val) in indexed_pyl.iter().take(n_residual) {
            // Skip if already well-fit
            if *pyl_val > 0.1 {
                continue;
            }

            let subject = &subjects[*subj_idx];

            // Start from weighted centroid
            let start = self.compute_weighted_centroid();

            // Find MAP estimate for this subject
            if let Ok(map_point) = optimize_subject_map(
                &self.equation,
                subject,
                &self.error_models,
                &self.ranges,
                &start,
                SUBJECT_MAP_EVALS,
            ) {
                // Check if point improves D-criterion
                let d = self.compute_d_criterion(&map_point)?;
                if d > 0.0 && self.theta.check_point(&map_point, THETA_D) {
                    self.theta.add_point(&map_point)?;
                    added += 1;
                    tracing::debug!(
                        "Subject {} MAP: pyl={:.4}, D={:.4}",
                        subject.id(),
                        pyl_val,
                        d
                    );
                }
            }
        }

        if added > 0 {
            tracing::debug!("Subject MAP injection: added {} points", added);
        }
        Ok(())
    }

    // ========================================================================
    // D-OPTIMAL REFINEMENT
    // ========================================================================

    /// Refine existing support points using COBYLA
    fn d_optimal_refinement(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let max_weight = self.w.iter().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let threshold = max_weight * DOPT_WEIGHT_THRESHOLD;

        let n_points = self.theta.nspp().min(self.w.len());
        let mut refined_count = 0;

        for i in 0..n_points {
            // Only refine important points
            if self.w[i] < threshold {
                continue;
            }

            let start: Vec<f64> = self.theta.matrix().row(i).iter().cloned().collect();

            if let Ok(refined) = refine_d_optimal(
                &self.equation,
                &self.data,
                &self.error_models,
                &self.pyl,
                &self.ranges,
                &start,
                DOPT_REFINE_EVALS,
            ) {
                // Check if refinement improves D
                let d_old = self.compute_d_criterion(&start)?;
                let d_new = self.compute_d_criterion(&refined)?;

                if d_new > d_old && self.theta.check_point(&refined, THETA_D) {
                    self.theta.add_point(&refined)?;
                    refined_count += 1;
                }
            }
        }

        if refined_count > 0 {
            tracing::debug!(
                "D-optimal refinement: added {} refined points",
                refined_count
            );
        }
        Ok(())
    }

    // ========================================================================
    // ELITE PRESERVATION
    // ========================================================================

    /// Update and inject elite points
    fn inject_elite_points(&mut self) -> Result<()> {
        // Age existing elite points
        for elite in &mut self.elite_points {
            elite.cycle_added += 1;
        }

        // Remove old elite points
        self.elite_points
            .retain(|e| self.cycle.saturating_sub(e.cycle_added) < ELITE_MAX_AGE);

        // Find current top points by weight
        if self.w.len() > 0 {
            let n_spp = self.theta.nspp().min(self.w.len());
            let mut indexed_weights: Vec<(usize, f64)> =
                self.w.iter().enumerate().take(n_spp).collect();
            indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (idx, _weight) in indexed_weights.iter().take(ELITE_COUNT) {
                if *idx >= self.theta.nspp() {
                    continue;
                }

                let params: Vec<f64> = self.theta.matrix().row(*idx).iter().cloned().collect();
                let d_value = self.compute_d_criterion(&params).unwrap_or(0.0);

                // Check if already elite (within distance threshold)
                let already_elite = self.elite_points.iter().any(|e| {
                    e.params
                        .iter()
                        .zip(params.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>()
                        < THETA_D * 10.0
                });

                if !already_elite && self.elite_points.len() < ELITE_COUNT * 2 {
                    self.elite_points.push(ElitePoint {
                        params: params.clone(),
                        d_value,
                        cycle_added: self.cycle,
                    });
                }
            }
        }

        // Re-inject elite points
        let mut added = 0;
        for elite in &self.elite_points {
            if self.theta.check_point(&elite.params, THETA_D) {
                self.theta.add_point(&elite.params)?;
                added += 1;
            }
        }

        if added > 0 {
            tracing::debug!("Elite injection: {} points", added);
        }
        Ok(())
    }

    /// Compute weighted centroid of current support points
    fn compute_weighted_centroid(&self) -> Vec<f64> {
        let n_dims = self.ranges.len();
        let mut centroid = vec![0.0; n_dims];
        let mut total_weight = 0.0;

        let n_points = self.theta.nspp().min(self.w.len());
        for i in 0..n_points {
            let w = self.w[i];
            total_weight += w;
            for d in 0..n_dims {
                centroid[d] += w * self.theta.matrix().get(i, d);
            }
        }

        if total_weight > 0.0 {
            for d in 0..n_dims {
                centroid[d] /= total_weight;
            }
        } else {
            // Fallback: center of ranges
            for d in 0..n_dims {
                centroid[d] = (self.ranges[d].0 + self.ranges[d].1) / 2.0;
            }
        }

        centroid
    }

    /// Evaluate D-criterion for all particles in parallel
    fn evaluate_particle_fitness(&self) -> Result<Vec<f64>> {
        let positions: Vec<Vec<f64>> = self
            .swarm
            .particles()
            .iter()
            .map(|p| p.position.clone())
            .collect();

        let fitness: Vec<f64> = positions
            .into_par_iter()
            .map(|pos| self.compute_d_criterion(&pos).unwrap_or(f64::NEG_INFINITY))
            .collect();

        Ok(fitness)
    }

    /// Compute D-criterion for a single point
    fn compute_d_criterion(&self, point: &[f64]) -> Result<f64> {
        let theta_single = Array1::from(point.to_vec()).insert_axis(Axis(0));

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

        for (p_i, pyl_i) in psi_single.iter().zip(self.pyl.iter()) {
            if *pyl_i > 1e-300 {
                d_sum += p_i / pyl_i;
            }
        }

        Ok(d_sum)
    }

    /// Update swarm fitness based on D-criterion
    fn update_swarm_fitness(&mut self) -> Result<()> {
        let particle_fitness = self.evaluate_particle_fitness()?;

        // Update personal bests
        self.swarm.update_personal_bests(&particle_fitness);

        // Find and update global best
        if let Some((best_idx, best_fitness)) = particle_fitness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            let best_pos = self.swarm.particles()[best_idx].position.clone();
            self.swarm.update_global_best(&best_pos, *best_fitness);

            tracing::debug!(
                "Swarm update: best D = {:.4}, gbest D = {:.4}",
                best_fitness,
                self.swarm.gbest_fitness()
            );
        }

        Ok(())
    }

    /// Compute swarm convergence ratio (how clustered the particles are)
    fn swarm_convergence_ratio(&self) -> f64 {
        let positions = self.swarm.get_positions();
        if positions.is_empty() {
            return 0.0;
        }

        let n_dims = self.ranges.len();
        let n_particles = positions.len();

        // Compute centroid
        let mut centroid = vec![0.0; n_dims];
        for pos in &positions {
            for (j, val) in pos.iter().enumerate() {
                centroid[j] += val;
            }
        }
        for c in &mut centroid {
            *c /= n_particles as f64;
        }

        // Compute average normalized distance from centroid
        let mut total_dist = 0.0;
        for pos in &positions {
            let mut dist = 0.0;
            for (j, val) in pos.iter().enumerate() {
                let range = self.ranges[j].1 - self.ranges[j].0;
                let normalized = (val - centroid[j]) / range;
                dist += normalized * normalized;
            }
            total_dist += dist.sqrt();
        }

        let avg_dist = total_dist / n_particles as f64;

        // Return inverse: high value means converged (clustered)
        1.0 / (1.0 + avg_dist * 10.0)
    }

    /// Adaptive inertia based on improvement rate
    fn adaptive_inertia(&self) -> f64 {
        if self.objf_history.len() < 3 {
            return INERTIA_MAX;
        }

        let recent: Vec<f64> = self.objf_history.iter().rev().take(5).copied().collect();
        let improvement = if recent.len() >= 2 {
            (recent[0] - recent[recent.len() - 1]).abs()
        } else {
            1.0
        };

        // High improvement → high inertia (explore)
        // Low improvement → low inertia (exploit)
        if improvement > 1.0 {
            INERTIA_MAX
        } else if improvement > 0.1 {
            (INERTIA_MAX + INERTIA_MIN) / 2.0
        } else {
            INERTIA_MIN
        }
    }

    /// Global optimality check using swarm exploration
    fn global_optimality_check(&mut self) -> Result<bool> {
        // Sample random points and check if any have high D-criterion
        let n_samples = GLOBAL_CHECK_SAMPLES;
        let mut max_d = f64::NEG_INFINITY;

        for _ in 0..n_samples {
            let point: Vec<f64> = self
                .ranges
                .iter()
                .map(|(lo, hi)| self.rng.random_range(*lo..*hi))
                .collect();

            let d = self.compute_d_criterion(&point)?;
            max_d = max_d.max(d);
        }

        // Also check current particle positions
        let particle_fitness = self.evaluate_particle_fitness()?;
        let max_particle_d = particle_fitness
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        max_d = max_d.max(max_particle_d);

        let passed = max_d < GLOBAL_D_THRESHOLD;
        tracing::debug!(
            "Global optimality check: max_D = {:.4}, threshold = {:.4}, passed = {}",
            max_d,
            GLOBAL_D_THRESHOLD,
            passed
        );

        Ok(passed)
    }

    /// Optimize error models (standard approach)
    fn optimize_error_models(&mut self) -> Result<()> {
        for (outeq, em) in self.error_models.clone().iter_mut() {
            if *em == ErrorModel::None || em.is_factor_fixed().unwrap_or(true) {
                continue;
            }

            let gamma_up = em.factor()? * (1.0 + self.gamma_delta[outeq]);
            let gamma_down = em.factor()? / (1.0 + self.gamma_delta[outeq]);

            let mut em_up = self.error_models.clone();
            em_up.set_factor(outeq, gamma_up)?;

            let mut em_down = self.error_models.clone();
            em_down.set_factor(outeq, gamma_down)?;

            let psi_up =
                calculate_psi(&self.equation, &self.data, &self.theta, &em_up, false, true)?;
            let psi_down = calculate_psi(
                &self.equation,
                &self.data,
                &self.theta,
                &em_down,
                false,
                true,
            )?;

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

        // Update pyl after error model changes
        if self.w.len() > 0 {
            let psi = self.psi.matrix().as_ref().into_ndarray();
            let w_arr: Array1<f64> = self.w.iter().collect();
            self.pyl = psi.dot(&w_arr);
        }

        Ok(())
    }

    /// Validate PSI matrix
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
            let bad_subjects: Vec<&String> =
                bad_indices.iter().map(|&i| subjects[i].id()).collect();
            bail!("Zero probability for subjects: {:?}", bad_subjects);
        }

        Ok(())
    }
}
