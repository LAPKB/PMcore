//! Expansion strategies for NPOPT

use super::constants::*;
use super::{ElitePoint, NPOPT};
use crate::routines::expansion::adaptative_grid::adaptative_grid;

use anyhow::Result;
use ndarray::parallel::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use ndarray::Array1;
use pharmsol::prelude::simulator::Equation;
use rand::prelude::*;
use sobol_burley::sample;

impl<E: Equation + Send + 'static> NPOPT<E> {
    // ========================================================================
    // PHASE-SPECIFIC EXPANSION
    // ========================================================================

    /// Exploration phase: Sobol initialization + grid expansion
    pub(crate) fn exploration_expansion(&mut self) -> Result<()> {
        tracing::debug!("Exploration expansion: Sobol + adaptive grid");

        // Stratified Sobol for initial coverage
        self.sobol_initialization(SOBOL_INIT_SAMPLES)?;

        // Adaptive grid expansion
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D)?;

        Ok(())
    }

    /// Refinement phase: D-optimal + SA + Fisher + Subject residual + Elite
    pub(crate) fn refinement_expansion(&mut self) -> Result<()> {
        let initial = self.theta.nspp();

        // 1. D-optimal refinement (parallel, hierarchical)
        self.d_optimal_refinement()?;
        let after_dopt = self.theta.nspp();

        // 2. Adaptive SA injection
        if self.temperature > MIN_TEMPERATURE {
            self.adaptive_sa_injection()?;
        }
        let after_sa = self.theta.nspp();

        // 3. Fisher-guided expansion
        self.fisher_expansion()?;
        let after_fisher = self.theta.nspp();

        // 4. Subject residual injection
        self.inject_residual_subjects()?;
        let after_subj = self.theta.nspp();

        // 5. Re-inject elite points
        self.inject_elite_points()?;
        let after_elite = self.theta.nspp();

        // 6. Periodic global check
        if self.cycle % GLOBAL_CHECK_INTERVAL == 0 {
            self.sobol_global_check()?;
        }

        tracing::debug!(
            "Refinement: {} → {} (D-opt) → {} (SA) → {} (Fisher) → {} (subj) → {} (elite)",
            initial,
            after_dopt,
            after_sa,
            after_fisher,
            after_subj,
            after_elite
        );

        Ok(())
    }

    /// Polishing phase: full D-optimal refinement only
    pub(crate) fn polishing_expansion(&mut self) -> Result<()> {
        tracing::debug!("Polishing expansion: full D-optimal refinement");
        self.full_d_optimal_refinement()?;
        Ok(())
    }

    // ========================================================================
    // SOBOL INITIALIZATION
    // ========================================================================

    /// Initialize with Sobol low-discrepancy sequence
    pub(crate) fn sobol_initialization(&mut self, n_samples: usize) -> Result<()> {
        let n_dims = self.ranges.len();
        let mut added = 0;

        for i in 0..n_samples {
            let idx = self.sobol_index + i as u32;
            let mut point = Vec::with_capacity(n_dims);

            for dim in 0..n_dims {
                let sobol_val = sample(idx, dim as u32, 0);
                let (lo, hi) = self.ranges[dim];
                let margin = (hi - lo) * BOUNDARY_MARGIN;
                point.push(lo + margin + sobol_val as f64 * (hi - lo - 2.0 * margin));
            }

            if self.theta.check_point(&point, THETA_D) {
                self.theta.add_point(&point)?;
                added += 1;
            }
        }

        self.sobol_index += n_samples as u32;
        tracing::debug!(
            "Sobol initialization: added {} of {} points",
            added,
            n_samples
        );

        Ok(())
    }

    // ========================================================================
    // D-OPTIMAL REFINEMENT
    // ========================================================================

    /// D-optimal refinement with hierarchical iteration allocation
    pub(crate) fn d_optimal_refinement(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();
        let error_models = self.error_models.clone();
        let max_weight = self.w.iter().fold(f64::NEG_INFINITY, |a, b| a.max(b));

        let n_points = self.theta.nspp().min(self.w.len());
        let min_threshold = max_weight * LOW_WEIGHT_THRESHOLD;

        // Collect points with meaningful weight
        let mut candidate_points: Vec<(Array1<f64>, f64)> = self
            .theta
            .matrix()
            .row_iter()
            .take(n_points)
            .enumerate()
            .filter(|(i, _)| self.w[*i] >= min_threshold)
            .map(|(i, spp)| {
                let point: Vec<f64> = spp.iter().cloned().collect();
                (Array1::from(point), self.w[i] / max_weight)
            })
            .collect();

        let ranges = self.ranges.clone();

        // Parallel optimization
        candidate_points
            .par_iter_mut()
            .for_each(|(spp, importance)| {
                let max_iters = if *importance > HIGH_WEIGHT_THRESHOLD {
                    DOPT_HIGH_ITERS
                } else if *importance > MED_WEIGHT_THRESHOLD {
                    DOPT_MED_ITERS
                } else {
                    DOPT_LOW_ITERS
                };

                let optimizer = super::optimizers::DOptimalOptimizer {
                    equation: &self.equation,
                    data: &self.data,
                    error_models: &error_models,
                    pyl: &pyl,
                };

                if let Ok(refined) = optimizer.optimize(spp.to_vec(), max_iters) {
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

    /// Full D-optimal refinement for polishing phase
    pub(crate) fn full_d_optimal_refinement(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();
        let error_models = self.error_models.clone();
        let n_points = self.theta.nspp().min(self.w.len());

        let mut candidate_points: Vec<Array1<f64>> = self
            .theta
            .matrix()
            .row_iter()
            .take(n_points)
            .map(|spp| Array1::from(spp.iter().cloned().collect::<Vec<_>>()))
            .collect();

        let ranges = self.ranges.clone();

        candidate_points.par_iter_mut().for_each(|spp| {
            let optimizer = super::optimizers::DOptimalOptimizer {
                equation: &self.equation,
                data: &self.data,
                error_models: &error_models,
                pyl: &pyl,
            };

            if let Ok(refined) = optimizer.optimize(spp.to_vec(), DOPT_HIGH_ITERS) {
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

        for cp in candidate_points {
            self.theta.suggest_point(cp.to_vec().as_slice(), THETA_D)?;
        }

        Ok(())
    }

    // ========================================================================
    // ADAPTIVE SA INJECTION
    // ========================================================================

    /// Adaptive simulated annealing injection with reheat mechanism
    pub(crate) fn adaptive_sa_injection(&mut self) -> Result<()> {
        let pyl = self.compute_pyl();

        // Temperature-scaled injection count
        let n_inject = ((SA_INJECT_COUNT as f64) * (self.temperature / INITIAL_TEMPERATURE).sqrt())
            .ceil() as usize;
        let n_inject = n_inject.max(5);

        let mut accepted = 0;
        let mut proposed = 0;

        for _ in 0..n_inject * 15 {
            proposed += 1;

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

        // Update SA tracking
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

    /// Adapt temperature based on acceptance history
    pub(crate) fn adapt_temperature(&mut self) {
        // Record this cycle's acceptance ratio
        if self.sa_proposed > 0 {
            let ratio = self.sa_accepted as f64 / self.sa_proposed as f64;
            self.sa_acceptance_history.push_back(ratio);
            if self.sa_acceptance_history.len() > SA_HISTORY_WINDOW {
                self.sa_acceptance_history.pop_front();
            }
        }

        // Compute average acceptance
        if !self.sa_acceptance_history.is_empty() {
            let avg_acceptance: f64 = self.sa_acceptance_history.iter().sum::<f64>()
                / self.sa_acceptance_history.len() as f64;

            // Adaptive cooling rate and reheat
            if avg_acceptance < REHEAT_TRIGGER {
                self.temperature *= REHEAT_FACTOR;
                self.cooling_rate = 0.95; // Slow down
                tracing::debug!("Reheating to T = {:.4}", self.temperature);
            } else if avg_acceptance > TARGET_ACCEPTANCE * 1.5 {
                self.cooling_rate = 0.85; // Speed up
            } else if avg_acceptance < TARGET_ACCEPTANCE * 0.5 {
                self.cooling_rate = 0.95; // Slow down
            } else {
                self.cooling_rate = BASE_COOLING_RATE;
            }

            tracing::debug!(
                "SA acceptance: {:.1}% | Cooling rate: {:.3}",
                avg_acceptance * 100.0,
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

    // ========================================================================
    // FISHER-GUIDED EXPANSION
    // ========================================================================

    /// Update Fisher Information diagonal estimate
    pub(crate) fn update_fisher_information(&mut self) {
        let n_params = self.ranges.len();
        let n_spp = self.theta.nspp();

        if n_spp < 2 {
            self.fisher_diagonal = vec![1.0; n_params];
            return;
        }

        let mut means = vec![0.0; n_params];
        let mut variances = vec![0.0; n_params];

        // Weighted means
        for (i, spp) in self.theta.matrix().row_iter().enumerate() {
            let weight = if i < self.w.len() { self.w[i] } else { 0.0 };
            for (j, val) in spp.iter().enumerate() {
                means[j] += weight * val;
            }
        }

        // Weighted variances
        for (i, spp) in self.theta.matrix().row_iter().enumerate() {
            let weight = if i < self.w.len() { self.w[i] } else { 0.0 };
            for (j, val) in spp.iter().enumerate() {
                variances[j] += weight * (val - means[j]).powi(2);
            }
        }

        // Fisher ∝ 1/variance, but we want to explore high-variance directions
        for (j, var) in variances.iter().enumerate() {
            let range_scale = (self.ranges[j].1 - self.ranges[j].0).powi(2);
            self.fisher_diagonal[j] = var.max(1e-10) / range_scale;
        }
    }

    /// Fisher-guided expansion in high-variance directions
    pub(crate) fn fisher_expansion(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();

        // Sort dimensions by variance (descending)
        let mut dim_indices: Vec<(usize, f64)> = self
            .fisher_diagonal
            .iter()
            .enumerate()
            .map(|(i, &fi)| (i, fi))
            .collect();
        dim_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Top half of dimensions
        let top_dims: Vec<usize> = dim_indices
            .iter()
            .take((self.ranges.len() + 1) / 2)
            .map(|(i, _)| *i)
            .collect();

        let mut candidates = Vec::new();

        for spp in self.theta.matrix().row_iter() {
            let base: Vec<f64> = spp.iter().cloned().collect();

            for &dim in &top_dims {
                if candidates.len() >= FISHER_CANDIDATES {
                    break;
                }

                let variance = self.fisher_diagonal[dim];
                let range = self.ranges[dim].1 - self.ranges[dim].0;
                let step = (variance.sqrt() * range).max(range * 0.05).min(range * 0.3);

                // Positive direction
                let mut plus = base.clone();
                plus[dim] = (plus[dim] + step).min(self.ranges[dim].1 - range * BOUNDARY_MARGIN);
                candidates.push(plus);

                // Negative direction
                let mut minus = base.clone();
                minus[dim] = (minus[dim] - step).max(self.ranges[dim].0 + range * BOUNDARY_MARGIN);
                candidates.push(minus);
            }
        }

        // Evaluate and add good candidates
        let mut added = 0;
        for candidate in candidates {
            if let Ok(d) = self.compute_d(&candidate, &pyl) {
                if d > 0.0 && self.theta.check_point(&candidate, THETA_D) {
                    self.theta.add_point(&candidate)?;
                    added += 1;
                }
            }
        }

        tracing::debug!("Fisher expansion: added {} points", added);
        Ok(())
    }

    // ========================================================================
    // SUBJECT RESIDUAL INJECTION
    // ========================================================================

    /// Inject points for worst-fit subjects
    pub(crate) fn inject_residual_subjects(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();
        let n_subjects = pyl.len();

        // Find worst-fit subjects (lowest P(y|G))
        let mut indexed_pyl: Vec<(usize, f64)> = pyl.iter().cloned().enumerate().collect();
        indexed_pyl.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n_residual = RESIDUAL_SUBJECTS.min(n_subjects);
        let subjects = self.data.subjects();
        let error_models = self.error_models.clone();

        let mut added = 0;

        for (subj_idx, _) in indexed_pyl.iter().take(n_residual) {
            let subject = &subjects[*subj_idx];

            // Start from weighted centroid
            let start = self.compute_weighted_centroid();

            // Quick subject MAP optimization
            let optimizer = super::optimizers::SubjectMapOptimizer {
                equation: &self.equation,
                subject,
                error_models: &error_models,
                ranges: &self.ranges,
            };

            if let Ok(map_point) = optimizer.optimize(start, SUBJECT_MAP_ITERS) {
                if let Ok(d) = self.compute_d(&map_point, &pyl) {
                    if d > 0.0 && self.theta.check_point(&map_point, THETA_D) {
                        self.theta.add_point(&map_point)?;
                        added += 1;
                    }
                }
            }
        }

        tracing::debug!("Subject residual: added {} points", added);
        Ok(())
    }

    // ========================================================================
    // ELITE PRESERVATION
    // ========================================================================

    /// Update elite points after condensation
    pub(crate) fn update_elite_points(&mut self) -> Result<()> {
        if self.w.len() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();

        // Age existing elite points
        for elite in &mut self.elite_points {
            elite.cycle_added += 1;
        }

        // Remove old elite points
        self.elite_points
            .retain(|e| self.cycle - e.cycle_added < ELITE_MAX_AGE);

        // Find top points by weight
        let n_spp = self.theta.nspp().min(self.w.len());
        let mut indexed_weights: Vec<(usize, f64)> =
            self.w.iter().enumerate().take(n_spp).collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (idx, _) in indexed_weights.iter().take(ELITE_COUNT) {
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
                    cycle_added: self.cycle,
                });
            }
        }

        // Keep only top elite points
        self.elite_points
            .sort_by(|a, b| b.d_value.partial_cmp(&a.d_value).unwrap());
        self.elite_points.truncate(ELITE_COUNT);

        Ok(())
    }

    /// Re-inject elite points into theta
    pub(crate) fn inject_elite_points(&mut self) -> Result<()> {
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

    // ========================================================================
    // GLOBAL OPTIMALITY CHECK
    // ========================================================================

    /// Sobol-based global optimality check
    pub(crate) fn sobol_global_check(&mut self) -> Result<()> {
        if self.theta.nspp() == 0 || self.w.len() == 0 {
            return Ok(());
        }

        let pyl = self.compute_pyl();
        let n_dims = self.ranges.len();

        let mut max_d = f64::NEG_INFINITY;
        let mut max_d_point = vec![0.0; n_dims];

        for i in 0..SOBOL_GLOBAL_SAMPLES {
            let idx = self.sobol_index + i as u32;
            let mut point = Vec::with_capacity(n_dims);

            for dim in 0..n_dims {
                let sobol_val = sample(idx, dim as u32, 0);
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

        self.sobol_index += SOBOL_GLOBAL_SAMPLES as u32;
        self.last_global_d_max = max_d;

        let passed = max_d < GLOBAL_D_THRESHOLD;

        tracing::debug!(
            "Global check: max_D = {:.4} (threshold {:.4}) → {}",
            max_d,
            GLOBAL_D_THRESHOLD,
            if passed { "PASSED" } else { "FAILED" }
        );

        if passed {
            self.global_check_passes += 1;
        } else {
            self.global_check_passes = 0;

            // Inject the violating point if it improves things
            if self.theta.check_point(&max_d_point, THETA_D) {
                self.theta.add_point(&max_d_point)?;
                tracing::debug!("Injected global check point with D = {:.4}", max_d);
            }
        }

        Ok(())
    }
}
