//! # NPCMA: Non-Parametric Covariance Matrix Adaptation Algorithm
//!
//! A CMA-ES (Covariance Matrix Adaptation Evolution Strategy) approach to
//! nonparametric population pharmacokinetics.
//!
//! ## Algorithm Overview
//!
//! CMA-ES is a state-of-the-art derivative-free optimization algorithm that
//! adapts a multivariate normal distribution to sample promising solutions.
//! It learns the covariance structure of the fitness landscape, making it
//! particularly effective for correlated parameters.
//!
//! ## Key Innovations for Pharmacometrics
//!
//! 1. **D-Criterion Fitness**: Each sample is evaluated using the D-optimality
//!    criterion, directing the search toward information-maximizing regions
//! 2. **Covariance Adaptation**: Automatically learns parameter correlations
//! 3. **Step Size Control**: Adaptive sigma prevents premature convergence
//! 4. **Restart Strategy**: Escapes local optima through intelligent restarts
//!
//! ## Algorithm Structure
//!
//! - **Warm-up (cycles 1-3)**: NPAG-style grid expansion for broad coverage
//! - **CMA Phase**: Sample from adapted distribution, evaluate D-criterion,
//!   update distribution parameters toward high-D regions
//! - **Estimation**: Standard IPM to compute weights
//! - **Condensation**: QR-based pruning of redundant points

mod cma;
mod constants;

use cma::CmaState;
pub use constants::*;

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
use faer_ext::IntoNdarray;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::{Array, Array1, ArrayBase, Axis, Dim, OwnedRepr};
use pharmsol::prelude::data::Data;
use pharmsol::prelude::simulator::Equation;
use pharmsol::{prelude::AssayErrorModel, AssayErrorModels, Subject};
use rand::prelude::*;
use rand::SeedableRng;

// ============================================================================
// NPCMA STRUCT
// ============================================================================

pub struct NPCMA<E: Equation + Send + 'static> {
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

    // CMA-ES specific
    cma: CmaState,
    rng: StdRng,
    /// Cached pyl vector for D-criterion evaluation
    pyl: Array1<f64>,
    /// Phase: true = warm-up (grid expansion), false = CMA-driven
    in_warmup: bool,
}

// ============================================================================
// ALGORITHMS TRAIT
// ============================================================================

impl<E: Equation + Send + 'static> Algorithms<E> for NPCMA<E> {
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
            // CMA-ES specific
            cma: CmaState::new(n_dims, &ranges, seed),
            rng: StdRng::seed_from_u64(seed),
            pyl: Array1::ones(n_subjects),
            in_warmup: true,
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
            tracing::info!("NPCMA: Warm-up complete, entering CMA-ES driven expansion");
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
            "Support points: {} | Phase: {} | Sigma: {:.4} | EPS: {:.4}",
            self.theta.nspp(),
            if self.in_warmup { "Warm-up" } else { "CMA-ES" },
            self.cma.sigma,
            self.eps
        );

        self.error_models.iter().for_each(|(outeq, em)| {
            if AssayErrorModel::None != *em {
                tracing::debug!(
                    "Error model outeq {}: {:.4}",
                    outeq,
                    em.factor().unwrap_or_default()
                );
            }
        });

        if self.last_objf > self.objf + 1e-4 {
            tracing::warn!(
                "Objective decreased: {:.4} -> {:.4}",
                -2.0 * self.last_objf,
                -2.0 * self.objf
            );
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
                    // Also check global optimality via CMA sampling
                    let global_check = self.global_optimality_check()?;
                    if global_check {
                        tracing::info!("NPCMA converged after {} cycles", self.cycle);
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

        // Update CMA distribution based on high-weight support points
        self.update_cma_from_weights()?;

        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        // Standard error model optimization
        self.optimize_error_models()?;

        Ok(())
    }

    fn expansion(&mut self) -> Result<()> {
        if self.in_warmup {
            // Warm-up: NPAG-style grid expansion for broad coverage
            self.warmup_expansion()?;
        } else {
            // CMA-ES driven expansion
            self.cma_expansion()?;
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
// NPCMA SPECIFIC METHODS
// ============================================================================

impl<E: Equation + Send + 'static> NPCMA<E> {
    /// Warm-up expansion using adaptive grid (like NPAG)
    fn warmup_expansion(&mut self) -> Result<()> {
        tracing::debug!("NPCMA warm-up: adaptive grid expansion");
        adaptative_grid(&mut self.theta, self.eps, &self.ranges, THETA_D)?;
        Ok(())
    }

    /// CMA-ES driven expansion: sample from adapted distribution
    fn cma_expansion(&mut self) -> Result<()> {
        let initial_points = self.theta.nspp();

        // Check for restart conditions
        if self.cma.should_restart() {
            tracing::info!("NPCMA: Restarting CMA-ES distribution");
            self.cma.restart(&mut self.rng);
        }

        // 1. Sample population from CMA distribution
        let population = self.cma.sample_population(POPULATION_SIZE, &mut self.rng);

        // 2. Evaluate D-criterion for all samples (in parallel)
        let samples: Vec<Vec<f64>> = population.iter().map(|a| a.to_vec()).collect();
        let fitness: Vec<f64> = samples
            .clone()
            .into_par_iter()
            .map(|pos| self.compute_d_criterion(&pos).unwrap_or(f64::NEG_INFINITY))
            .collect();

        // 3. Sort by fitness (descending - higher D is better)
        let mut indexed: Vec<(usize, f64)> = fitness.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_population: Vec<Array1<f64>> = indexed
            .iter()
            .map(|(i, _)| population[*i].clone())
            .collect();
        let sorted_fitness: Vec<f64> = indexed.iter().map(|(_, f)| *f).collect();

        // 4. Update CMA-ES distribution
        self.cma.update(&sorted_population, &sorted_fitness);

        // 5. Add high-fitness samples as support point candidates
        let max_fitness = sorted_fitness.first().copied().unwrap_or(f64::NEG_INFINITY);
        let threshold = max_fitness * D_THRESHOLD_FRACTION;

        let mut added = 0;
        for (i, sample) in samples.iter().enumerate() {
            if fitness[i] > threshold.max(0.0) {
                if self.theta.check_point(sample, THETA_D) {
                    self.theta.add_point(sample)?;
                    added += 1;
                }
            }
        }

        // 6. Sparse grid expansion to fill gaps every few cycles
        if self.cycle % 3 == 0 {
            let sparse_eps = self.eps * 0.5;
            adaptative_grid(&mut self.theta, sparse_eps, &self.ranges, THETA_D * 2.0)?;
        }

        tracing::debug!(
            "CMA expansion: {} -> {} (added {}, sigma={:.4})",
            initial_points,
            self.theta.nspp(),
            added,
            self.cma.sigma
        );

        Ok(())
    }

    /// Update CMA distribution to center on high-weight support points
    fn update_cma_from_weights(&mut self) -> Result<()> {
        if self.w.len() == 0 || self.theta.nspp() == 0 {
            return Ok(());
        }

        // Find high-weight points
        let max_weight = self.w.iter().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let threshold = max_weight * 0.1;

        let n_points = self.theta.nspp().min(self.w.len());
        let mut high_weight_points: Vec<(Vec<f64>, f64)> = Vec::new();

        for (i, row) in self.theta.matrix().row_iter().enumerate().take(n_points) {
            if self.w[i] >= threshold {
                let point: Vec<f64> = row.iter().copied().collect();
                high_weight_points.push((point, self.w[i]));
            }
        }

        if !high_weight_points.is_empty() {
            // Update CMA mean toward weighted centroid of high-weight points
            self.cma.update_mean_from_points(&high_weight_points);
        }

        Ok(())
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

    /// Global optimality check using CMA sampling
    fn global_optimality_check(&mut self) -> Result<bool> {
        // Sample from CMA distribution and check if any have high D
        let samples = self
            .cma
            .sample_population(GLOBAL_CHECK_SAMPLES, &mut self.rng);
        let mut max_d = f64::NEG_INFINITY;

        for sample in &samples {
            let d = self.compute_d_criterion(&sample.to_vec())?;
            max_d = max_d.max(d);
        }

        // Also sample uniformly random
        for _ in 0..GLOBAL_CHECK_SAMPLES {
            let point: Vec<f64> = self
                .ranges
                .iter()
                .map(|(lo, hi)| self.rng.random_range(*lo..*hi))
                .collect();
            let d = self.compute_d_criterion(&point)?;
            max_d = max_d.max(d);
        }

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
            if *em == AssayErrorModel::None || em.is_factor_fixed().unwrap_or(true) {
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
