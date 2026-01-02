//! Sufficient statistics for SAEM algorithm
//!
//! This module defines the [`SufficientStats`] structure used in the SAEM
//! (Stochastic Approximation Expectation-Maximization) algorithm.
//!
//! # Mathematical Background
//!
//! In the SAEM algorithm, the E-step is replaced by a stochastic approximation
//! using sufficient statistics. For a normal population distribution, the
//! sufficient statistics are:
//!
//! ```text
//! S₁ = Σᵢ h(ψᵢ)           (sum of transformed parameters)
//! S₂ = Σᵢ h(ψᵢ)h(ψᵢ)ᵀ     (sum of outer products)
//! ```
//!
//! where h(ψ) is typically log(ψ) for log-normally distributed parameters,
//! or simply ψ for normally distributed parameters.
//!
//! # Stochastic Approximation
//!
//! At each iteration k, the sufficient statistics are updated using:
//!
//! ```text
//! sₖ = sₖ₋₁ + γₖ(Sₖ - sₖ₋₁)
//! ```
//!
//! where γₖ is the step size (typically 1/k for convergence guarantees).
//!
//! # M-Step
//!
//! The M-step updates the population parameters using closed-form solutions:
//!
//! ```text
//! μ = S₁ / n
//! Ω = S₂ / n - μμᵀ
//! ```

use anyhow::{bail, Result};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};

/// Sufficient statistics for parametric population estimation
///
/// Used primarily in the SAEM algorithm to accumulate statistics
/// from individual parameter samples for the M-step update.
///
/// Based on saemix R reference, tracks:
/// - statphi1 (s1): sum of parameters over chains
/// - statphi2 (s2): sum of outer products over chains
/// - statphi3 (s3): sum of squared parameters (for conditional variance)
/// - statrese: residual error sufficient statistic
#[derive(Debug, Clone)]
pub struct SufficientStats {
    /// S₁: Sum of (transformed) individual parameters (statphi1 in R)
    /// Dimension: n_params × 1
    s1: Col<f64>,
    /// S₂: Sum of outer products of (transformed) individual parameters (statphi2 in R)
    /// Dimension: n_params × n_params
    s2: Mat<f64>,
    /// S₃: Sum of squared parameters per component (statphi3 in R)
    /// Used for conditional variance: var = s3 - s1²
    /// Dimension: n_params × 1
    s3: Col<f64>,
    /// Residual error sufficient statistic (statrese in R)
    /// Sum of squared residuals for residual error update
    stat_rese: f64,
    /// Number of observations (subjects) accumulated
    count: usize,
    /// Number of observations (data points) for residual error
    n_obs: usize,
}

impl SufficientStats {
    /// Create new sufficient statistics for a given number of parameters
    pub fn new(n_params: usize) -> Self {
        Self {
            s1: Col::zeros(n_params),
            s2: Mat::zeros(n_params, n_params),
            s3: Col::zeros(n_params),
            stat_rese: 0.0,
            count: 0,
            n_obs: 0,
        }
    }

    /// Reset all statistics to zero
    pub fn reset(&mut self) {
        let n = self.s1.nrows();
        self.s1 = Col::zeros(n);
        self.s2 = Mat::zeros(n, n);
        self.s3 = Col::zeros(n);
        self.stat_rese = 0.0;
        self.count = 0;
        self.n_obs = 0;
    }

    /// Get the dimension (number of parameters)
    pub fn npar(&self) -> usize {
        self.s1.nrows()
    }

    /// Get S₁ (sum of parameters)
    pub fn s1(&self) -> &Col<f64> {
        &self.s1
    }

    /// Get a mutable reference to S₁
    pub fn s1_mut(&mut self) -> &mut Col<f64> {
        &mut self.s1
    }

    /// Get S₂ (sum of outer products)
    pub fn s2(&self) -> &Mat<f64> {
        &self.s2
    }

    /// Get a mutable reference to S₂
    pub fn s2_mut(&mut self) -> &mut Mat<f64> {
        &mut self.s2
    }

    /// Get the count of accumulated samples (subjects)
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get S₃ (sum of squared parameters)
    pub fn s3(&self) -> &Col<f64> {
        &self.s3
    }

    /// Get a mutable reference to S₃
    pub fn s3_mut(&mut self) -> &mut Col<f64> {
        &mut self.s3
    }

    /// Get residual error statistic
    pub fn stat_rese(&self) -> f64 {
        self.stat_rese
    }

    /// Set residual error statistic
    pub fn set_stat_rese(&mut self, value: f64) {
        self.stat_rese = value;
    }

    /// Add to residual error statistic
    pub fn add_stat_rese(&mut self, value: f64) {
        self.stat_rese += value;
    }

    /// Get number of observations (data points)
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Set number of observations
    pub fn set_n_obs(&mut self, n: usize) {
        self.n_obs = n;
    }

    /// Add to observation count
    pub fn add_n_obs(&mut self, n: usize) {
        self.n_obs += n;
    }

    /// Add an individual parameter vector to the statistics
    ///
    /// Updates:
    /// - S₁ += h(ψ)
    /// - S₂ += h(ψ)h(ψ)ᵀ
    /// - S₃ += h(ψ)² (element-wise)
    /// - count += 1
    ///
    /// # Arguments
    ///
    /// * `psi` - Individual parameter vector (already transformed if needed)
    pub fn accumulate(&mut self, psi: &Col<f64>) -> Result<()> {
        let n = self.npar();

        if psi.nrows() != n {
            bail!(
                "Parameter vector length ({}) doesn't match statistics dimension ({})",
                psi.nrows(),
                n
            );
        }

        // Update S₁: sum of parameters
        for i in 0..n {
            self.s1[i] += psi[i];
        }

        // Update S₂: sum of outer products
        for i in 0..n {
            for j in 0..n {
                self.s2[(i, j)] += psi[i] * psi[j];
            }
        }

        // Update S₃: sum of squared parameters (for conditional variance)
        for i in 0..n {
            self.s3[i] += psi[i] * psi[i];
        }

        self.count += 1;

        Ok(())
    }

    /// Accumulate a batch of parameter vectors
    ///
    /// # Arguments
    ///
    /// * `samples` - Slice of individual parameter vectors
    pub fn accumulate_batch(&mut self, samples: &[Col<f64>]) -> Result<()> {
        for sample in samples {
            self.accumulate(sample)?;
        }
        Ok(())
    }

    /// Perform stochastic approximation update
    ///
    /// Updates the current statistics using:
    /// ```text
    /// s = s + γ(S_new - s)
    /// ```
    ///
    /// This is the core of the SAEM algorithm's stochastic approximation.
    /// Matches the R saemix reference implementation.
    ///
    /// # Arguments
    ///
    /// * `new_stats` - New statistics computed from current iteration's samples
    /// * `step_size` - The step size γₖ (typically 1/k or follows a decreasing schedule)
    pub fn stochastic_update(&mut self, new_stats: &SufficientStats, step_size: f64) -> Result<()> {
        if self.npar() != new_stats.npar() {
            bail!(
                "Statistics dimension mismatch: {} vs {}",
                self.npar(),
                new_stats.npar()
            );
        }

        // If step_size is 0, skip update (pure burn-in phase in R)
        if step_size == 0.0 {
            return Ok(());
        }

        let n = self.npar();

        // Update S₁: statphi1 in R
        for i in 0..n {
            self.s1[i] += step_size * (new_stats.s1[i] - self.s1[i]);
        }

        // Update S₂: statphi2 in R
        for i in 0..n {
            for j in 0..n {
                self.s2[(i, j)] += step_size * (new_stats.s2[(i, j)] - self.s2[(i, j)]);
            }
        }

        // Update S₃: statphi3 in R
        for i in 0..n {
            self.s3[i] += step_size * (new_stats.s3[i] - self.s3[i]);
        }

        // Update stat_rese: residual error statistic
        self.stat_rese += step_size * (new_stats.stat_rese - self.stat_rese);

        // Update counts with weighted average
        self.count = ((1.0 - step_size) * self.count as f64 + step_size * new_stats.count as f64)
            .round() as usize;
        self.n_obs = ((1.0 - step_size) * self.n_obs as f64 + step_size * new_stats.n_obs as f64)
            .round() as usize;

        Ok(())
    }

    /// Compute the M-step estimates from current statistics
    ///
    /// Returns (μ, Ω) where:
    /// - μ = S₁ / n
    /// - Ω = S₂ / n - μμᵀ
    ///
    /// # Returns
    ///
    /// Tuple of (mean vector, covariance matrix)
    pub fn compute_m_step(&self) -> Result<(Col<f64>, Mat<f64>)> {
        if self.count == 0 {
            bail!("Cannot compute M-step with zero samples");
        }

        let n = self.npar();
        let count_f64 = self.count as f64;

        // Compute μ = S₁ / n
        let mu = Col::from_fn(n, |i| self.s1[i] / count_f64);

        // Compute Ω = S₂ / n - μμᵀ
        let omega = Mat::from_fn(n, n, |i, j| self.s2[(i, j)] / count_f64 - mu[i] * mu[j]);

        Ok((mu, omega))
    }

    /// Merge another set of sufficient statistics into this one
    ///
    /// Useful for combining statistics from parallel workers
    pub fn merge(&mut self, other: &SufficientStats) -> Result<()> {
        if self.npar() != other.npar() {
            bail!(
                "Cannot merge statistics with different dimensions: {} vs {}",
                self.npar(),
                other.npar()
            );
        }

        let n = self.npar();

        for i in 0..n {
            self.s1[i] += other.s1[i];
        }

        for i in 0..n {
            for j in 0..n {
                self.s2[(i, j)] += other.s2[(i, j)];
            }
        }

        for i in 0..n {
            self.s3[i] += other.s3[i];
        }

        self.stat_rese += other.stat_rese;
        self.count += other.count;
        self.n_obs += other.n_obs;

        Ok(())
    }
}

impl Default for SufficientStats {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Serialize for SufficientStats {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("SufficientStats", 6)?;

        // Serialize s1 as Vec<f64>
        let s1_vec: Vec<f64> = (0..self.s1.nrows()).map(|i| self.s1[i]).collect();
        state.serialize_field("s1", &s1_vec)?;

        // Serialize s2 as Vec<Vec<f64>>
        let s2_vec: Vec<Vec<f64>> = (0..self.s2.nrows())
            .map(|i| (0..self.s2.ncols()).map(|j| self.s2[(i, j)]).collect())
            .collect();
        state.serialize_field("s2", &s2_vec)?;

        // Serialize s3 as Vec<f64>
        let s3_vec: Vec<f64> = (0..self.s3.nrows()).map(|i| self.s3[i]).collect();
        state.serialize_field("s3", &s3_vec)?;

        state.serialize_field("stat_rese", &self.stat_rese)?;
        state.serialize_field("count", &self.count)?;
        state.serialize_field("n_obs", &self.n_obs)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for SufficientStats {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SufficientStatsData {
            s1: Vec<f64>,
            s2: Vec<Vec<f64>>,
            #[serde(default)]
            s3: Option<Vec<f64>>,
            #[serde(default)]
            stat_rese: f64,
            count: usize,
            #[serde(default)]
            n_obs: usize,
        }

        let data = SufficientStatsData::deserialize(deserializer)?;

        let n = data.s1.len();
        let s1 = Col::from_fn(n, |i| data.s1[i]);

        if data.s2.len() != n {
            return Err(serde::de::Error::custom(
                "S2 row count doesn't match S1 length",
            ));
        }

        let s2 = Mat::from_fn(n, n, |i, j| {
            if j < data.s2[i].len() {
                data.s2[i][j]
            } else {
                0.0
            }
        });

        // Handle optional s3 (for backwards compatibility)
        let s3 = match data.s3 {
            Some(s3_data) if s3_data.len() == n => Col::from_fn(n, |i| s3_data[i]),
            _ => Col::zeros(n),
        };

        Ok(SufficientStats {
            s1,
            s2,
            s3,
            stat_rese: data.stat_rese,
            count: data.count,
            n_obs: data.n_obs,
        })
    }
}

/// Step size schedule for SAEM stochastic approximation
#[derive(Debug, Clone, Copy)]
pub enum StepSizeSchedule {
    /// Constant step size (for exploration/burn-in)
    Constant(f64),
    /// Decreasing step size: γₖ = 1/k
    Harmonic,
    /// Robbins-Monro: γₖ = a / (k + b)
    RobbinsMonro { a: f64, b: f64 },
    /// Polyak-Ruppert averaging schedule
    PolyakRuppert { start_averaging: usize },
}

impl StepSizeSchedule {
    /// Create a SAEM-style step size schedule
    ///
    /// Uses constant step size (γ=1) during burn-in, then decreasing
    /// step sizes during stochastic approximation phase.
    pub fn new_saem(n_burn_in: usize, _n_stochastic: usize) -> Self {
        StepSizeSchedule::PolyakRuppert {
            start_averaging: n_burn_in,
        }
    }

    /// Compute the step size for iteration k (1-indexed)
    pub fn step_size(&self, k: usize) -> f64 {
        match self {
            StepSizeSchedule::Constant(gamma) => *gamma,
            StepSizeSchedule::Harmonic => 1.0 / k as f64,
            StepSizeSchedule::RobbinsMonro { a, b } => a / (k as f64 + b),
            StepSizeSchedule::PolyakRuppert { start_averaging } => {
                if k < *start_averaging {
                    1.0 // Full update during burn-in
                } else {
                    1.0 / (k - start_averaging + 1) as f64
                }
            }
        }
    }
}

impl Default for StepSizeSchedule {
    fn default() -> Self {
        StepSizeSchedule::PolyakRuppert {
            start_averaging: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sufficient_stats_accumulation() {
        let mut stats = SufficientStats::new(2);

        // Add some samples
        let sample1 = Col::from_fn(2, |i| if i == 0 { 1.0 } else { 2.0 });
        let sample2 = Col::from_fn(2, |i| if i == 0 { 3.0 } else { 4.0 });

        stats.accumulate(&sample1).unwrap();
        stats.accumulate(&sample2).unwrap();

        assert_eq!(stats.count(), 2);
        assert_eq!(stats.s1()[0], 4.0); // 1 + 3
        assert_eq!(stats.s1()[1], 6.0); // 2 + 4
    }

    #[test]
    fn test_m_step_computation() {
        let mut stats = SufficientStats::new(2);

        // Add samples: [1, 2], [3, 4], [5, 6]
        for i in 0..3 {
            let sample = Col::from_fn(2, |j| (2 * i + j + 1) as f64);
            stats.accumulate(&sample).unwrap();
        }

        let (mu, _omega) = stats.compute_m_step().unwrap();

        // Mean should be [3, 4]
        assert!((mu[0] - 3.0).abs() < 1e-10);
        assert!((mu[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_stochastic_update() {
        let mut stats = SufficientStats::new(2);
        stats.s1[0] = 10.0;
        stats.s1[1] = 20.0;
        stats.count = 10;

        let mut new_stats = SufficientStats::new(2);
        new_stats.s1[0] = 12.0;
        new_stats.s1[1] = 22.0;
        new_stats.count = 5;

        stats.stochastic_update(&new_stats, 0.5).unwrap();

        // s1 should be updated: 10 + 0.5*(12-10) = 11
        assert!((stats.s1[0] - 11.0).abs() < 1e-10);
        assert!((stats.s1[1] - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_size_schedules() {
        let harmonic = StepSizeSchedule::Harmonic;
        assert_eq!(harmonic.step_size(1), 1.0);
        assert_eq!(harmonic.step_size(10), 0.1);

        let rm = StepSizeSchedule::RobbinsMonro { a: 1.0, b: 10.0 };
        assert!((rm.step_size(1) - 1.0 / 11.0).abs() < 1e-10);
    }
}
