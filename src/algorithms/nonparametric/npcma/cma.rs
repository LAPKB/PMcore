//! CMA-ES Core Implementation
//!
//! Covariance Matrix Adaptation Evolution Strategy components.

use super::constants::*;
use ndarray::{Array1, Array2};
use rand::prelude::*;

/// CMA-ES State
#[derive(Debug, Clone)]
pub struct CmaState {
    /// Mean of the distribution (center)
    pub mean: Array1<f64>,
    /// Step size (overall scale)
    pub sigma: f64,
    /// Covariance matrix
    pub c: Array2<f64>,
    /// Evolution path for sigma
    pub p_sigma: Array1<f64>,
    /// Evolution path for C
    pub p_c: Array1<f64>,
    /// Number of dimensions
    pub n_dims: usize,
    /// Parameter ranges
    pub ranges: Vec<(f64, f64)>,
    /// Generation counter
    pub generation: usize,
    /// Best fitness seen
    pub best_fitness: f64,
    /// Generations without improvement
    pub stagnation: usize,
}

impl CmaState {
    pub fn new(n_dims: usize, ranges: &[(f64, f64)], seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Initialize mean at center of domain
        let mean: Array1<f64> = ranges
            .iter()
            .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
            .collect();

        // Identity covariance (scaled by range)
        let mut c = Array2::<f64>::eye(n_dims);
        for i in 0..n_dims {
            let (lo, hi) = ranges[i];
            let scale = (hi - lo) / 4.0; // Initial scale
            c[[i, i]] = scale * scale;
        }

        // Evolution paths start at zero
        let p_sigma = Array1::zeros(n_dims);
        let p_c = Array1::zeros(n_dims);

        Self {
            mean,
            sigma: INITIAL_SIGMA,
            c,
            p_sigma,
            p_c,
            n_dims,
            ranges: ranges.to_vec(),
            generation: 0,
            best_fitness: f64::NEG_INFINITY,
            stagnation: 0,
        }
    }

    /// Sample a population from the current distribution
    pub fn sample_population<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<Array1<f64>> {
        let mut population = Vec::with_capacity(n);

        // Simple sampling: mean + sigma * C^(1/2) * z
        // For simplicity, we use diagonal approximation initially
        for _ in 0..n {
            let mut sample = self.mean.clone();

            for i in 0..self.n_dims {
                let std = (self.c[[i, i]]).sqrt() * self.sigma;
                let z: f64 = sample_standard_normal(rng);
                sample[i] += std * z;

                // Clamp to bounds
                let (lo, hi) = self.ranges[i];
                sample[i] = sample[i].clamp(lo, hi);
            }

            population.push(sample);
        }

        population
    }

    /// Update the CMA-ES state given sorted population (best first)
    pub fn update(&mut self, sorted_population: &[Array1<f64>], sorted_fitness: &[f64]) {
        self.generation += 1;

        // Check for improvement
        if sorted_fitness[0] > self.best_fitness {
            self.best_fitness = sorted_fitness[0];
            self.stagnation = 0;
        } else {
            self.stagnation += 1;
        }

        // Compute weighted mean of best individuals
        let mu = N_PARENTS.min(sorted_population.len());
        let weights = compute_weights(mu);

        let mut new_mean = Array1::zeros(self.n_dims);
        for i in 0..mu {
            new_mean = new_mean + weights[i] * &sorted_population[i];
        }

        // Mean displacement
        let y = (&new_mean - &self.mean) / self.sigma;

        // Update evolution path for sigma (cumulative step-size adaptation)
        let c_s = C_SIGMA;
        let chi_n = (self.n_dims as f64).sqrt(); // Expected length of N(0,I) vector
        self.p_sigma = (1.0 - c_s) * &self.p_sigma + (c_s * (2.0 - c_s)).sqrt() * &y;

        // Update sigma
        let ps_norm = norm(&self.p_sigma);
        self.sigma *= ((c_s / D_SIGMA) * (ps_norm / chi_n - 1.0)).exp();
        self.sigma = self.sigma.clamp(MIN_SIGMA, MAX_SIGMA);

        // Update evolution path for covariance
        let hsig = if ps_norm / ((1.0 - (1.0 - c_s).powi(2 * self.generation as i32)).sqrt())
            < (1.4 + 2.0 / (self.n_dims as f64 + 1.0)) * chi_n
        {
            1.0
        } else {
            0.0
        };

        self.p_c = (1.0 - C_C) * &self.p_c + hsig * (C_C * (2.0 - C_C)).sqrt() * &y;

        // Update covariance matrix
        // Rank-1 update
        let rank1 = outer(&self.p_c, &self.p_c);

        // Rank-mu update
        let mut rank_mu = Array2::<f64>::zeros((self.n_dims, self.n_dims));
        for i in 0..mu {
            let yi = (&sorted_population[i] - &self.mean) / self.sigma;
            rank_mu = rank_mu + weights[i] * outer(&yi, &yi);
        }

        // Combined update
        self.c = (1.0 - C_1 - C_MU) * &self.c + C_1 * &rank1 + C_MU * &rank_mu;

        // Ensure symmetry and positive definiteness
        self.repair_covariance();

        // Update mean
        self.mean = new_mean;
    }

    /// Repair covariance matrix for numerical stability
    fn repair_covariance(&mut self) {
        // Ensure symmetry
        for i in 0..self.n_dims {
            for j in 0..i {
                let avg = (self.c[[i, j]] + self.c[[j, i]]) / 2.0;
                self.c[[i, j]] = avg;
                self.c[[j, i]] = avg;
            }
        }

        // Ensure positive diagonal
        for i in 0..self.n_dims {
            self.c[[i, i]] = self.c[[i, i]].max(EIGENVALUE_FLOOR);
        }

        // Check condition number (diagonal approximation)
        let diag: Vec<f64> = (0..self.n_dims).map(|i| self.c[[i, i]]).collect();
        let max_d = diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_d = diag.iter().cloned().fold(f64::INFINITY, f64::min);

        if max_d / min_d > CONDITION_THRESHOLD {
            // Scale up small eigenvalues
            let floor = max_d / CONDITION_THRESHOLD;
            for i in 0..self.n_dims {
                self.c[[i, i]] = self.c[[i, i]].max(floor);
            }
        }
    }

    /// Should we restart?
    pub fn should_restart(&self) -> bool {
        self.stagnation >= MAX_STAGNATION || self.sigma < MIN_SIGMA
    }

    /// Restart with new random mean
    pub fn restart<R: Rng>(&mut self, rng: &mut R) {
        // New random mean
        self.mean = self
            .ranges
            .iter()
            .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
            .collect();

        // Reset covariance
        self.c = Array2::<f64>::eye(self.n_dims);
        for i in 0..self.n_dims {
            let (lo, hi) = self.ranges[i];
            let scale = (hi - lo) / 4.0;
            self.c[[i, i]] = scale * scale;
        }

        // Reset paths
        self.p_sigma = Array1::zeros(self.n_dims);
        self.p_c = Array1::zeros(self.n_dims);

        // Reset sigma
        self.sigma = INITIAL_SIGMA;

        // Reset stagnation
        self.stagnation = 0;
    }

    /// Update mean based on high-weight support points
    pub fn update_mean_from_points(&mut self, weighted_points: &[(Vec<f64>, f64)]) {
        if weighted_points.is_empty() {
            return;
        }

        // Compute weighted centroid
        let total_weight: f64 = weighted_points.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return;
        }

        let mut centroid = vec![0.0; self.n_dims];
        for (point, weight) in weighted_points {
            for (i, &val) in point.iter().enumerate() {
                centroid[i] += val * weight / total_weight;
            }
        }

        // Move mean toward centroid (partial update to avoid instability)
        let learning_rate = 0.3;
        for i in 0..self.n_dims {
            self.mean[i] = (1.0 - learning_rate) * self.mean[i] + learning_rate * centroid[i];

            // Ensure within bounds
            let (lo, hi) = self.ranges[i];
            self.mean[i] = self.mean[i].clamp(lo, hi);
        }

        // Also update covariance to reflect the spread of high-weight points
        if weighted_points.len() > 1 {
            for i in 0..self.n_dims {
                let mut variance = 0.0;
                for (point, weight) in weighted_points {
                    let diff = point[i] - self.mean[i];
                    variance += weight * diff * diff / total_weight;
                }
                // Blend with current covariance
                self.c[[i, i]] = 0.7 * self.c[[i, i]] + 0.3 * variance.max(EIGENVALUE_FLOOR);
            }
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Compute recombination weights (log-linear)
fn compute_weights(mu: usize) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..mu)
        .map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0))
        .collect();

    let sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= sum;
    }

    weights
}

/// Sample from standard normal using Box-Muller
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Compute vector norm
fn norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Outer product
fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let mut result = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cma_basic() {
        let ranges = vec![(0.0, 1.0), (0.0, 1.0)];
        let state = CmaState::new(2, &ranges, 42);

        assert_eq!(state.n_dims, 2);
        assert!(state.sigma > 0.0);
    }

    #[test]
    fn test_sampling() {
        let ranges = vec![(0.0, 1.0), (0.0, 1.0)];
        let state = CmaState::new(2, &ranges, 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let pop = state.sample_population(10, &mut rng);
        assert_eq!(pop.len(), 10);

        // Check bounds
        for ind in &pop {
            for (i, &x) in ind.iter().enumerate() {
                let (lo, hi) = ranges[i];
                assert!(x >= lo && x <= hi);
            }
        }
    }
}
