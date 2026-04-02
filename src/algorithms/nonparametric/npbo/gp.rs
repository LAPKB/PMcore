//! Gaussian Process implementation for NPBO
//!
//! A simple but efficient GP with RBF kernel for surrogate modeling.

use super::constants::*;
use ndarray::{Array1, Array2};

/// Gaussian Process with RBF kernel
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    /// Training points (N x D)
    x_train: Vec<Vec<f64>>,
    /// Training targets (N)
    y_train: Vec<f64>,
    /// Per-dimension length scales (ARD)
    length_scales: Vec<f64>,
    /// Signal variance
    signal_var: f64,
    /// Noise variance  
    noise_var: f64,
    /// Precomputed inverse covariance (for prediction)
    k_inv_y: Option<Array1<f64>>,
    /// Precomputed Cholesky factor
    l_chol: Option<Array2<f64>>,
    /// Number of dimensions
    n_dims: usize,
    /// Parameter ranges for normalization
    ranges: Vec<(f64, f64)>,
}

impl GaussianProcess {
    pub fn new(n_dims: usize, ranges: &[(f64, f64)]) -> Self {
        let length_scales = if USE_ARD {
            vec![INITIAL_LENGTH_SCALE; n_dims]
        } else {
            vec![INITIAL_LENGTH_SCALE]
        };

        Self {
            x_train: Vec::new(),
            y_train: Vec::new(),
            length_scales,
            signal_var: INITIAL_SIGNAL_VAR,
            noise_var: NOISE_VAR,
            k_inv_y: None,
            l_chol: None,
            n_dims,
            ranges: ranges.to_vec(),
        }
    }

    /// Normalize a point to [0, 1] range
    fn normalize(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.ranges)
            .map(|(&xi, &(lo, hi))| (xi - lo) / (hi - lo))
            .collect()
    }

    /// RBF kernel between two points
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let mut sq_dist = 0.0;
        for i in 0..x1.len() {
            let ls = if USE_ARD {
                self.length_scales[i]
            } else {
                self.length_scales[0]
            };
            let d = (x1[i] - x2[i]) / ls;
            sq_dist += d * d;
        }
        self.signal_var * (-0.5 * sq_dist).exp()
    }

    /// Add a training point
    pub fn add_point(&mut self, x: &[f64], y: f64) {
        let x_norm = self.normalize(x);
        self.x_train.push(x_norm);
        self.y_train.push(y);
        // Invalidate precomputed matrices
        self.k_inv_y = None;
        self.l_chol = None;
    }

    /// Get number of training points
    pub fn n_points(&self) -> usize {
        self.x_train.len()
    }

    /// Fit the GP (compute Cholesky decomposition)
    pub fn fit(&mut self) -> Result<(), String> {
        let n = self.x_train.len();
        if n < MIN_GP_POINTS {
            return Err("Not enough training points".to_string());
        }

        // Build covariance matrix
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let kij = self.kernel(&self.x_train[i], &self.x_train[j]);
                k[[i, j]] = kij;
                k[[j, i]] = kij;
            }
            // Add noise on diagonal
            k[[i, i]] += self.noise_var;
        }

        // Cholesky decomposition
        let l = match cholesky_decomp(&k) {
            Ok(l) => l,
            Err(e) => return Err(format!("Cholesky failed: {}", e)),
        };

        // Solve L * L^T * alpha = y for alpha
        let y = Array1::from_vec(self.y_train.clone());
        let alpha = cholesky_solve(&l, &y);

        self.l_chol = Some(l);
        self.k_inv_y = Some(alpha);

        Ok(())
    }

    /// Predict mean and variance at a point
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        let (k_inv_y, l_chol) = match (&self.k_inv_y, &self.l_chol) {
            (Some(a), Some(l)) => (a, l),
            _ => return (0.0, self.signal_var), // Prior
        };

        let x_norm = self.normalize(x);

        // Compute k_star (covariance with training points)
        let k_star: Array1<f64> = self
            .x_train
            .iter()
            .map(|xi| self.kernel(&x_norm, xi))
            .collect();

        // Mean: k_star^T * alpha
        let mean = k_star.dot(k_inv_y);

        // Variance: k(x,x) - k_star^T * K^-1 * k_star
        let k_xx = self.signal_var + self.noise_var;

        // Solve L * v = k_star
        let v = forward_solve(l_chol, &k_star);
        let var = (k_xx - v.dot(&v)).max(1e-10);

        (mean, var)
    }

    /// Expected Improvement acquisition function
    pub fn expected_improvement(&self, x: &[f64], y_best: f64) -> f64 {
        let (mean, var) = self.predict(x);
        let std = var.sqrt();

        if std < 1e-12 {
            return 0.0;
        }

        let z = (mean - y_best) / std;
        let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let cdf = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));

        (mean - y_best) * cdf + std * pdf
    }

    /// Upper Confidence Bound acquisition function (alternative to EI)
    #[allow(dead_code)]
    pub fn ucb(&self, x: &[f64], beta: f64) -> f64 {
        let (mean, var) = self.predict(x);
        mean + beta * var.sqrt()
    }

    /// Optimize length scales using marginal likelihood gradient descent
    pub fn optimize_hyperparameters(&mut self, _iterations: usize) {
        // Simple grid search for now (more robust than gradient descent)
        if self.x_train.len() < MIN_GP_POINTS {
            return;
        }

        let mut best_ll = f64::NEG_INFINITY;
        let mut best_ls = self.length_scales.clone();

        let ls_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0];

        if USE_ARD {
            // For ARD, just optimize the mean length scale
            for &ls in &ls_values {
                self.length_scales = vec![ls; self.n_dims];
                if let Ok(()) = self.fit() {
                    let ll = self.log_marginal_likelihood();
                    if ll > best_ll {
                        best_ll = ll;
                        best_ls = self.length_scales.clone();
                    }
                }
            }
        } else {
            for &ls in &ls_values {
                self.length_scales = vec![ls];
                if let Ok(()) = self.fit() {
                    let ll = self.log_marginal_likelihood();
                    if ll > best_ll {
                        best_ll = ll;
                        best_ls = self.length_scales.clone();
                    }
                }
            }
        }

        self.length_scales = best_ls;
        let _ = self.fit();
    }

    /// Compute log marginal likelihood
    fn log_marginal_likelihood(&self) -> f64 {
        let (k_inv_y, l_chol) = match (&self.k_inv_y, &self.l_chol) {
            (Some(a), Some(l)) => (a, l),
            _ => return f64::NEG_INFINITY,
        };

        let n = self.y_train.len() as f64;
        let y = Array1::from_vec(self.y_train.clone());

        // -0.5 * y^T * K^-1 * y
        let data_fit = -0.5 * y.dot(k_inv_y);

        // -0.5 * log|K| = -sum(log(diag(L)))
        let log_det: f64 = -l_chol.diag().iter().map(|&x| x.ln()).sum::<f64>();

        // -0.5 * n * log(2*pi)
        let const_term = -0.5 * n * (2.0 * std::f64::consts::PI).ln();

        data_fit + log_det + const_term
    }

    /// Get best observed point (useful for tracking optimization progress)
    #[allow(dead_code)]
    pub fn get_best(&self) -> Option<(Vec<f64>, f64)> {
        if self.y_train.is_empty() {
            return None;
        }

        let (idx, &y_best) = self
            .y_train
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        // Denormalize
        let x_norm = &self.x_train[idx];
        let x: Vec<f64> = x_norm
            .iter()
            .zip(&self.ranges)
            .map(|(&xi, &(lo, hi))| xi * (hi - lo) + lo)
            .collect();

        Some((x, y_best))
    }

    /// Prune old points if we exceed MAX_GP_POINTS
    pub fn prune_if_needed(&mut self) {
        if self.x_train.len() <= MAX_GP_POINTS {
            return;
        }

        // Keep the best points and most recent
        let n = self.x_train.len();
        let keep = MAX_GP_POINTS;

        // Sort by y value (descending) and keep top half
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            self.y_train[j]
                .partial_cmp(&self.y_train[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top scoring and most recent
        let mut keep_set: std::collections::HashSet<usize> = std::collections::HashSet::new();

        // Top half by score
        for &idx in indices.iter().take(keep / 2) {
            keep_set.insert(idx);
        }

        // Most recent
        for idx in (n - keep / 2)..n {
            keep_set.insert(idx);
        }

        let mut keep_indices: Vec<usize> = keep_set.into_iter().collect();
        keep_indices.sort();

        let new_x: Vec<Vec<f64>> = keep_indices
            .iter()
            .map(|&i| self.x_train[i].clone())
            .collect();
        let new_y: Vec<f64> = keep_indices.iter().map(|&i| self.y_train[i]).collect();

        self.x_train = new_x;
        self.y_train = new_y;
        self.k_inv_y = None;
        self.l_chol = None;
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Cholesky decomposition (lower triangular)
fn cholesky_decomp(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }

            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    return Err("Matrix not positive definite".to_string());
                }
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Solve L * x = b (forward substitution)
fn forward_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * x[j];
        }
        x[i] = (b[i] - sum) / l[[i, i]];
    }

    x
}

/// Solve L * L^T * x = b
fn cholesky_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();

    // Forward: L * y = b
    let y = forward_solve(l, &b);

    // Backward: L^T * x = y
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        x[i] = (y[i] - sum) / l[[i, i]];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gp_basic() {
        let ranges = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut gp = GaussianProcess::new(2, &ranges);

        // Add some points
        for i in 0..30 {
            let x = vec![(i as f64) / 30.0, (i as f64) / 30.0];
            let y = -(x[0] - 0.5).powi(2) - (x[1] - 0.5).powi(2);
            gp.add_point(&x, y);
        }

        assert!(gp.fit().is_ok());

        // Predict at optimum
        let (mean, var) = gp.predict(&[0.5, 0.5]);
        assert!(mean > -0.1);
        assert!(var > 0.0);
    }
}
