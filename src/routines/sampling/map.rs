//! MAP (Maximum A Posteriori) Estimation for f-SAEM
//!
//! This module implements individual MAP estimation used in the fourth kernel
//! of the f-SAEM algorithm. The MAP estimate provides a good starting point
//! and proposal distribution for MCMC sampling.
//!
//! # Mathematical Background
//!
//! For subject i, we find the MAP estimate by maximizing the conditional posterior:
//!
//! ```text
//! η̂ᵢ = argmax_η log p(η | yᵢ, θ)
//!    = argmax_η [log p(yᵢ | η) + log p(η | Ω)]
//! ```
//!
//! The objective function (to minimize) is:
//!
//! ```text
//! U(η) = -log p(y | η) + 0.5 * ηᵀ Ω⁻¹ η
//! ```
//!
//! # Laplace Approximation
//!
//! The posterior covariance at the MAP is approximated by:
//!
//! ```text
//! Γ = [∇²U(η̂)]⁻¹ = [J(η̂)ᵀ Σ⁻¹ J(η̂) + Ω⁻¹]⁻¹
//! ```
//!
//! where J is the Jacobian of the model predictions with respect to η.

use anyhow::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Col, Mat};

use super::kernels::MapEstimate;

/// Configuration for MAP estimation
#[derive(Debug, Clone)]
pub struct MapConfig {
    /// Maximum number of optimization iterations
    pub max_iter: usize,
    /// Convergence tolerance for gradient norm
    pub gradient_tol: f64,
    /// Convergence tolerance for function value change
    pub function_tol: f64,
    /// Step size for numerical gradient computation
    pub gradient_step: f64,
    /// Initial step size for line search
    pub initial_step: f64,
    /// Armijo condition parameter
    pub armijo_c: f64,
    /// Backtracking factor for line search
    pub backtrack_factor: f64,
    /// Regularization for Hessian (ensure positive definiteness)
    pub hessian_regularization: f64,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            gradient_tol: 1e-4,
            function_tol: 1e-6,
            gradient_step: 1e-4,
            initial_step: 1.0,
            armijo_c: 1e-4,
            backtrack_factor: 0.5,
            hessian_regularization: 1e-6,
        }
    }
}

/// MAP estimator for individual parameters
pub struct MapEstimator {
    /// Configuration
    config: MapConfig,
    /// Number of parameters
    n_params: usize,
    /// Population covariance inverse (Ω⁻¹)
    omega_inv: Mat<f64>,
    /// Residual error variance (for Gaussian likelihood)
    sigma_sq: f64,
}

impl MapEstimator {
    /// Create a new MAP estimator
    pub fn new(omega_inv: Mat<f64>, sigma_sq: f64, config: MapConfig) -> Self {
        let n_params = omega_inv.nrows();
        Self {
            config,
            n_params,
            omega_inv,
            sigma_sq,
        }
    }

    /// Update the omega inverse matrix
    pub fn update_omega_inv(&mut self, omega_inv: Mat<f64>) {
        self.omega_inv = omega_inv;
    }

    /// Update the residual error variance
    pub fn update_sigma_sq(&mut self, sigma_sq: f64) {
        self.sigma_sq = sigma_sq;
    }

    /// Compute MAP estimate for a single subject
    ///
    /// # Arguments
    ///
    /// * `eta_init` - Initial guess for η
    /// * `mean_phi` - Population mean φ (including covariate effects)
    /// * `objective_fn` - Function computing negative log-likelihood given φ = η + mean_phi
    /// * `gradient_fn` - Optional function computing gradient of neg-log-likelihood
    ///
    /// # Returns
    ///
    /// MAP estimate including covariance from Laplace approximation
    pub fn estimate<F, G>(
        &self,
        eta_init: &Col<f64>,
        mean_phi: &Col<f64>,
        objective_fn: F,
        gradient_fn: Option<G>,
    ) -> Result<MapEstimate>
    where
        F: Fn(&Col<f64>) -> f64,
        G: Fn(&Col<f64>) -> Col<f64>,
    {
        let n = self.n_params;

        // Wrapper that computes full objective (likelihood + prior)
        let full_objective = |eta: &Col<f64>| -> f64 {
            let phi = self.add_vectors(eta, mean_phi);
            let neg_log_lik = objective_fn(&phi);
            let prior_term = self.quadratic_form(eta, &self.omega_inv);
            neg_log_lik + 0.5 * prior_term
        };

        // Run optimization
        let mut eta = eta_init.clone();
        let mut f_val = full_objective(&eta);

        for _iter in 0..self.config.max_iter {
            // Compute gradient (numerically if not provided)
            let grad = if let Some(ref grad_fn) = gradient_fn {
                let phi = self.add_vectors(&eta, mean_phi);
                let grad_lik = grad_fn(&phi);
                // Add prior gradient: Ω⁻¹ η
                let mut full_grad = grad_lik;
                for i in 0..n {
                    let mut prior_grad_i = 0.0;
                    for j in 0..n {
                        prior_grad_i += self.omega_inv[(i, j)] * eta[j];
                    }
                    full_grad[i] += prior_grad_i;
                }
                full_grad
            } else {
                self.numerical_gradient(&eta, &full_objective)
            };

            // Check convergence
            let grad_norm = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
            if grad_norm < self.config.gradient_tol {
                break;
            }

            // Compute search direction (negative gradient for steepest descent)
            // Could improve with BFGS, but steepest descent is simple and robust
            let direction: Col<f64> = Col::from_fn(n, |i| -grad[i]);

            // Line search with backtracking
            let mut step = self.config.initial_step;
            let f_init = f_val;
            let slope = grad
                .iter()
                .zip(direction.iter())
                .map(|(g, d)| g * d)
                .sum::<f64>();

            for _ in 0..20 {
                let eta_new: Col<f64> = Col::from_fn(n, |i| eta[i] + step * direction[i]);
                let f_new = full_objective(&eta_new);

                // Armijo condition
                if f_new <= f_init + self.config.armijo_c * step * slope {
                    eta = eta_new;
                    f_val = f_new;
                    break;
                }
                step *= self.config.backtrack_factor;
            }

            // Check function value convergence
            if (f_init - f_val).abs() < self.config.function_tol {
                break;
            }
        }

        // Compute Hessian at MAP for Laplace approximation
        let hessian = self.numerical_hessian(&eta, &full_objective);

        // Invert Hessian to get posterior covariance
        let covariance = self.invert_hessian(&hessian)?;

        MapEstimate::new(eta, covariance)
    }

    /// Estimate MAP with prediction-based Jacobian computation
    ///
    /// This method computes the Laplace approximation covariance using the
    /// Jacobian of the model predictions, which is more efficient than
    /// computing the full Hessian numerically.
    ///
    /// # Arguments
    ///
    /// * `eta_init` - Initial guess for η
    /// * `mean_phi` - Population mean φ
    /// * `objective_fn` - Function computing negative log-likelihood
    /// * `prediction_fn` - Function computing model predictions given φ
    ///
    /// # Returns
    ///
    /// MAP estimate with covariance from Jacobian-based Laplace approximation
    pub fn estimate_with_jacobian<F, P>(
        &self,
        eta_init: &Col<f64>,
        mean_phi: &Col<f64>,
        objective_fn: F,
        prediction_fn: P,
    ) -> Result<MapEstimate>
    where
        F: Fn(&Col<f64>) -> f64,
        P: Fn(&Col<f64>) -> Col<f64>,
    {
        // First, get the MAP estimate using standard optimization
        let map_result = self.estimate(
            eta_init,
            mean_phi,
            &objective_fn,
            None::<fn(&Col<f64>) -> Col<f64>>,
        )?;
        let eta_map = &map_result.eta;

        // Compute Jacobian of predictions at MAP
        let phi_map = self.add_vectors(eta_map, mean_phi);
        let jacobian = self.compute_jacobian(&phi_map, &prediction_fn);

        // Compute Laplace approximation covariance
        // Γ = [J' Σ⁻¹ J + Ω⁻¹]⁻¹
        let covariance = self.laplace_covariance(&jacobian)?;

        MapEstimate::new(eta_map.clone(), covariance)
    }

    /// Compute Jacobian of predictions with respect to φ
    fn compute_jacobian<P>(&self, phi: &Col<f64>, prediction_fn: &P) -> Mat<f64>
    where
        P: Fn(&Col<f64>) -> Col<f64>,
    {
        let n = self.n_params;
        let f0 = prediction_fn(phi);
        let n_obs = f0.nrows();

        let mut jacobian = Mat::zeros(n_obs, n);
        let h = self.config.gradient_step;

        for j in 0..n {
            let mut phi_plus = phi.clone();
            phi_plus[j] += h;
            let f_plus = prediction_fn(&phi_plus);

            for i in 0..n_obs {
                jacobian[(i, j)] = (f_plus[i] - f0[i]) / h;
            }
        }

        jacobian
    }

    /// Compute Laplace approximation covariance from Jacobian
    /// Γ = [J' Σ⁻¹ J + Ω⁻¹]⁻¹
    fn laplace_covariance(&self, jacobian: &Mat<f64>) -> Result<Mat<f64>> {
        let n = self.n_params;
        let n_obs = jacobian.nrows();

        // Compute J' J / σ²
        let mut jtj_over_sigma = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n_obs {
                    sum += jacobian[(k, i)] * jacobian[(k, j)];
                }
                jtj_over_sigma[(i, j)] = sum / self.sigma_sq;
            }
        }

        // Add Ω⁻¹
        let mut fisher_info = jtj_over_sigma;
        for i in 0..n {
            for j in 0..n {
                fisher_info[(i, j)] += self.omega_inv[(i, j)];
            }
        }

        // Invert to get covariance
        self.invert_hessian(&fisher_info)
    }

    /// Compute numerical gradient using central differences
    fn numerical_gradient<F>(&self, x: &Col<f64>, f: &F) -> Col<f64>
    where
        F: Fn(&Col<f64>) -> f64,
    {
        let n = x.nrows();
        let h = self.config.gradient_step;
        let mut grad = Col::zeros(n);

        for i in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h;
            x_minus[i] -= h;

            grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * h);
        }

        grad
    }

    /// Compute numerical Hessian using central differences
    fn numerical_hessian<F>(&self, x: &Col<f64>, f: &F) -> Mat<f64>
    where
        F: Fn(&Col<f64>) -> f64,
    {
        let n = x.nrows();
        let h = self.config.gradient_step;
        let mut hessian = Mat::zeros(n, n);

        let f0 = f(x);

        for i in 0..n {
            // Diagonal elements
            let mut x_pp = x.clone();
            let mut x_mm = x.clone();
            x_pp[i] += h;
            x_mm[i] -= h;

            hessian[(i, i)] = (f(&x_pp) - 2.0 * f0 + f(&x_mm)) / (h * h);

            // Off-diagonal elements (symmetric)
            for j in (i + 1)..n {
                let mut x_pp = x.clone();
                let mut x_pm = x.clone();
                let mut x_mp = x.clone();
                let mut x_mm = x.clone();

                x_pp[i] += h;
                x_pp[j] += h;
                x_pm[i] += h;
                x_pm[j] -= h;
                x_mp[i] -= h;
                x_mp[j] += h;
                x_mm[i] -= h;
                x_mm[j] -= h;

                let mixed = (f(&x_pp) - f(&x_pm) - f(&x_mp) + f(&x_mm)) / (4.0 * h * h);
                hessian[(i, j)] = mixed;
                hessian[(j, i)] = mixed;
            }
        }

        hessian
    }

    /// Invert Hessian with regularization for positive definiteness
    fn invert_hessian(&self, hessian: &Mat<f64>) -> Result<Mat<f64>> {
        let n = hessian.nrows();
        let reg = self.config.hessian_regularization;

        // Add regularization to diagonal
        let mut h_reg = hessian.clone();
        for i in 0..n {
            h_reg[(i, i)] += reg;
        }

        // Try Cholesky-based inversion using faer
        match h_reg.llt(faer::Side::Lower) {
            Ok(llt) => Ok(llt.inverse()),
            Err(_) => {
                // Fallback: diagonal approximation
                let mut inv = Mat::zeros(n, n);
                for i in 0..n {
                    let diag = h_reg[(i, i)].abs().max(reg);
                    inv[(i, i)] = 1.0 / diag;
                }
                Ok(inv)
            }
        }
    }

    /// Compute quadratic form xᵀ A x
    fn quadratic_form(&self, x: &Col<f64>, a: &Mat<f64>) -> f64 {
        let n = x.nrows();
        let mut result = 0.0;
        for i in 0..n {
            for j in 0..n {
                result += x[i] * a[(i, j)] * x[j];
            }
        }
        result
    }

    /// Add two vectors
    fn add_vectors(&self, a: &Col<f64>, b: &Col<f64>) -> Col<f64> {
        let n = a.nrows();
        Col::from_fn(n, |i| a[i] + b[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_estimator_creation() {
        let omega_inv = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let config = MapConfig::default();
        let estimator = MapEstimator::new(omega_inv, 1.0, config);
        assert_eq!(estimator.n_params, 2);
    }

    #[test]
    fn test_numerical_gradient() {
        let omega_inv = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let config = MapConfig::default();
        let estimator = MapEstimator::new(omega_inv, 1.0, config);

        // Test gradient of f(x) = x[0]^2 + x[1]^2
        let f = |x: &Col<f64>| x[0] * x[0] + x[1] * x[1];
        let x = Col::from_fn(2, |i| (i + 1) as f64);

        let grad = estimator.numerical_gradient(&x, &f);

        // Expected gradient: [2*x[0], 2*x[1]] = [2, 4]
        assert!((grad[0] - 2.0).abs() < 1e-4);
        assert!((grad[1] - 4.0).abs() < 1e-4);
    }
}
