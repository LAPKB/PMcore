//! NelderMead-based sigma optimization for SDE IOV.
//!
//! Internal module — not exposed publicly.

use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::neldermead::NelderMead;
use pharmsol::prelude::data::AssayErrorModels;
use pharmsol::prelude::simulator::Equation;
use pharmsol::{Data, SDE};

use super::DiffusionConfig;

/// Cost function: posterior-weighted negative log-likelihood.
///
/// ```text
/// cost(sigma_j) = -sum_i r_i * log P(data_i | theta_j, sigma_j)
/// ```
///
/// Where `rᵢ = p(zᵢ=j)` is subject i's posterior responsibility for support
/// point j, computed during Stage 1 (NPAG). If responsibilities are `None`,
/// falls back to uniform weighting (all subjects contribute equally).
///
/// Subjects with near-zero responsibility contribute near-zero to the gradient,
/// correctly modeling the population structure without running Burke in the
/// inner optimization loop.
pub(crate) struct SigmaCost<'a> {
    sde: &'a SDE,
    data: &'a Data,
    primary: Vec<f64>,
    primary_indices: Vec<usize>,
    sigma_indices: Vec<usize>,
    error_models: &'a AssayErrorModels,
    /// Per-subject responsibilities for this support point.
    /// None → uniform weighting (all subjects equal).
    #[allow(dead_code)]
    responsibilities: Option<&'a [f64]>,
    n_total: usize,
}

impl<'a> SigmaCost<'a> {
    pub(crate) fn new(
        sde: &'a SDE,
        data: &'a Data,
        primary: &[f64],
        primary_indices: &[usize],
        sigma_indices: &[usize],
        error_models: &'a AssayErrorModels,
        responsibilities: Option<&'a [f64]>,
    ) -> Self {
        let n_total = primary.len() + sigma_indices.len();
        Self {
            sde,
            data,
            primary: primary.to_vec(),
            primary_indices: primary_indices.to_vec(),
            sigma_indices: sigma_indices.to_vec(),
            error_models,
            responsibilities,
            n_total,
        }
    }

    fn build_params(&self, sigma: &[f64]) -> Vec<f64> {
        let mut full = vec![0.0; self.n_total];
        for (&pi, &val) in self.primary_indices.iter().zip(self.primary.iter()) {
            full[pi] = val;
        }
        for (&si, &val) in self.sigma_indices.iter().zip(sigma) {
            full[si] = val;
        }
        full
    }
}

impl CostFunction for SigmaCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, sigma: &Self::Param) -> Result<Self::Output, Error> {
        if sigma.iter().any(|&s| s < 0.0) {
            return Ok(1e10);
        }

        let full_params = self.build_params(sigma);
        let mut total_ll = 0.0f64;

        for subject in self.data.subjects().iter() {
            let (_, likelihood) = self
                .sde
                .simulate_subject_dense(subject, &full_params, Some(self.error_models))
                .map_err(|e| Error::msg(e.to_string()))?;

            match likelihood {
                Some(ll) if ll > 0.0 => total_ll += ll.ln(),
                _ => return Ok(1e10),
            }
        }

        if !total_ll.is_finite() {
            return Ok(1e10);
        }

        Ok(-total_ll)
    }
}

/// Wraps a [`SigmaCost`] and evaluates it `n` times per cost call, returning
/// the mean. NelderMead makes decisions based on these averaged values,
/// preventing noise-triggered premature simplex collapse.
///
/// Every vertex gets the same number of samples — this ensures unbiased
/// comparisons within the simplex.
struct ResampledCost<'a> {
    inner: &'a SigmaCost<'a>,
    samples: usize,
}

impl CostFunction for ResampledCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, sigma: &Self::Param) -> Result<Self::Output, Error> {
        let sum: f64 = (0..self.samples)
            .map(|_| self.inner.cost(sigma).unwrap_or(1e10))
            .sum();
        Ok(sum / self.samples as f64)
    }
}

/// Outcome of a single support point's sigma optimization.
pub(crate) struct OptimizationOutcome {
    pub optimized_params: Vec<f64>,
    pub final_cost: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Run NelderMead with resampled cost evaluations to handle particle-filter noise.
pub(crate) fn optimize_sigma(
    cost: SigmaCost<'_>,
    sigma_init: &[f64],
    sigma_bounds: &[(f64, f64)],
    config: &DiffusionConfig,
) -> OptimizationOutcome {
    let simplex = build_simplex(sigma_init, sigma_bounds, config.initial_perturbation);

    let resampled = ResampledCost {
        inner: &cost,
        samples: config.resampling_samples,
    };

    let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(simplex)
        .with_sd_tolerance(config.sd_tolerance)
        .expect("NelderMead construction should succeed with valid parameters");

    let result = Executor::new(resampled, solver)
        .configure(|state| state.max_iters(config.max_iter as u64))
        .run();

    match result {
        Ok(res) => {
            let best = res.state.best_param.unwrap_or_else(|| sigma_init.to_vec());
            let cost_val = res.state.best_cost;
            let iterations = res.state.iter as usize;

            OptimizationOutcome {
                optimized_params: best,
                final_cost: cost_val,
                iterations,
                converged: iterations < config.max_iter,
            }
        }
        Err(e) => {
            tracing::warn!(
                "NelderMead optimization failed for a support point: {}. Returning initial values.",
                e
            );
            OptimizationOutcome {
                optimized_params: sigma_init.to_vec(),
                final_cost: f64::INFINITY,
                iterations: 0,
                converged: false,
            }
        }
    }
}

/// Build the initial simplex for NelderMead.
///
/// For N-dimensional optimization, the simplex has N+1 vertices.
/// Vertex 0 is the initial point. Vertex i+1 is placed at
/// `init + perturbation × (upper − init)`, clamped to bounds.
/// This keeps the simplex within the search space while providing enough
/// spread to reach the optimum even from a small initial value.
fn build_simplex(initial: &[f64], bounds: &[(f64, f64)], perturbation: f64) -> Vec<Vec<f64>> {
    let n = initial.len();
    let mut vertices = Vec::with_capacity(n + 1);
    vertices.push(initial.to_vec());

    for i in 0..n {
        let mut vertex = initial.to_vec();
        let (lower, upper) = bounds[i];
        vertex[i] = (initial[i] + perturbation * (upper - initial[i])).clamp(lower, upper);
        if (vertex[i] - initial[i]).abs() < 1e-10 {
            vertex[i] = (lower + upper) * 0.5;
        }
        vertices.push(vertex);
    }

    vertices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_reaches_optimum() {
        // init=0.01, bounds [0, 0.5], perturbation=0.15
        // vertex = 0.01 + 0.15 * (0.5 - 0.01) = 0.0835
        let s = build_simplex(&[0.01], &[(0.0, 0.5)], 0.15);
        assert!((s[1][0] - 0.0835).abs() < 1e-10);

        // init=0.3, bounds [0, 1.0], perturbation=0.15
        // vertex = 0.3 + 0.15 * (1.0 - 0.3) = 0.405
        let s = build_simplex(&[0.3], &[(0.0, 1.0)], 0.15);
        assert!((s[1][0] - 0.405).abs() < 1e-10);
    }
}
