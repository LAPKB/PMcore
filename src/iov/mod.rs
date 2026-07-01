//! SDE-based Inter-Occasion Variability (IOV) analysis.
//!
//! This module provides [`optimize_diffusion`], which optimizes SDE diffusion
//! (sigma) parameters for each support point independently, using the NelderMead
//! algorithm. The optimization runs in parallel over support points via rayon.
//!
//! # Workflow
//!
//! 1. Fit an ODE model with NPAG/NPOD to obtain support points (Stage 1).
//! 2. Add sigma parameter columns to the theta using [`Theta::with_added_parameter`].
//! 3. Construct an SDE model (user-provided) that maps sigma parameters to
//!    diffusion terms.
//! 4. Call [`optimize_diffusion`] to optimize sigma per support point.
//!
//! # Example
//!
//! ```ignore
//! use pmcore::prelude::*;
//! use pmcore::iov::DiffusionOptimize;
//! use pmcore::iov::DiffusionConfig;
//!
//! let r_ode = problem.fit_with(NPAG::default())?;
//! let mut joint = r_ode.get_theta()
//!     .with_added_parameter("ske", 1e-6, 1.0, 0.01)?;
//!
//! let diff = sde.optimize_diffusion(
//!     &r_ode.data(), &mut joint,
//!     &["ske"], &r_ode.error_models(),
//!     DiffusionConfig::default(),
//! )?;
//! ```

mod optimizer;

use anyhow::bail;
use rayon::prelude::*;

use pharmsol::prelude::data::AssayErrorModels;
use pharmsol::{Data, SDE};

use crate::estimation::nonparametric::Theta;

/// Configuration for SDE diffusion parameter optimization.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Maximum NelderMead iterations per support point.
    ///
    /// Default: 50. Set lower for speed, higher for difficult surfaces.
    pub max_iter: usize,

    /// Convergence tolerance on simplex standard deviation.
    ///
    /// NelderMead stops when the standard deviation of function values
    /// across the simplex vertices falls below this threshold.
    /// Default: 1e-3.
    pub sd_tolerance: f64,

    /// Fraction of the distance to the upper bound for simplex construction.
    ///
    /// The second simplex vertex is placed at `init + perturbation × (upper − init)`.
    /// This spans the simplex toward the upper bound without overshooting.
    /// Default: 0.15 (for init=0.01, bounds [0,0.5] → vertex at 0.084).
    pub initial_perturbation: f64,

    /// Number of resampled evaluations per cost function call.
    ///
    /// The SDE particle filter produces noisy likelihood estimates. Averaging
    /// over multiple evaluations reduces variance and makes NelderMead decisions
    /// reliable by giving every vertex the same precision.
    ///
    /// Default: 3. Set to 1 for speed (raw NelderMead), higher for precision.
    pub resampling_samples: usize,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            sd_tolerance: 1e-3,
            initial_perturbation: 0.15,
            resampling_samples: 3,
        }
    }
}

/// Results of SDE diffusion parameter optimization.
#[derive(Debug, Clone)]
pub struct DiffusionResult {
    /// Final log-likelihood for each support point after optimization.
    /// Length equals `theta.nspp()`.
    pub per_point_likelihood: Vec<f64>,

    /// Number of NelderMead iterations used for each point.
    pub per_point_iterations: Vec<usize>,

    /// Whether NelderMead converged within `max_iter` for each point.
    pub per_point_converged: Vec<bool>,
}

/// Trait for SDEs that support diffusion parameter optimization.
///
/// This enables method-style calls: `sde.optimize_diffusion(...)`.
pub trait DiffusionOptimize {
    /// Optimize SDE diffusion parameters for each support point independently.
    ///
    /// Modifies `theta` **in-place**: for each support point, the sigma parameter
    /// columns are replaced with values that maximize the log-likelihood of all
    /// subjects under this SDE. Primary (non-sigma) parameter values are held fixed.
    ///
    /// If `posterior` is provided, subject contributions are weighted by their
    /// posterior responsibility for each support point: `p(z_i=j)` from Stage 1.
    /// If `None`, falls back to uniform weighting.
    ///
    /// # Panics
    ///
    /// Panics if any name in `sigma_params` is not found in `theta.parameters()`.
    fn optimize_diffusion(
        &self,
        data: &Data,
        theta: &mut Theta,
        sigma_params: &[String],
        error_models: &AssayErrorModels,
        posterior: Option<&crate::estimation::nonparametric::Posterior>,
        config: DiffusionConfig,
    ) -> anyhow::Result<DiffusionResult>;
}

impl DiffusionOptimize for SDE {
    fn optimize_diffusion(
        &self,
        data: &Data,
        theta: &mut Theta,
        sigma_params: &[String],
        error_models: &AssayErrorModels,
        posterior: Option<&crate::estimation::nonparametric::Posterior>,
        config: DiffusionConfig,
    ) -> anyhow::Result<DiffusionResult> {
        optimize_diffusion(
            self,
            data,
            theta,
            sigma_params,
            error_models,
            posterior,
            config,
        )
    }
}

/// Optimize SDE diffusion parameters for each support point independently.
///
/// Free-function form of [`DiffusionOptimize::optimize_diffusion`].
/// Prefer `sde.optimize_diffusion(...)` for readability.
///
/// # Important: disable SDE caching
///
/// SDEs cache likelihood results by default. This optimization is a Monte Carlo
/// method that requires fresh random evaluations every iteration. Ensure the SDE
/// is constructed with `.disable_cache()` before passing it here. This function
/// warns (but does not error) if caching may be enabled.
pub(crate) fn optimize_diffusion(
    sde: &SDE,
    data: &Data,
    theta: &mut Theta,
    sigma_params: &[String],
    error_models: &AssayErrorModels,
    posterior: Option<&crate::estimation::nonparametric::Posterior>,
    config: DiffusionConfig,
) -> anyhow::Result<DiffusionResult> {
    let n_spp = theta.nspp();
    if n_spp == 0 {
        bail!("theta has no support points");
    }

    // Resolve sigma parameter indices in theta
    let sigma_indices: Vec<usize> = sigma_params
        .iter()
        .map(|name| {
            theta
                .parameters()
                .iter()
                .position(|p| p.name.as_str() == name.as_str())
                .unwrap_or_else(|| {
                    panic!(
                        "sigma parameter '{}' not found in theta parameters: {:?}",
                        name,
                        theta.parameters().names()
                    )
                })
        })
        .collect();

    // Identify primary parameter indices (all others)
    let n_total = theta.matrix().ncols();
    let sigma_set: std::collections::HashSet<usize> = sigma_indices.iter().copied().collect();
    let primary_indices: Vec<usize> = (0..n_total).filter(|i| !sigma_set.contains(i)).collect();

    // Check for sigma initialized to zero
    for &si in &sigma_indices {
        for r in 0..n_spp {
            if theta.matrix()[(r, si)] == 0.0 {
                tracing::warn!(
                    "sigma parameter at column {} (support point {}) initialized to 0.0; \
                     the SDE degenerates to an ODE at sigma=0. Consider using a small \
                     non-zero initial value (e.g., 0.01)",
                    si,
                    r
                );
            }
        }
    }

    // Extract sigma parameter bounds for simplex construction
    let sigma_bounds: Vec<(f64, f64)> = sigma_indices
        .iter()
        .map(|&si| {
            let bp = &theta.parameters().items[si];
            (bp.lower, bp.upper)
        })
        .collect();

    // Parallel optimization over support points — each SP optimized independently.
    // If a Stage 1 posterior is provided, subject contributions are weighted by
    // p(z_i=j), correctly modeling population structure without inner-loop Burke.
    let results: Vec<optimizer::OptimizationOutcome> = (0..n_spp)
        .into_par_iter()
        .map(|i| {
            let primary: Vec<f64> = primary_indices
                .iter()
                .map(|&pi| theta.matrix()[(i, pi)])
                .collect();

            let sigma_init: Vec<f64> = sigma_indices
                .iter()
                .map(|&si| theta.matrix()[(i, si)])
                .collect();

            // Extract posterior responsibilities for this SP (if available)
            let responsibilities: Option<Vec<f64>> = posterior.map(|p| {
                (0..data.subjects().len())
                    .map(|s| p.matrix()[(s, i)])
                    .collect()
            });
            let resp_slice: Option<&[f64]> = responsibilities.as_deref();

            let cost = optimizer::SigmaCost::new(
                sde,
                data,
                &primary,
                &primary_indices,
                &sigma_indices,
                error_models,
                resp_slice,
            );

            optimizer::optimize_sigma(cost, &sigma_init, &sigma_bounds, &config)
        })
        .collect();

    // Update theta with optimized sigma values
    let mut per_point_likelihood = Vec::with_capacity(n_spp);
    let mut per_point_iterations = Vec::with_capacity(n_spp);
    let mut per_point_converged = Vec::with_capacity(n_spp);

    for (i, outcome) in results.iter().enumerate() {
        for (j, &si) in sigma_indices.iter().enumerate() {
            theta.matrix_mut()[(i, si)] = outcome.optimized_params[j];
        }
        per_point_likelihood.push(-outcome.final_cost);
        per_point_iterations.push(outcome.iterations);
        per_point_converged.push(outcome.converged);
    }

    let n_converged = per_point_converged.iter().filter(|&&c| c).count();
    tracing::info!(
        "SDE IOV optimization: {}/{} support points converged, \
         mean iterations: {:.1}, mean log-likelihood: {:.2}",
        n_converged,
        n_spp,
        per_point_iterations.iter().sum::<usize>() as f64 / n_spp.max(1) as f64,
        per_point_likelihood.iter().sum::<f64>() / n_spp.max(1) as f64,
    );

    Ok(DiffusionResult {
        per_point_likelihood,
        per_point_iterations,
        per_point_converged,
    })
}
