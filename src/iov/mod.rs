//! SDE diffusion optimization for nonparametric support points.
//!
//! This module provides [`optimize_diffusion`](crate::iov::DiffusionOptimize::optimize_diffusion),
//! which optimizes SDE diffusion (sigma) parameters for each support point independently, using the
//! NelderMead algorithm. The optimization runs in parallel over support points via rayon. Each
//! objective evaluation uses sequential observation-conditioned particle filtering.
//!
//! # Workflow
//!
//! 1. Fit an ODE model with NPAG/NPOD to obtain support points (Stage 1).
//! 2. Add sigma parameter columns to the theta using
//!    [`Theta::with_added_parameter`](crate::estimation::nonparametric::Theta::with_added_parameter).
//! 3. Construct an SDE model (user-provided) that maps sigma parameters to
//!    diffusion terms.
//! 4. Call [`optimize_diffusion`](crate::iov::DiffusionOptimize::optimize_diffusion) to optimize sigma per support point.
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
//!     &["ske".to_string()], &r_ode.error_models(),
//!     None, DiffusionConfig::default(),
//! )?;
//! ```

mod optimizer;

use anyhow::{bail, Context};
use pharmsol::{Data, SDE};
use rayon::prelude::*;

use crate::estimation::nonparametric::Theta;
use crate::AssayErrorModels;

#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    pub max_iter: usize,
    pub sd_tolerance: f64,
    pub initial_perturbation: f64,
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

impl DiffusionConfig {
    fn validate(&self) -> anyhow::Result<()> {
        if self.max_iter == 0 {
            bail!("max_iter must be greater than zero");
        }
        if !self.sd_tolerance.is_finite() || self.sd_tolerance <= 0.0 {
            bail!("sd_tolerance must be finite and greater than zero");
        }
        if !self.initial_perturbation.is_finite()
            || self.initial_perturbation <= 0.0
            || self.initial_perturbation > 1.0
        {
            bail!("initial_perturbation must be finite and in the interval (0, 1]");
        }
        if self.resampling_samples == 0 {
            bail!("resampling_samples must be greater than zero");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DiffusionResult {
    pub per_point_likelihood: Vec<f64>,
    pub per_point_iterations: Vec<usize>,
    pub per_point_converged: Vec<bool>,
}

pub trait DiffusionOptimize {
    /// Optimize diffusion columns in place while holding all other columns fixed.
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
        config.validate()?;

        let sde = self;
        let support_points = theta.nspp();
        if support_points == 0 {
            bail!("theta has no support points");
        }
        if sigma_params.is_empty() {
            bail!("at least one diffusion parameter must be selected");
        }

        let parameter_names = theta.parameters().names();
        let mut sigma_set = std::collections::HashSet::with_capacity(sigma_params.len());
        let sigma_indices = sigma_params
            .iter()
            .map(|name| {
                let index = parameter_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("diffusion parameter `{name}` is not in theta")
                    })?;
                if !sigma_set.insert(index) {
                    bail!("duplicate diffusion parameter `{name}`");
                }
                Ok(index)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let primary_indices = (0..theta.matrix().ncols())
            .filter(|index| !sigma_set.contains(index))
            .collect::<Vec<_>>();
        let sigma_bounds = sigma_indices
            .iter()
            .map(|&index| {
                let parameter = &theta.parameters().items[index];
                if !parameter.lower.is_finite()
                    || !parameter.upper.is_finite()
                    || parameter.lower < 0.0
                    || parameter.lower > parameter.upper
                {
                    bail!(
                        "diffusion parameter `{}` must have finite nonnegative inclusive bounds, got [{}, {}]",
                        parameter.name,
                        parameter.lower,
                        parameter.upper
                    );
                }
                Ok((parameter.lower, parameter.upper))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        for support_index in 0..support_points {
            for (sigma_position, &parameter_index) in sigma_indices.iter().enumerate() {
                let value = theta.matrix()[(support_index, parameter_index)];
                let (lower, upper) = sigma_bounds[sigma_position];
                if !value.is_finite() || value < lower || value > upper {
                    bail!(
                        "initial diffusion parameter `{}` for support point {} must be finite and within inclusive bounds [{}, {}], got {}",
                        parameter_names[parameter_index],
                        support_index,
                        lower,
                        upper,
                        value
                    );
                }
            }
        }

        if let Some(posterior) = posterior {
            let matrix = posterior.matrix();
            let subject_count = data.subjects().len();
            if matrix.nrows() != subject_count {
                bail!(
                    "posterior row count ({}) must match data subject count ({}) for diffusion optimization",
                    matrix.nrows(),
                    subject_count
                );
            }
            if matrix.ncols() != support_points {
                bail!(
                    "posterior column count ({}) must match theta support point count ({}) for diffusion optimization",
                    matrix.ncols(),
                    support_points
                );
            }
            for row in 0..matrix.nrows() {
                for column in 0..matrix.ncols() {
                    let value = matrix[(row, column)];
                    if !value.is_finite() {
                        bail!(
                            "posterior value at row {}, column {} must be finite, got {}",
                            row,
                            column,
                            value
                        );
                    }
                }
            }
        }

        let particle_count = sde
            .metadata()
            .and_then(|metadata| metadata.particles())
            .ok_or_else(|| anyhow::anyhow!("SDE metadata must declare its particle count"))?;

        let outcomes = (0..support_points)
            .into_par_iter()
            .map(|support_index| {
                let primary = primary_indices
                    .iter()
                    .map(|&index| theta.matrix()[(support_index, index)])
                    .collect::<Vec<_>>();
                let sigma = sigma_indices
                    .iter()
                    .map(|&index| theta.matrix()[(support_index, index)])
                    .collect::<Vec<_>>();
                let responsibilities = posterior.map(|posterior| {
                    (0..data.subjects().len())
                        .map(|subject| posterior.matrix()[(subject, support_index)])
                        .collect::<Vec<_>>()
                });
                let cost = optimizer::SigmaCost::new(
                    sde,
                    data,
                    &primary,
                    &primary_indices,
                    &sigma_indices,
                    parameter_names.clone(),
                    error_models,
                    responsibilities.as_deref(),
                    particle_count,
                );
                optimizer::optimize_sigma(cost, &sigma, &sigma_bounds, &config).with_context(|| {
                    format!("diffusion optimization failed for support point {support_index}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut result = DiffusionResult {
            per_point_likelihood: Vec::with_capacity(support_points),
            per_point_iterations: Vec::with_capacity(support_points),
            per_point_converged: Vec::with_capacity(support_points),
        };
        for (support_index, outcome) in outcomes.into_iter().enumerate() {
            for (sigma_position, &parameter_index) in sigma_indices.iter().enumerate() {
                theta.matrix_mut()[(support_index, parameter_index)] =
                    outcome.optimized_params[sigma_position];
            }
            result.per_point_likelihood.push(-outcome.final_cost);
            result.per_point_iterations.push(outcome.iterations);
            result.per_point_converged.push(outcome.converged);
        }
        Ok(result)
    }
}
