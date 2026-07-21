//! Error-model factor optimization used by non-parametric algorithms.

use anyhow::Result;
use pharmsol::prelude::{data::Data, simulator::Equation};
use serde::{Deserialize, Serialize};

use crate::estimation::nonparametric::{calculate_psi, ipm::burke, Psi, Theta, Weights};
use crate::AssayErrorModels;

/// Configuration for the error-model factor (gamma/lambda) optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorOptimConfig {
    /// Initial and reset perturbation step applied to each error-model factor.
    pub step: f64,
    /// Minimum perturbation step; once the adaptive step reaches this value it is reset to `step`.
    pub min_step: f64,
    /// Factor by which the step grows after a successful improvement.
    pub growth: f64,
    /// Factor by which the step shrinks after each optimization pass.
    pub shrink: f64,
}

impl Default for ErrorOptimConfig {
    fn default() -> Self {
        Self {
            step: 0.1,
            min_step: 0.01,
            growth: 4.0,
            shrink: 0.5,
        }
    }
}

impl ErrorOptimConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    pub fn min_step(mut self, min_step: f64) -> Self {
        self.min_step = min_step;
        self
    }

    pub fn growth(mut self, growth: f64) -> Self {
        self.growth = growth;
        self
    }

    pub fn shrink(mut self, shrink: f64) -> Self {
        self.shrink = shrink;
        self
    }
}

/// Perform one pass of the error-model factor (gamma/lambda) optimization.
///
/// For each optimizable error model, the factor is perturbed up and down,
/// `psi` is recomputed for each perturbation, and the IPM ([`burke`]) is run.
/// The direction with the highest objective function that also improves on the
/// current `objf` is adopted, updating `error_models`, `objf`, `lambda`, and
/// `psi` in place. The per-output adaptive step in `gamma_delta` is grown on
/// improvement and shrunk every pass, resetting once it falls to `config.min_step`.
///
/// A failure of the IPM for one perturbation direction is treated as a warning
/// (that direction is skipped); if both directions fail the factor is left
/// unchanged for this pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_error_models<E: Equation + Send + 'static>(
    equation: &E,
    data: &Data,
    theta: &Theta,
    error_models: &mut AssayErrorModels,
    gamma_delta: &mut [f64],
    objf: &mut f64,
    lambda: &mut Weights,
    psi: &mut Psi,
    config: &ErrorOptimConfig,
) -> Result<()> {
    error_models
        .clone()
        .iter_mut()
        .filter_map(|(outeq, em)| {
            if em.optimize() {
                Some((outeq, em))
            } else {
                None
            }
        })
        .try_for_each(|(outeq, em)| -> Result<()> {
            // Optimize the best value of gamma/lambda

            let gamma_up = em.factor()? * (1.0 + gamma_delta[outeq]);
            let gamma_down = em.factor()? / (1.0 + gamma_delta[outeq]);

            let mut error_model_up = error_models.clone();
            error_model_up.set_factor(outeq, gamma_up)?;

            let mut error_model_down = error_models.clone();
            error_model_down.set_factor(outeq, gamma_down)?;

            let psi_up = calculate_psi(equation, data, theta, &error_model_up, false)?;
            let psi_down = calculate_psi(equation, data, theta, &error_model_down, false)?;

            // Treat errors in IPM, such as subjects with zero probability or non-finite
            // likelihoods as warnings, and continue with the other direction. If both
            // directions fail, we will not update the error model factor.
            let up = match burke(&psi_up) {
                Ok((lambda, objf)) => Some((lambda, objf)),
                Err(err) => {
                    tracing::warn!(
                        "Error in IPM during optim (up) for outeq {}: {:?}",
                        outeq,
                        err
                    );
                    None
                }
            };
            let down = match burke(&psi_down) {
                Ok((lambda, objf)) => Some((lambda, objf)),
                Err(err) => {
                    tracing::warn!(
                        "Error in IPM during optim (down) for outeq {}: {:?}",
                        outeq,
                        err
                    );
                    None
                }
            };

            // Select the best improving candidate (if any) over the current
            // objective. Among the two directions, the one with the higher
            // objective function wins.
            let mut best: Option<(f64, Weights, Psi, f64)> = None;
            if let Some((lambda_up, objf_up)) = up {
                if objf_up > *objf {
                    best = Some((objf_up, lambda_up, psi_up, gamma_up));
                }
            }
            if let Some((lambda_down, objf_down)) = down {
                let threshold = best.as_ref().map_or(*objf, |(o, ..)| *o);
                if objf_down > threshold {
                    best = Some((objf_down, lambda_down, psi_down, gamma_down));
                }
            }
            if let Some((new_objf, new_lambda, new_psi, gamma)) = best {
                error_models.set_factor(outeq, gamma)?;
                *objf = new_objf;
                gamma_delta[outeq] *= config.growth;
                *lambda = new_lambda;
                *psi = new_psi;
            }
            gamma_delta[outeq] *= config.shrink;
            if gamma_delta[outeq] <= config.min_step {
                gamma_delta[outeq] = config.step;
            }
            Ok(())
        })?;

    Ok(())
}
