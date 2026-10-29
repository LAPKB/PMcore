//! Error-model factor optimization used by non-parametric algorithms.

use anyhow::Result;
use pharmsol::prelude::{data::AssayErrorModels, simulator::Equation};
use pharmsol::OutputLabel;
use serde::{Deserialize, Serialize};

use super::fit::FitState;
use super::solution::Solution;
use crate::estimation::nonparametric::calculate_psi;

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

/// Optimizes the error-model factors (gamma/lambda) of a fit, one pass per cycle.
///
/// Holds the per-output adaptive step that carries over between cycles.
#[derive(Debug)]
pub(crate) struct ErrorModelOptimizer {
    config: ErrorOptimConfig,
    /// Current relative perturbation step for each error model, in iteration order.
    steps: Vec<f64>,
}

impl ErrorModelOptimizer {
    pub(crate) fn new(config: ErrorOptimConfig, error_models: &AssayErrorModels) -> Self {
        let steps = vec![config.step; error_models.len()];
        Self { config, steps }
    }

    /// Run one optimization pass over every optimizable error model in `state`.
    ///
    /// Each factor is perturbed up and down, and the current support points are re-solved
    /// under both. The direction with the highest log-likelihood is adopted if it improves on
    /// the current one. The step grows on improvement and shrinks every pass, resetting to
    /// `config.step` once it falls to `config.min_step`.
    ///
    /// An IPM failure in one direction is logged and that direction is skipped; if both fail
    /// the factor is left unchanged for this pass.
    pub(crate) fn optimize<E: Equation>(&mut self, state: &mut FitState<E>) -> Result<()> {
        let targets: Vec<(usize, OutputLabel, f64)> = state
            .error_models()
            .iter()
            .enumerate()
            .filter(|(_, (_, em))| em.optimize())
            .map(|(slot, (outeq, em))| Ok((slot, outeq.clone(), em.factor()?)))
            .collect::<Result<_>>()?;

        for (slot, outeq, factor) in targets {
            let step = &mut self.steps[slot];
            let gamma_up = factor * (1.0 + *step);
            let gamma_down = factor / (1.0 + *step);

            let up = evaluate_factor(state, &outeq, gamma_up, "up")?;
            let down = evaluate_factor(state, &outeq, gamma_down, "down")?;

            // Of the directions that improve on the current log-likelihood, keep the best.
            let current = state.solution().log_likelihood();
            let best = [up, down]
                .into_iter()
                .flatten()
                .filter(|candidate| candidate.solution.log_likelihood() > current)
                .reduce(|best, candidate| {
                    if candidate.solution.log_likelihood() > best.solution.log_likelihood() {
                        candidate
                    } else {
                        best
                    }
                });

            if let Some(best) = best {
                let weights = state.solution().weights().clone();
                state.adopt_error_models(
                    best.error_models,
                    best.solution.with_stale_weights(weights),
                );
                *step *= self.config.growth;
            }
            *step *= self.config.shrink;
            if *step <= self.config.min_step {
                *step = self.config.step;
            }
        }
        Ok(())
    }
}

/// The error models with one factor set to a trial value, and the solution they give.
struct Candidate {
    error_models: AssayErrorModels,
    solution: Solution,
}

/// Re-solve the current support points with the factor of `outeq` set to `factor`.
///
/// Returns `None` (after logging a warning) if the IPM fails for this factor.
fn evaluate_factor<E: Equation>(
    state: &FitState<E>,
    outeq: &OutputLabel,
    factor: f64,
    direction: &str,
) -> Result<Option<Candidate>> {
    let mut error_models = state.error_models().clone();
    error_models.set_factor(outeq, factor)?;
    let theta = state.solution().theta();
    let psi = calculate_psi(state.equation(), state.data(), theta, &error_models, false)?;

    match Solution::optimal(theta.clone(), psi) {
        Ok(solution) => Ok(Some(Candidate {
            error_models,
            solution,
        })),
        Err(err) => {
            tracing::warn!(
                "Error in IPM during optim ({}) for outeq {}: {:?}",
                direction,
                outeq,
                err
            );
            Ok(None)
        }
    }
}
