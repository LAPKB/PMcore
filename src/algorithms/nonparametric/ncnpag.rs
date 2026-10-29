use anyhow::Result;
use faer::Mat;
use pharmsol::prelude::{
    data::{AssayErrorModels, Data},
    simulator::Equation,
};
use serde::{Deserialize, Serialize};

use super::fit::{Fit, FitState, Runner};
use super::npag::{Npag, NpagConfig};
use crate::algorithms::{Status, StopReason};
use crate::estimation::nonparametric::{calculate_psi, Psi, Theta, Weights};

/// Configuration options for the non-collapsing NPAG (NCNPAG) algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NcnpagConfig {
    /// Number of NPAG cycles used to refine each surviving support point.
    ///
    /// `0` disables refinement, leaving a pure Bayesian reweighting of the input
    /// grid. The default (`500`) matches the historical NPAGFULL behavior.
    pub cycles: usize,
    /// Whether to show NPAG progress output during refinement.
    pub progress: bool,
}

impl Default for NcnpagConfig {
    fn default() -> Self {
        Self {
            cycles: 500,
            progress: false,
        }
    }
}

impl NcnpagConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of NPAG refinement cycles per support point (`0` to skip).
    pub fn cycles(mut self, cycles: usize) -> Self {
        self.cycles = cycles;
        self
    }

    /// Enable or disable NPAG progress output during refinement.
    pub fn progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }
}

/// Non-collapsing NPAG (NCNPAG) algorithm.
///
/// Individualizes a set of prior support points to a subject's data in two
/// steps, without collapsing (merging) the points:
///
/// 1. **Bayesian filtering.** Evaluate the likelihood `P(data | θⱼ)` for each
///    prior support point, apply a flat (uniform) prior so the posterior weight
///    is proportional to the likelihood (`postⱼ ∝ ∏ᵢ P(dataᵢ | θⱼ)`), drop
///    points whose normalized weight falls below `1e-100 × max`, and renormalize
///    the survivors.
/// 2. **Per-point NPAG refinement.** When `cycles > 0`, seed a full NPAG run
///    from each surviving point and replace it with the resulting daughter
///    point, preserving its filter weight. Points whose refinement fails or
///    yields nothing are kept at their original location.
///
/// The result is returned as a standard [`NonParametricResult`], so the
/// `(theta, weights)` can be consumed exactly like any other fit.
#[derive(Debug)]
pub(crate) struct Ncnpag {
    config: NcnpagConfig,
}

impl Ncnpag {
    pub(crate) fn new(config: NcnpagConfig) -> Self {
        Self { config }
    }
}

/// Refine each support point with a full NPAG seeded from that single point,
/// preserving the point's filter weight (NPAGFULL). Points whose refinement
/// fails or produces no output are kept at their original location.
fn refine_points<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &AssayErrorModels,
    theta: &Theta,
    weights: &Weights,
    cycles: usize,
    progress: bool,
) -> Result<(Theta, Weights)> {
    let parameter_space = theta.parameters().clone();
    let n_points = theta.matrix().nrows();
    let mut refined_points: Vec<Vec<f64>> = Vec::with_capacity(n_points);
    let mut kept_weights: Vec<f64> = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let point: Vec<f64> = theta.matrix().row(i).iter().copied().collect();
        let single = Mat::from_fn(1, point.len(), |_r, c| point[c]);
        let single_theta = Theta::from_parts(single, parameter_space.clone())?;

        let npag_config = NpagConfig {
            max_cycles: cycles,
            progress,
            ..Default::default()
        };
        let npag = Npag::new(npag_config, &single_theta, error_models);
        let state = FitState::from_parts(
            equation.clone(),
            data.clone(),
            error_models.clone(),
            single_theta,
        );
        let run = Fit::new(state, Box::new(npag)).and_then(|mut fit| {
            fit.run()?;
            Ok(fit)
        });

        match run {
            Ok(fit) if fit.state().theta.nspp() > 0 => {
                let refined: Vec<f64> = fit.state().theta.matrix().row(0).iter().copied().collect();
                refined_points.push(refined);
            }
            Ok(_) => {
                tracing::warn!(
                    "NCNPAG: refinement produced no points for support point {} — keeping original",
                    i + 1
                );
                refined_points.push(point);
            }
            Err(e) => {
                tracing::warn!(
                    "NCNPAG: refinement failed for support point {}: {} — keeping original",
                    i + 1,
                    e
                );
                refined_points.push(point);
            }
        }
        kept_weights.push(weights[i]);
    }

    let n_params = parameter_space.len();
    let matrix = Mat::from_fn(refined_points.len(), n_params, |r, c| refined_points[r][c]);
    let refined_theta = Theta::from_parts(matrix, parameter_space)?;

    let weight_sum: f64 = kept_weights.iter().sum();
    let refined_weights = if weight_sum > 0.0 {
        Weights::from_vec(kept_weights.iter().map(|w| w / weight_sum).collect())
    } else {
        Weights::uniform(refined_points.len())
    };

    Ok((refined_theta, refined_weights))
}

/// Marginal log-likelihood of the data under a discrete `(psi, weights)` model.
fn marginal_loglik(psi: &Psi, w: &Weights) -> f64 {
    let m = psi.matrix();
    (0..m.nrows())
        .map(|s| {
            let acc: f64 = (0..m.ncols()).map(|j| *m.get(s, j) * w[j]).sum();
            acc.max(f64::MIN_POSITIVE).ln()
        })
        .sum::<f64>()
        + psi.log_scale()
}

impl<E: Equation> Runner<E> for Ncnpag {
    fn step(&mut self, state: &mut FitState<E>) -> Result<Status> {
        // Likelihood of each fixed support point for the data.
        let psi = calculate_psi(
            &state.equation,
            &state.data,
            &state.theta,
            &state.error_models,
            false,
        )?;

        // Flat (uniform) prior: postⱼ ∝ ∏ᵢ P(dataᵢ | θⱼ). Accumulate in log space.
        let n_points = state.theta.matrix().nrows();
        let mut log_weights = vec![f64::NEG_INFINITY; n_points];
        for (j, slot) in log_weights.iter_mut().enumerate() {
            let mut log_weight = 0.0; // ln(uniform prior) is constant, drops out on normalization
            let mut is_zero = false;
            for s in 0..psi.matrix().nrows() {
                let likelihood = psi.matrix()[(s, j)];
                if likelihood <= 0.0 {
                    is_zero = true;
                    break;
                }
                log_weight += likelihood.ln();
            }
            if !is_zero {
                *slot = log_weight;
            }
        }

        let max_log_weight = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_log_weight.is_finite() {
            anyhow::bail!("NCNPAG: every support point has zero joint likelihood for the data");
        }

        let mut weights: Vec<f64> = log_weights
            .iter()
            .map(|&lw| {
                if lw.is_finite() {
                    (lw - max_log_weight).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            anyhow::bail!("NCNPAG: filtering produced non-positive posterior mass");
        }
        for w in &mut weights {
            *w /= total;
        }

        // Non-collapsing filter: keep points within 1e-100 of the maximum weight.
        let max_weight = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let threshold = 1e-100;
        let keep: Vec<usize> = weights
            .iter()
            .enumerate()
            .filter(|(_, w)| **w > threshold * max_weight)
            .map(|(i, _)| i)
            .collect();

        // Filter theta and renormalize the surviving weights (NPAGFULL11).
        state.theta.filter_indices(&keep);
        let kept: Vec<f64> = keep.iter().map(|&i| weights[i]).collect();
        let sum: f64 = kept.iter().sum();
        state.weights = Weights::from_vec(kept.iter().map(|w| w / sum).collect());

        // NPAGFULL: refine each surviving point with a full NPAG seeded from it.
        if self.config.cycles > 0 {
            let (refined_theta, refined_weights) = refine_points(
                &state.equation,
                &state.data,
                &state.error_models,
                &state.theta,
                &state.weights,
                self.config.cycles,
                self.config.progress,
            )?;
            state.theta = refined_theta;
            state.weights = refined_weights;
        }

        // Recompute psi over the final support points so psi/weights stay aligned.
        state.psi = calculate_psi(
            &state.equation,
            &state.data,
            &state.theta,
            &state.error_models,
            false,
        )?;

        state.log_likelihood = marginal_loglik(&state.psi, &state.weights);
        Ok(Status::Stop(StopReason::Converged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn marginal_loglik_restores_subject_scaling() -> anyhow::Result<()> {
        let log_likelihoods = Array2::from_shape_vec((2, 2), vec![-1000.0, -1001.0, -2.0, -4.0])?;
        let psi = Psi::from_log_likelihoods(log_likelihoods)?;
        let weights = Weights::from_vec(vec![0.6, 0.4]);

        let expected = -1000.0 + (0.6 + 0.4 * (-1.0_f64).exp()).ln() - 2.0
            + (0.6 + 0.4 * (-2.0_f64).exp()).ln();

        assert_relative_eq!(marginal_loglik(&psi, &weights), expected, epsilon = 1e-12);
        Ok(())
    }
}
