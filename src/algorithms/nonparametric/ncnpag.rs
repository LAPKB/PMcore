use crate::{
    algorithms::{
        nonparametric::{npag::NPAG, NpagConfig},
        NonParametricRunner, Status, StopReason,
    },
    estimation::nonparametric::{
        calculate_psi, CycleLog, NPCycle, NonParametricResult, Psi, Theta, Weights,
    },
};

use anyhow::Result;
use faer::Mat;
use pharmsol::prelude::{
    data::{AssayErrorModels, Data},
    simulator::Equation,
};

use serde::{Deserialize, Serialize};

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
pub struct NCNPAG<E: Equation + Send + 'static> {
    equation: E,
    psi: Psi,
    theta: Theta,
    w: Weights,
    objf: f64,
    cycle: usize,
    status: Status,
    data: Data,
    cyclelog: CycleLog,
    error_models: AssayErrorModels,
    prior: Theta,
    cycles: usize,
    progress: bool,
}

impl<E: Equation + Send + 'static> NCNPAG<E> {
    pub(crate) fn from_parts(
        equation: E,
        data: Data,
        error_models: AssayErrorModels,
        theta: Theta,
        config: NcnpagConfig,
    ) -> Result<Self> {
        Ok(Self {
            equation,
            psi: Psi::new(),
            theta: theta.clone(),
            w: Weights::default(),
            objf: f64::INFINITY,
            cycle: 0,
            status: Status::Continue,
            data,
            cyclelog: CycleLog::new(),
            error_models,
            prior: theta,
            cycles: config.cycles,
            progress: config.progress,
        })
    }
}

/// Refine each support point with a full NPAG seeded from that single point,
/// preserving the point's filter weight (NPAGFULL). Points whose refinement
/// fails or produces no output are kept at their original location.
fn refine_points<E: Equation + Send + 'static>(
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
        let mut npag = NPAG::from_parts(
            equation.clone(),
            data.clone(),
            error_models.clone(),
            single_theta,
            npag_config,
        )?;

        #[allow(clippy::while_let_loop)]
        let run = npag.initialize().and_then(|_| {
            loop {
                match npag.next_cycle()? {
                    Status::Continue => continue,
                    Status::Stop(_) => break,
                }
            }
            Ok(())
        });

        match run {
            Ok(()) if npag.theta().matrix().nrows() > 0 => {
                let refined: Vec<f64> = npag.theta().matrix().row(0).iter().copied().collect();
                refined_points.push(refined);
            }
            Ok(()) => {
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

impl<E: Equation + Send + 'static> NonParametricRunner<E> for NCNPAG<E> {
    fn into_result(&self) -> Result<NonParametricResult<E>> {
        NonParametricResult::new(
            self.equation.clone(),
            self.data.clone(),
            self.error_models.clone(),
            self.prior.clone(),
            self.theta.clone(),
            self.psi.clone(),
            self.w.clone(),
            self.objf,
            self.cycle,
            self.status.clone(),
            self.cyclelog.clone(),
        )
    }

    fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }

    fn equation(&self) -> &E {
        &self.equation
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn likelihood(&self) -> f64 {
        self.objf
    }

    fn increment_cycle(&mut self) -> usize {
        0
    }

    fn cycle(&self) -> usize {
        0
    }

    fn set_theta(&mut self, theta: Theta) {
        self.theta = theta;
    }

    fn theta(&self) -> &Theta {
        &self.theta
    }

    fn psi(&self) -> &Psi {
        &self.psi
    }

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn evaluation(&mut self) -> Result<Status> {
        self.status = Status::Stop(StopReason::Converged);
        Ok(self.status.clone())
    }

    fn estimation(&mut self) -> Result<()> {
        // Likelihood of each fixed support point for the data.
        let psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &self.error_models,
            false,
        )?;

        // Flat (uniform) prior: postⱼ ∝ ∏ᵢ P(dataᵢ | θⱼ). Accumulate in log space.
        let n_points = self.theta.matrix().nrows();
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
        self.theta.filter_indices(&keep);
        let kept: Vec<f64> = keep.iter().map(|&i| weights[i]).collect();
        let sum: f64 = kept.iter().sum();
        self.w = Weights::from_vec(kept.iter().map(|w| w / sum).collect());

        // NPAGFULL: refine each surviving point with a full NPAG seeded from it.
        if self.cycles > 0 {
            let (refined_theta, refined_weights) = refine_points(
                &self.equation,
                &self.data,
                &self.error_models,
                &self.theta,
                &self.w,
                self.cycles,
                self.progress,
            )?;
            self.theta = refined_theta;
            self.w = refined_weights;
        }

        // Recompute psi over the final support points so psi/weights stay aligned.
        self.psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &self.error_models,
            false,
        )?;

        self.objf = marginal_loglik(&self.psi, &self.w);
        Ok(())
    }

    fn condensation(&mut self) -> Result<()> {
        Ok(())
    }

    fn optimizations(&mut self) -> Result<()> {
        Ok(())
    }

    fn expansion(&mut self) -> Result<()> {
        Ok(())
    }

    fn log_cycle_state(&mut self) {
        let state = NPCycle::new(
            self.cycle,
            self.objf,
            self.error_models.clone(),
            self.theta.clone(),
            self.w.clone(),
            self.theta.nspp(),
            0.0,
            self.status.clone(),
        );
        self.cyclelog.push(state);
    }

    /// NCNPAG is a single-pass reweighting: it evaluates the likelihood of the
    /// fixed prior support points once, rather than iterating cycles.
    fn fit(&mut self) -> Result<NonParametricResult<E>> {
        self.estimation()?;
        self.evaluation()?;
        self.log_cycle_state();

        self.into_result()
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
