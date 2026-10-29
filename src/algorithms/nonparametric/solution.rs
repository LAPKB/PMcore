//! A set of support points together with everything derived from them.

use anyhow::Result;

use crate::estimation::nonparametric::{ipm::burke, Psi, Theta, Weights};

/// Support points with their likelihoods, weights, and the log-likelihood those give.
///
/// The four parts always describe the same fit. A `Solution` is built either by solving for the
/// optimal weights ([`optimal`](Self::optimal)) or by computing the log-likelihood of given
/// weights ([`with_weights`](Self::with_weights)), and the fit only ever replaces it as a whole.
/// The one deliberate exception is [`with_stale_weights`](Self::with_stale_weights).
#[derive(Debug, Clone)]
pub(crate) struct Solution {
    theta: Theta,
    /// Likelihood of each subject (row) at each support point (column) of `theta`.
    psi: Psi,
    /// Probability of each support point in `theta`.
    weights: Weights,
    /// Log-likelihood of the data given `theta` and `weights`.
    log_likelihood: f64,
}

impl Solution {
    /// The starting point of a fit: the prior support points, not yet evaluated.
    pub(crate) fn unsolved(prior: Theta) -> Self {
        Self {
            theta: prior,
            psi: Psi::new(),
            weights: Weights::default(),
            log_likelihood: f64::NEG_INFINITY,
        }
    }

    /// Solve for the weights that maximize the likelihood, using Burke's interior point method.
    pub(crate) fn optimal(theta: Theta, psi: Psi) -> Result<Self> {
        let (weights, log_likelihood) = burke(&psi)?;
        Ok(Self {
            theta,
            psi,
            weights,
            log_likelihood,
        })
    }

    /// Use the given weights and compute the log-likelihood they give.
    pub(crate) fn with_weights(theta: Theta, psi: Psi, weights: Weights) -> Self {
        let log_likelihood = marginal_log_likelihood(&psi, &weights);
        Self {
            theta,
            psi,
            weights,
            log_likelihood,
        }
    }

    /// Replace the weights, keeping this solution's Ψ and log-likelihood.
    ///
    /// This breaks the invariant on purpose: the error-model optimization has always adopted a
    /// better Ψ while keeping the weights from before it. Whether the weights should follow is
    /// an open question (LAPKB/PMcore#334); dropping this call is all it takes to change that.
    pub(crate) fn with_stale_weights(self, weights: Weights) -> Self {
        Self { weights, ..self }
    }

    pub(crate) fn theta(&self) -> &Theta {
        &self.theta
    }

    pub(crate) fn psi(&self) -> &Psi {
        &self.psi
    }

    pub(crate) fn weights(&self) -> &Weights {
        &self.weights
    }

    pub(crate) fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    pub(crate) fn into_parts(self) -> (Theta, Psi, Weights, f64) {
        (self.theta, self.psi, self.weights, self.log_likelihood)
    }
}

/// Log-likelihood of the data under a discrete `(psi, weights)` model.
fn marginal_log_likelihood(psi: &Psi, weights: &Weights) -> f64 {
    let m = psi.matrix();
    (0..m.nrows())
        .map(|s| {
            let acc: f64 = (0..m.ncols()).map(|j| *m.get(s, j) * weights[j]).sum();
            acc.max(f64::MIN_POSITIVE).ln()
        })
        .sum::<f64>()
        + psi.log_scale()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn marginal_log_likelihood_restores_subject_scaling() -> anyhow::Result<()> {
        let log_likelihoods = Array2::from_shape_vec((2, 2), vec![-1000.0, -1001.0, -2.0, -4.0])?;
        let psi = Psi::from_log_likelihoods(log_likelihoods)?;
        let weights = Weights::from_vec(vec![0.6, 0.4]);

        let expected = -1000.0 + (0.6 + 0.4 * (-1.0_f64).exp()).ln() - 2.0
            + (0.6 + 0.4 * (-2.0_f64).exp()).ln();

        assert_relative_eq!(
            marginal_log_likelihood(&psi, &weights),
            expected,
            epsilon = 1e-12
        );
        Ok(())
    }

    #[test]
    fn optimal_weights_give_the_reported_log_likelihood() -> anyhow::Result<()> {
        let log_likelihoods = Array2::from_shape_vec((2, 2), vec![-1.0, -3.0, -4.0, -2.0])?;
        let psi = Psi::from_log_likelihoods(log_likelihoods)?;

        let solution = Solution::optimal(Theta::new(), psi)?;
        let recomputed = marginal_log_likelihood(solution.psi(), solution.weights());

        assert_relative_eq!(solution.log_likelihood(), recomputed, epsilon = 1e-8);
        Ok(())
    }
}
