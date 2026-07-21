use anyhow::Result;
use ndarray::Array2;

use super::covariance::{cholesky_log_determinant, cholesky_lower, solve_lower};

const LOG_2PI: f64 = 1.8378770664093453_f64;

/// Subject-level proposal score used by SAEM MCMC kernels and future FOCE diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SubjectPosteriorScore {
    pub(crate) log_likelihood: f64,
    pub(crate) eta_log_prior: f64,
    pub(crate) kappa_log_prior: f64,
}

impl SubjectPosteriorScore {
    pub(crate) fn log_posterior(self) -> f64 {
        self.log_likelihood + self.eta_log_prior + self.kappa_log_prior
    }

    pub(crate) fn log_acceptance_ratio(self, proposed: Self) -> f64 {
        proposed.log_posterior() - self.log_posterior()
    }
}

/// Log-density of η under a multivariate normal N(0, Ω).
///
/// The normalizing constant is retained for diagnostics and objective assembly.
/// MCMC acceptance ratios cancel it when Ω is unchanged, matching established
/// kernel logic while keeping PMcore's score explicit.
pub(crate) fn eta_log_prior(eta: &[f64], cholesky: &[Vec<f64>], log_det: f64) -> Result<f64> {
    let z = solve_lower(cholesky, eta)?;
    let quadratic = z.iter().map(|value| value * value).sum::<f64>();
    Ok(-0.5 * (eta.len() as f64 * LOG_2PI + log_det + quadratic))
}

pub(crate) fn eta_log_prior_from_omega(eta: &[f64], omega: &Array2<f64>) -> Result<f64> {
    let cholesky = cholesky_lower(omega)?;
    let log_det = cholesky_log_determinant(&cholesky);
    eta_log_prior(eta, &cholesky, log_det)
}

pub(crate) fn eta_log_priors(
    etas: &[Vec<Vec<f64>>],
    omega: &Array2<f64>,
    chain_index: usize,
) -> Result<Vec<f64>> {
    let cholesky = cholesky_lower(omega)?;
    let log_det = cholesky_log_determinant(&cholesky);

    etas.iter()
        .map(|subject_chains| eta_log_prior(&subject_chains[chain_index], &cholesky, log_det))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parametric::covariance::{cholesky_lower, identity_matrix};

    #[test]
    fn eta_log_prior_uses_full_normal_density() {
        let omega = identity_matrix(2);
        let eta = vec![1.0, 2.0];
        let cholesky = cholesky_lower(&omega).unwrap();
        let actual = eta_log_prior(&eta, &cholesky, 0.0).unwrap();
        let expected = -0.5 * (2.0 * LOG_2PI + 5.0);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn eta_log_prior_handles_correlated_omega() {
        let omega = ndarray::array![[4.0, 1.0], [1.0, 2.0]];
        let eta = vec![1.0, -1.0];
        let cholesky = cholesky_lower(&omega).unwrap();
        let actual = eta_log_prior(&eta, &cholesky, 7.0_f64.ln()).unwrap();
        let expected = -0.5 * (2.0 * LOG_2PI + 7.0_f64.ln() + 8.0 / 7.0);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn posterior_score_produces_acceptance_ratio() {
        let current = SubjectPosteriorScore {
            log_likelihood: -10.0,
            eta_log_prior: -1.0,
            kappa_log_prior: 0.0,
        };
        let proposed = SubjectPosteriorScore {
            log_likelihood: -9.0,
            eta_log_prior: -1.5,
            kappa_log_prior: 0.0,
        };
        assert_eq!(current.log_acceptance_ratio(proposed), 0.5);
    }
}
