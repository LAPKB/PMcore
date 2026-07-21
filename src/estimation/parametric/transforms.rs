use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::model::ParameterScale;

/// Transform model-space ψ to estimation-space φ.
///
/// SAEM/FOCE-style η values are additive in φ-space. Model execution remains in
/// ψ-space, so every algorithm should use these shared helpers for consistency.
pub(crate) fn psi_to_phi(psi: f64, scale: ParameterScale) -> f64 {
    match scale {
        ParameterScale::Identity => psi,
        ParameterScale::Log => psi.ln(),
        ParameterScale::Logit { lower, upper } => ((psi - lower) / (upper - psi)).ln(),
        ParameterScale::Probit { lower, upper } => {
            standard_normal().inverse_cdf((psi - lower) / (upper - lower))
        }
    }
}

/// Transform estimation-space φ back to model-space ψ.
pub(crate) fn phi_to_psi(phi: f64, scale: ParameterScale) -> f64 {
    match scale {
        ParameterScale::Identity => phi,
        ParameterScale::Log => phi.exp(),
        ParameterScale::Logit { lower, upper } => {
            let exp_phi = phi.exp();
            lower + (upper - lower) * exp_phi / (1.0 + exp_phi)
        }
        ParameterScale::Probit { lower, upper } => {
            lower + (upper - lower) * standard_normal().cdf(phi)
        }
    }
}

fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal parameters are valid")
}

/// Exact derivative dψ/dφ of the inverse transform φ → ψ.
///
/// Used for the delta-method transformation of free-coordinate standard errors
/// from estimation (φ) space to natural (ψ) space. The absolute value is the
/// Jacobian scaling factor: sd_ψ = |dψ/dφ| × sd_φ.
pub(crate) fn phi_to_psi_derivative(phi: f64, scale: ParameterScale) -> f64 {
    match scale {
        ParameterScale::Identity => 1.0,
        ParameterScale::Log => phi.exp(),
        ParameterScale::Logit { lower, upper } => {
            let exp_negative_abs_phi = (-phi.abs()).exp();
            (upper - lower) * exp_negative_abs_phi / (1.0 + exp_negative_abs_phi).powi(2)
        }
        ParameterScale::Probit { lower, upper } => {
            let norm = standard_normal();
            (upper - lower) * norm.pdf(phi)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_transforms_round_trip() {
        let scales = [
            ParameterScale::Identity,
            ParameterScale::Log,
            ParameterScale::Logit {
                lower: 0.0,
                upper: 1.0,
            },
            ParameterScale::Probit {
                lower: -2.0,
                upper: 3.0,
            },
        ];
        let values = [0.5, 2.0, 0.75, 1.25];

        for (value, scale) in values.into_iter().zip(scales.into_iter()) {
            let phi = psi_to_phi(value, scale);
            let psi = phi_to_psi(phi, scale);
            assert!((psi - value).abs() < 1e-10, "scale={scale:?}");
        }
    }

    #[test]
    fn population_uncertainty_identity_derivative_is_one() {
        assert_eq!(phi_to_psi_derivative(0.0, ParameterScale::Identity), 1.0);
        assert_eq!(phi_to_psi_derivative(5.0, ParameterScale::Identity), 1.0);
    }

    #[test]
    fn population_uncertainty_log_derivative_is_exp_phi() {
        for phi in [-2.0_f64, 0.0, 1.5] {
            let expected = phi.exp();
            let actual = phi_to_psi_derivative(phi, ParameterScale::Log);
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn population_uncertainty_logit_derivative_is_stable_logistic_probability() {
        let scale = ParameterScale::Logit {
            lower: 0.0,
            upper: 1.0,
        };
        // At phi = 0, p = 0.5, derivative = 1 * 0.5 * 0.5 = 0.25
        assert!((phi_to_psi_derivative(0.0, scale) - 0.25).abs() < 1e-12);
        // At large positive phi, p ≈ 1, derivative ≈ 0
        let large = 50.0;
        let d_large = phi_to_psi_derivative(large, scale);
        assert!(d_large > 0.0);
        assert!(d_large < 1e-10);
        assert!(d_large.is_finite());
    }

    #[test]
    fn population_uncertainty_probit_derivative_is_standard_normal_pdf_scaled() {
        let scale = ParameterScale::Probit {
            lower: 10.0,
            upper: 20.0,
        };
        let norm = Normal::new(0.0, 1.0).expect("valid standard normal");
        for phi in [-2.0, 0.0, 1.5] {
            let expected = (20.0 - 10.0) * norm.pdf(phi);
            let actual = phi_to_psi_derivative(phi, scale);
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
