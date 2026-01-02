//! Parameter transformations for constrained optimization
//!
//! This module implements parameter transformations that map between:
//! - **ψ (psi)**: The natural/constrained parameter space (e.g., clearance must be positive)
//! - **φ (phi)**: The unconstrained/transformed space used internally by MCMC
//!
//! # Mathematical Background
//!
//! In SAEM and other parametric algorithms, random effects η are assumed to follow
//! a multivariate normal distribution: η ~ N(0, Ω). However, the actual parameters
//! (like clearance, volume) often have natural constraints (e.g., must be positive).
//!
//! The transformation φ = h(ψ) maps from the constrained space to an unconstrained space
//! where normality assumptions are more appropriate.
//!
//! # Supported Transformations
//!
//! | Transform | ψ Domain | φ = h(ψ) | ψ = h⁻¹(φ) |
//! |-----------|----------|----------|------------|
//! | None | (-∞, ∞) | φ = ψ | ψ = φ |
//! | LogNormal | (0, ∞) | φ = ln(ψ) | ψ = exp(φ) |
//! | Logit | (a, b) | φ = ln((ψ-a)/(b-ψ)) | ψ = a + (b-a)/(1+exp(-φ)) |
//! | Probit | (a, b) | φ = Φ⁻¹((ψ-a)/(b-a)) | ψ = a + (b-a)Φ(φ) |
//!
//! # R saemix Correspondence
//!
//! These transformations correspond to R saemix's `transform.par` vector:
//! - 0 → `ParameterTransform::None`
//! - 1 → `ParameterTransform::LogNormal`
//! - 2 → `ParameterTransform::Probit(lower, upper)`
//! - 3 → `ParameterTransform::Logit(lower, upper)`

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Parameter transformation type for constrained optimization
///
/// Defines how parameters are transformed between the natural (constrained)
/// space ψ and the unconstrained space φ used internally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterTransform {
    /// No transformation: φ = ψ
    /// Use when parameter is truly unconstrained (can be any real number)
    None,

    /// Log-normal transformation: φ = ln(ψ)
    /// Use for strictly positive parameters (clearance, volume, rate constants)
    /// ψ ∈ (0, ∞), φ ∈ (-∞, ∞)
    LogNormal,

    /// Logit transformation: φ = ln((ψ-a)/(b-ψ))
    /// Use for parameters bounded between a and b (e.g., bioavailability ∈ (0,1))
    /// ψ ∈ (a, b), φ ∈ (-∞, ∞)
    Logit {
        /// Lower bound (a)
        lower: f64,
        /// Upper bound (b)
        upper: f64,
    },

    /// Probit transformation: φ = Φ⁻¹((ψ-a)/(b-a))
    /// Alternative to logit for bounded parameters, uses normal CDF
    /// ψ ∈ (a, b), φ ∈ (-∞, ∞)
    Probit {
        /// Lower bound (a)
        lower: f64,
        /// Upper bound (b)
        upper: f64,
    },
}

impl Default for ParameterTransform {
    fn default() -> Self {
        ParameterTransform::None
    }
}

impl ParameterTransform {
    /// Create a logit transform for parameters bounded between 0 and 1
    pub fn logit_unit() -> Self {
        ParameterTransform::Logit {
            lower: 0.0,
            upper: 1.0,
        }
    }

    /// Create a logit transform with custom bounds
    pub fn logit(lower: f64, upper: f64) -> Self {
        ParameterTransform::Logit { lower, upper }
    }

    /// Create a probit transform with custom bounds
    pub fn probit(lower: f64, upper: f64) -> Self {
        ParameterTransform::Probit { lower, upper }
    }

    /// Transform from constrained (ψ) to unconstrained (φ) space
    ///
    /// This is h(ψ) in mathematical notation.
    ///
    /// # Arguments
    /// * `psi` - Parameter value in the constrained space
    ///
    /// # Returns
    /// * Parameter value in the unconstrained space
    ///
    /// # Panics
    /// * If `psi` is outside the valid domain for the transformation
    pub fn psi_to_phi(&self, psi: f64) -> f64 {
        match self {
            ParameterTransform::None => psi,

            ParameterTransform::LogNormal => {
                debug_assert!(
                    psi > 0.0,
                    "LogNormal transform requires psi > 0, got {}",
                    psi
                );
                psi.ln()
            }

            ParameterTransform::Logit { lower, upper } => {
                debug_assert!(
                    psi > *lower && psi < *upper,
                    "Logit transform requires {} < psi < {}, got {}",
                    lower,
                    upper,
                    psi
                );
                let normalized = (psi - lower) / (upper - psi);
                normalized.ln()
            }

            ParameterTransform::Probit { lower, upper } => {
                debug_assert!(
                    psi > *lower && psi < *upper,
                    "Probit transform requires {} < psi < {}, got {}",
                    lower,
                    upper,
                    psi
                );
                let normalized = (psi - lower) / (upper - lower);
                probit(normalized)
            }
        }
    }

    /// Transform from unconstrained (φ) to constrained (ψ) space
    ///
    /// This is h⁻¹(φ) in mathematical notation.
    ///
    /// # Arguments
    /// * `phi` - Parameter value in the unconstrained space
    ///
    /// # Returns
    /// * Parameter value in the constrained space
    pub fn phi_to_psi(&self, phi: f64) -> f64 {
        match self {
            ParameterTransform::None => phi,

            ParameterTransform::LogNormal => phi.exp(),

            ParameterTransform::Logit { lower, upper } => {
                let exp_phi = phi.exp();
                lower + (upper - lower) * exp_phi / (1.0 + exp_phi)
            }

            ParameterTransform::Probit { lower, upper } => {
                let normalized = normal_cdf(phi);
                lower + (upper - lower) * normalized
            }
        }
    }

    /// Compute the derivative dψ/dφ (Jacobian for likelihood adjustment)
    ///
    /// When transforming parameters, the likelihood must be adjusted by the
    /// Jacobian of the transformation. This returns |dψ/dφ| for use in
    /// likelihood calculations.
    ///
    /// # Arguments
    /// * `phi` - Parameter value in the unconstrained space
    ///
    /// # Returns
    /// * The absolute value of dψ/dφ at the given φ
    pub fn dpsi_dphi(&self, phi: f64) -> f64 {
        match self {
            ParameterTransform::None => 1.0,

            ParameterTransform::LogNormal => phi.exp(),

            ParameterTransform::Logit { lower, upper } => {
                let exp_phi = phi.exp();
                let denom = (1.0 + exp_phi).powi(2);
                (upper - lower) * exp_phi / denom
            }

            ParameterTransform::Probit { lower, upper } => {
                // dψ/dφ = (b-a) * φ'(Φ⁻¹) = (b-a) * pdf(φ)
                (upper - lower) * normal_pdf(phi)
            }
        }
    }

    /// Compute the log of the Jacobian: ln|dψ/dφ|
    ///
    /// More numerically stable than computing dpsi_dphi and taking the log.
    /// Used directly in log-likelihood calculations.
    pub fn log_jacobian(&self, phi: f64) -> f64 {
        match self {
            ParameterTransform::None => 0.0,

            ParameterTransform::LogNormal => phi,

            ParameterTransform::Logit { lower, upper } => {
                // ln(dψ/dφ) = ln(b-a) + φ - 2*ln(1+exp(φ))
                (upper - lower).ln() + phi - 2.0 * (1.0 + phi.exp()).ln()
            }

            ParameterTransform::Probit { lower, upper } => {
                // ln(dψ/dφ) = ln(b-a) + ln(pdf(φ))
                (upper - lower).ln() + log_normal_pdf(phi)
            }
        }
    }

    /// Check if a ψ value is within the valid domain
    pub fn is_valid_psi(&self, psi: f64) -> bool {
        match self {
            ParameterTransform::None => true,
            ParameterTransform::LogNormal => psi > 0.0,
            ParameterTransform::Logit { lower, upper } => psi > *lower && psi < *upper,
            ParameterTransform::Probit { lower, upper } => psi > *lower && psi < *upper,
        }
    }

    /// Get the domain bounds for ψ (if bounded)
    pub fn psi_bounds(&self) -> Option<(f64, f64)> {
        match self {
            ParameterTransform::None => None,
            ParameterTransform::LogNormal => Some((0.0, f64::INFINITY)),
            ParameterTransform::Logit { lower, upper } => Some((*lower, *upper)),
            ParameterTransform::Probit { lower, upper } => Some((*lower, *upper)),
        }
    }

    /// Get the R saemix transform code (0-3)
    pub fn to_saemix_code(&self) -> u8 {
        match self {
            ParameterTransform::None => 0,
            ParameterTransform::LogNormal => 1,
            ParameterTransform::Probit { .. } => 2,
            ParameterTransform::Logit { .. } => 3,
        }
    }

    /// Create from R saemix transform code
    ///
    /// For Logit/Probit, bounds default to (0, 1). Use the dedicated constructors
    /// for custom bounds.
    pub fn from_saemix_code(code: u8) -> Self {
        match code {
            0 => ParameterTransform::None,
            1 => ParameterTransform::LogNormal,
            2 => ParameterTransform::Probit {
                lower: 0.0,
                upper: 1.0,
            },
            3 => ParameterTransform::Logit {
                lower: 0.0,
                upper: 1.0,
            },
            _ => ParameterTransform::None,
        }
    }
}

// ============================================================================
// Helper functions for normal distribution
// ============================================================================

/// Standard normal probability density function
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Log of standard normal PDF (more stable)
fn log_normal_pdf(x: f64) -> f64 {
    -0.5 * (x * x + (2.0 * PI).ln())
}

/// Standard normal cumulative distribution function (Φ)
///
/// Uses the approximation from Abramowitz and Stegun
fn normal_cdf(x: f64) -> f64 {
    // Use the complementary error function relationship
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse standard normal CDF (probit function, Φ⁻¹)
///
/// Uses Acklam's algorithm which is accurate to about 1.15e-9
fn probit(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0, "probit requires 0 < p < 1, got {}", p);

    // Coefficients for Acklam's algorithm
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Approximation from Abramowitz and Stegun
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_none_transform() {
        let t = ParameterTransform::None;
        assert!((t.psi_to_phi(5.0) - 5.0).abs() < EPSILON);
        assert!((t.phi_to_psi(5.0) - 5.0).abs() < EPSILON);
        assert!((t.dpsi_dphi(5.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_lognormal_transform() {
        let t = ParameterTransform::LogNormal;

        // psi = 1 -> phi = 0
        assert!((t.psi_to_phi(1.0) - 0.0).abs() < EPSILON);
        assert!((t.phi_to_psi(0.0) - 1.0).abs() < EPSILON);

        // psi = e -> phi = 1
        assert!((t.psi_to_phi(std::f64::consts::E) - 1.0).abs() < EPSILON);
        assert!((t.phi_to_psi(1.0) - std::f64::consts::E).abs() < EPSILON);

        // Round-trip
        let psi = 2.5;
        let phi = t.psi_to_phi(psi);
        assert!((t.phi_to_psi(phi) - psi).abs() < EPSILON);
    }

    #[test]
    fn test_logit_transform() {
        let t = ParameterTransform::Logit {
            lower: 0.0,
            upper: 1.0,
        };

        // psi = 0.5 -> phi = 0
        assert!((t.psi_to_phi(0.5) - 0.0).abs() < EPSILON);
        assert!((t.phi_to_psi(0.0) - 0.5).abs() < EPSILON);

        // Round-trip
        let psi = 0.7;
        let phi = t.psi_to_phi(psi);
        assert!((t.phi_to_psi(phi) - psi).abs() < EPSILON);
    }

    #[test]
    fn test_logit_custom_bounds() {
        let t = ParameterTransform::Logit {
            lower: 10.0,
            upper: 100.0,
        };

        // Midpoint maps to 0
        assert!((t.psi_to_phi(55.0) - 0.0).abs() < EPSILON);
        assert!((t.phi_to_psi(0.0) - 55.0).abs() < EPSILON);

        // Round-trip
        let psi = 30.0;
        let phi = t.psi_to_phi(psi);
        assert!((t.phi_to_psi(phi) - psi).abs() < EPSILON);
    }

    #[test]
    fn test_probit_transform() {
        let t = ParameterTransform::Probit {
            lower: 0.0,
            upper: 1.0,
        };

        // psi = 0.5 -> phi ≈ 0
        assert!(t.psi_to_phi(0.5).abs() < 1e-6);

        // Test probit directly: probit(0.7) should be approximately 0.524
        let probit_07 = probit(0.7);
        assert!(
            (probit_07 - 0.524).abs() < 0.01,
            "probit(0.7) should be ~0.524, got {}",
            probit_07
        );

        // Round-trip
        let psi = 0.7;
        let phi = t.psi_to_phi(psi);
        let psi_back = t.phi_to_psi(phi);
        assert!(
            (psi_back - psi).abs() < 1e-4,
            "Round-trip failed: psi={}, phi={}, psi_back={}",
            psi,
            phi,
            psi_back
        );
    }

    #[test]
    fn test_jacobian_lognormal() {
        let t = ParameterTransform::LogNormal;
        let phi: f64 = 1.0;

        // For lognormal: dψ/dφ = exp(φ)
        let expected = phi.exp();
        assert!((t.dpsi_dphi(phi) - expected).abs() < EPSILON);

        // log jacobian should equal phi
        assert!((t.log_jacobian(phi) - phi).abs() < EPSILON);
    }

    #[test]
    fn test_saemix_codes() {
        assert_eq!(ParameterTransform::None.to_saemix_code(), 0);
        assert_eq!(ParameterTransform::LogNormal.to_saemix_code(), 1);
        assert_eq!(
            ParameterTransform::Probit {
                lower: 0.0,
                upper: 1.0
            }
            .to_saemix_code(),
            2
        );
        assert_eq!(
            ParameterTransform::Logit {
                lower: 0.0,
                upper: 1.0
            }
            .to_saemix_code(),
            3
        );
    }

    #[test]
    fn test_validity_checks() {
        let t = ParameterTransform::LogNormal;
        assert!(!t.is_valid_psi(-1.0));
        assert!(!t.is_valid_psi(0.0));
        assert!(t.is_valid_psi(0.1));
        assert!(t.is_valid_psi(100.0));

        let t = ParameterTransform::Logit {
            lower: 0.0,
            upper: 1.0,
        };
        assert!(!t.is_valid_psi(-0.1));
        assert!(!t.is_valid_psi(0.0));
        assert!(t.is_valid_psi(0.5));
        assert!(!t.is_valid_psi(1.0));
        assert!(!t.is_valid_psi(1.1));
    }
}
