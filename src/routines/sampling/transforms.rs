//! Parameter transformation utilities for SAEM
//!
//! This module provides utilities for applying parameter transformations
//! in the context of the SAEM algorithm.
//!
//! # Transformation Direction
//!
//! In SAEM, we work internally in the unconstrained φ-space where random effects
//! are normally distributed: η ~ N(0, Ω). When we need to compute likelihoods
//! or report results, we transform to the constrained ψ-space.
//!
//! | Space | Symbol | Distribution | Example |
//! |-------|--------|--------------|---------|
//! | Unconstrained | φ | Normal | φ = log(CL) ∈ (-∞, ∞) |
//! | Constrained | ψ | Log-normal, etc. | ψ = CL ∈ (0, ∞) |
//!
//! # R saemix Correspondence
//!
//! These functions implement the `transphi` and `transpsi` functions from
//! R saemix's `func_aux.R`:
//!
//! - `transform_phi_to_psi` ≈ `transpsi()`: φ → ψ
//! - `transform_psi_to_phi` ≈ `transphi()`: ψ → φ

use faer::Col;

use crate::structs::parametric::ParameterTransform;

/// A collection of parameter transformations for a model
///
/// This struct holds one transformation per parameter and provides
/// vectorized transformation methods.
#[derive(Debug, Clone)]
pub struct ParameterTransforms {
    /// One transformation per parameter
    transforms: Vec<ParameterTransform>,
}

impl ParameterTransforms {
    /// Create a new parameter transforms collection
    ///
    /// # Arguments
    /// * `transforms` - Vector of transforms, one per parameter
    pub fn new(transforms: Vec<ParameterTransform>) -> Self {
        Self { transforms }
    }

    /// Create transforms with all parameters untransformed (identity)
    pub fn identity(n_params: usize) -> Self {
        Self {
            transforms: vec![ParameterTransform::None; n_params],
        }
    }

    /// Create transforms from R saemix-style transform codes
    ///
    /// # Arguments
    /// * `codes` - Vector of transform codes (0=none, 1=log, 2=probit, 3=logit)
    /// * `bounds` - Optional bounds for logit/probit transforms (lower, upper)
    pub fn from_saemix_codes(codes: &[u8], bounds: Option<&[(f64, f64)]>) -> Self {
        let transforms: Vec<ParameterTransform> = codes
            .iter()
            .enumerate()
            .map(|(i, &code)| {
                let (lower, upper) = bounds
                    .and_then(|b| b.get(i))
                    .copied()
                    .unwrap_or((0.0, 1.0));

                match code {
                    0 => ParameterTransform::None,
                    1 => ParameterTransform::LogNormal,
                    2 => ParameterTransform::Probit { lower, upper },
                    3 => ParameterTransform::Logit { lower, upper },
                    _ => ParameterTransform::None,
                }
            })
            .collect();

        Self::new(transforms)
    }

    /// Get the number of parameters
    pub fn n_params(&self) -> usize {
        self.transforms.len()
    }

    /// Get the transformation for a specific parameter
    pub fn get(&self, idx: usize) -> Option<&ParameterTransform> {
        self.transforms.get(idx)
    }

    /// Transform from unconstrained (φ) to constrained (ψ) space
    ///
    /// This is used when:
    /// - Computing likelihood (model needs actual parameter values)
    /// - Reporting individual estimates
    /// - Output/visualization
    ///
    /// # Arguments
    /// * `phi` - Parameter vector in unconstrained space
    ///
    /// # Returns
    /// * Parameter vector in constrained space
    pub fn phi_to_psi(&self, phi: &Col<f64>) -> Col<f64> {
        debug_assert_eq!(phi.nrows(), self.transforms.len());

        Col::from_fn(phi.nrows(), |i| self.transforms[i].phi_to_psi(phi[i]))
    }

    /// Transform from unconstrained (φ) to constrained (ψ) space (Vec version)
    pub fn phi_to_psi_vec(&self, phi: &[f64]) -> Vec<f64> {
        debug_assert_eq!(phi.len(), self.transforms.len());

        phi.iter()
            .enumerate()
            .map(|(i, &p)| self.transforms[i].phi_to_psi(p))
            .collect()
    }

    /// Transform from constrained (ψ) to unconstrained (φ) space
    ///
    /// This is used when:
    /// - Initializing from user-provided values
    /// - Reading data in natural units
    ///
    /// # Arguments
    /// * `psi` - Parameter vector in constrained space
    ///
    /// # Returns
    /// * Parameter vector in unconstrained space
    pub fn psi_to_phi(&self, psi: &Col<f64>) -> Col<f64> {
        debug_assert_eq!(psi.nrows(), self.transforms.len());

        Col::from_fn(psi.nrows(), |i| self.transforms[i].psi_to_phi(psi[i]))
    }

    /// Transform from constrained (ψ) to unconstrained (φ) space (Vec version)
    pub fn psi_to_phi_vec(&self, psi: &[f64]) -> Vec<f64> {
        debug_assert_eq!(psi.len(), self.transforms.len());

        psi.iter()
            .enumerate()
            .map(|(i, &p)| self.transforms[i].psi_to_phi(p))
            .collect()
    }

    /// Compute the log of the Jacobian determinant for φ → ψ
    ///
    /// This is needed for likelihood correction when working in transformed space:
    /// p(ψ) = p(φ) × |dφ/dψ| = p(φ) / |dψ/dφ|
    ///
    /// Returns: Σᵢ log|dψᵢ/dφᵢ|
    pub fn log_jacobian(&self, phi: &Col<f64>) -> f64 {
        debug_assert_eq!(phi.nrows(), self.transforms.len());

        (0..phi.nrows())
            .map(|i| self.transforms[i].log_jacobian(phi[i]))
            .sum()
    }

    /// Check if all ψ values are in valid domains
    pub fn all_valid_psi(&self, psi: &Col<f64>) -> bool {
        (0..psi.nrows()).all(|i| self.transforms[i].is_valid_psi(psi[i]))
    }

    /// Check if any parameters have non-identity transforms
    pub fn has_transforms(&self) -> bool {
        self.transforms.iter().any(|t| *t != ParameterTransform::None)
    }

    /// Get R saemix-style transform codes
    pub fn to_saemix_codes(&self) -> Vec<u8> {
        self.transforms.iter().map(|t| t.to_saemix_code()).collect()
    }
}

impl Default for ParameterTransforms {
    fn default() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transforms() {
        let t = ParameterTransforms::identity(3);

        let phi = Col::from_fn(3, |i| (i + 1) as f64);
        let psi = t.phi_to_psi(&phi);

        for i in 0..3 {
            assert!((phi[i] - psi[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mixed_transforms() {
        let t = ParameterTransforms::new(vec![
            ParameterTransform::None,
            ParameterTransform::LogNormal,
            ParameterTransform::Logit {
                lower: 0.0,
                upper: 1.0,
            },
        ]);

        // Test round-trip
        let psi = Col::from_fn(3, |i| match i {
            0 => 50.0,   // V - no transform
            1 => 5.0,    // CL - log-normal
            2 => 0.8,    // F - logit (0,1)
            _ => 0.0,
        });

        let phi = t.psi_to_phi(&psi);
        let psi_back = t.phi_to_psi(&phi);

        for i in 0..3 {
            assert!(
                (psi[i] - psi_back[i]).abs() < 1e-10,
                "Round-trip failed for param {}: {} != {}",
                i,
                psi[i],
                psi_back[i]
            );
        }
    }

    #[test]
    fn test_from_saemix_codes() {
        let t = ParameterTransforms::from_saemix_codes(
            &[0, 1, 3],
            Some(&[(0.0, 100.0), (0.0, 100.0), (0.0, 1.0)]),
        );

        assert_eq!(t.transforms[0], ParameterTransform::None);
        assert_eq!(t.transforms[1], ParameterTransform::LogNormal);
        assert_eq!(
            t.transforms[2],
            ParameterTransform::Logit {
                lower: 0.0,
                upper: 1.0
            }
        );
    }

    #[test]
    fn test_log_jacobian() {
        let t = ParameterTransforms::new(vec![
            ParameterTransform::LogNormal,
            ParameterTransform::LogNormal,
        ]);

        let phi = Col::from_fn(2, |i| (i + 1) as f64);
        let log_jac = t.log_jacobian(&phi);

        // For log-normal, log|dψ/dφ| = φ, so total = φ₀ + φ₁ = 1 + 2 = 3
        assert!((log_jac - 3.0).abs() < 1e-10);
    }
}
