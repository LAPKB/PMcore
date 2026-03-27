use faer::Col;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use super::state::{PhiVector, PsiVector};
use super::Population;
use anyhow::Result;
use faer::Mat;

pub(crate) struct InitializedPopulationInPhiSpace {
    pub mu_psi: PsiVector,
    pub mu_phi: PhiVector,
    pub omega_phi: Mat<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterTransform {
    None,
    LogNormal,
    Logit { lower: f64, upper: f64 },
    Probit { lower: f64, upper: f64 },
}

impl Default for ParameterTransform {
    fn default() -> Self {
        ParameterTransform::None
    }
}

impl ParameterTransform {
    pub fn logit_unit() -> Self {
        ParameterTransform::Logit {
            lower: 0.0,
            upper: 1.0,
        }
    }

    pub fn logit(lower: f64, upper: f64) -> Self {
        ParameterTransform::Logit { lower, upper }
    }

    pub fn probit(lower: f64, upper: f64) -> Self {
        ParameterTransform::Probit { lower, upper }
    }

    pub fn psi_to_phi(&self, psi: f64) -> f64 {
        match self {
            ParameterTransform::None => psi,
            ParameterTransform::LogNormal => psi.ln(),
            ParameterTransform::Logit { lower, upper } => {
                let normalized = (psi - lower) / (upper - psi);
                normalized.ln()
            }
            ParameterTransform::Probit { lower, upper } => {
                let normalized = (psi - lower) / (upper - lower);
                probit(normalized)
            }
        }
    }

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

    pub fn dpsi_dphi(&self, phi: f64) -> f64 {
        match self {
            ParameterTransform::None => 1.0,
            ParameterTransform::LogNormal => phi.exp(),
            ParameterTransform::Logit { lower, upper } => {
                let exp_phi = phi.exp();
                let denom = (1.0 + exp_phi).powi(2);
                (upper - lower) * exp_phi / denom
            }
            ParameterTransform::Probit { lower, upper } => (upper - lower) * normal_pdf(phi),
        }
    }

    pub fn log_jacobian(&self, phi: f64) -> f64 {
        match self {
            ParameterTransform::None => 0.0,
            ParameterTransform::LogNormal => phi,
            ParameterTransform::Logit { lower, upper } => {
                (upper - lower).ln() + phi - 2.0 * (1.0 + phi.exp()).ln()
            }
            ParameterTransform::Probit { lower, upper } => {
                (upper - lower).ln() + log_normal_pdf(phi)
            }
        }
    }

    pub fn is_valid_psi(&self, psi: f64) -> bool {
        match self {
            ParameterTransform::None => true,
            ParameterTransform::LogNormal => psi > 0.0,
            ParameterTransform::Logit { lower, upper } => psi > *lower && psi < *upper,
            ParameterTransform::Probit { lower, upper } => psi > *lower && psi < *upper,
        }
    }

    pub fn psi_bounds(&self) -> Option<(f64, f64)> {
        match self {
            ParameterTransform::None => None,
            ParameterTransform::LogNormal => Some((0.0, f64::INFINITY)),
            ParameterTransform::Logit { lower, upper } => Some((*lower, *upper)),
            ParameterTransform::Probit { lower, upper } => Some((*lower, *upper)),
        }
    }

    pub fn to_saemix_code(&self) -> u8 {
        match self {
            ParameterTransform::None => 0,
            ParameterTransform::LogNormal => 1,
            ParameterTransform::Probit { .. } => 2,
            ParameterTransform::Logit { .. } => 3,
        }
    }

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

pub fn transforms_from_saemix_codes(codes: &[u8]) -> Vec<ParameterTransform> {
    codes
        .iter()
        .map(|&code| ParameterTransform::from_saemix_code(code))
        .collect()
}

pub fn phi_to_psi_vec(transforms: &[ParameterTransform], phi: &Col<f64>) -> Col<f64> {
    Col::from_fn(phi.nrows(), |index| {
        transforms[index].phi_to_psi(phi[index])
    })
}

pub fn psi_to_phi_vec(transforms: &[ParameterTransform], psi: &Col<f64>) -> Col<f64> {
    Col::from_fn(psi.nrows(), |index| {
        transforms[index].psi_to_phi(psi[index])
    })
}

pub fn phi_to_psi(transforms: &[ParameterTransform], phi: &PhiVector) -> PsiVector {
    PsiVector::from(&phi_to_psi_vec(transforms, &phi.to_col()))
}

pub fn psi_to_phi(transforms: &[ParameterTransform], psi: &PsiVector) -> PhiVector {
    PhiVector::from(&psi_to_phi_vec(transforms, &psi.to_col()))
}

pub fn transform_label(transform: &ParameterTransform) -> &'static str {
    match transform {
        ParameterTransform::None => "Normal",
        ParameterTransform::LogNormal => "LogNormal",
        ParameterTransform::Logit { .. } => "Logit",
        ParameterTransform::Probit { .. } => "Probit",
    }
}

pub fn default_phi_variance(transform: &ParameterTransform) -> Option<f64> {
    match transform {
        ParameterTransform::LogNormal => {
            let cv: f64 = 0.5;
            Some((1.0 + cv * cv).ln())
        }
        ParameterTransform::Logit { .. } | ParameterTransform::Probit { .. } => Some(1.0),
        ParameterTransform::None => None,
    }
}

pub(crate) fn initialize_population_in_phi_space(
    population: &mut Population,
    transforms: &[ParameterTransform],
) -> Result<InitializedPopulationInPhiSpace> {
    let mu_psi = PsiVector::from(population.mu());
    let mu_phi = psi_to_phi(transforms, &mu_psi);
    population.update_mu(mu_phi.to_col())?;

    let n_params = population.npar();
    let mut omega_phi = population.omega().clone();
    for i in 0..n_params {
        if let Some(default_variance) = default_phi_variance(&transforms[i]) {
            omega_phi[(i, i)] = default_variance;
        }
        for j in 0..n_params {
            if i != j {
                omega_phi[(i, j)] = 0.0;
            }
        }
    }
    population.update_omega(omega_phi.clone())?;

    Ok(InitializedPopulationInPhiSpace {
        mu_psi,
        mu_phi,
        omega_phi,
    })
}

#[inline]
fn standard_normal() -> Normal {
    Normal::standard()
}

#[inline]
fn normal_pdf(x: f64) -> f64 {
    standard_normal().pdf(x)
}

#[inline]
fn log_normal_pdf(x: f64) -> f64 {
    standard_normal().ln_pdf(x)
}

#[inline]
fn normal_cdf(x: f64) -> f64 {
    standard_normal().cdf(x)
}

#[inline]
fn probit(p: f64) -> f64 {
    standard_normal().inverse_cdf(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parametric::Population;
    use crate::model::{ParameterSpace, ParameterSpec};

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
        assert!((t.psi_to_phi(1.0) - 0.0).abs() < EPSILON);
        assert!((t.phi_to_psi(0.0) - 1.0).abs() < EPSILON);
        assert!((t.psi_to_phi(std::f64::consts::E) - 1.0).abs() < EPSILON);
        assert!((t.phi_to_psi(1.0) - std::f64::consts::E).abs() < EPSILON);

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
        assert!((t.psi_to_phi(0.5) - 0.0).abs() < EPSILON);
        assert!((t.phi_to_psi(0.0) - 0.5).abs() < EPSILON);

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
        assert!((t.psi_to_phi(55.0) - 0.0).abs() < EPSILON);
        assert!((t.phi_to_psi(0.0) - 55.0).abs() < EPSILON);

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
        assert!(t.psi_to_phi(0.5).abs() < 1e-6);

        let probit_07 = probit(0.7);
        assert!((probit_07 - 0.524).abs() < 0.01);

        let psi = 0.7;
        let phi = t.psi_to_phi(psi);
        let psi_back = t.phi_to_psi(phi);
        assert!((psi_back - psi).abs() < 1e-4);
    }

    #[test]
    fn test_jacobian_lognormal() {
        let t = ParameterTransform::LogNormal;
        let phi: f64 = 1.0;
        let expected = phi.exp();
        assert!((t.dpsi_dphi(phi) - expected).abs() < EPSILON);
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

    #[test]
    fn test_transforms_from_saemix_codes() {
        let transforms = transforms_from_saemix_codes(&[0u8, 1u8, 2u8, 3u8]);
        assert!(matches!(transforms[0], ParameterTransform::None));
        assert!(matches!(transforms[1], ParameterTransform::LogNormal));
        assert!(matches!(transforms[2], ParameterTransform::Probit { .. }));
        assert!(matches!(transforms[3], ParameterTransform::Logit { .. }));
    }

    #[test]
    fn test_phi_psi_vector_roundtrip() {
        let transforms = vec![
            ParameterTransform::None,
            ParameterTransform::LogNormal,
            ParameterTransform::logit(0.0, 1.0),
        ];
        let psi = Col::from_fn(3, |index| match index {
            0 => 2.0,
            1 => 3.0,
            _ => 0.25,
        });

        let phi = psi_to_phi_vec(&transforms, &psi);
        let back = phi_to_psi_vec(&transforms, &phi);

        for index in 0..psi.nrows() {
            assert!((psi[index] - back[index]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_typed_phi_psi_roundtrip() {
        let transforms = vec![
            ParameterTransform::None,
            ParameterTransform::LogNormal,
            ParameterTransform::logit(0.0, 1.0),
        ];
        let psi = PsiVector(vec![2.0, 3.0, 0.25]);

        let phi = psi_to_phi(&transforms, &psi);
        let back = phi_to_psi(&transforms, &phi);

        assert_eq!(psi.0.len(), back.0.len());
        for index in 0..psi.0.len() {
            assert!((psi.0[index] - back.0[index]).abs() < 1e-10);
        }
    }

    #[test]
    fn initialize_population_converts_to_phi_space_and_resets_covariance() {
        let parameters = ParameterSpace::new()
            .add(ParameterSpec::bounded("ke", 0.1, 1.0))
            .add(ParameterSpec::bounded("v", 1.0, 20.0));
        let mut population = Population::from_parameter_space(parameters).unwrap();
        let transforms = vec![ParameterTransform::LogNormal, ParameterTransform::None];

        let initialized = initialize_population_in_phi_space(&mut population, &transforms).unwrap();

        assert!((initialized.mu_psi.0[0] - 0.55).abs() < 1e-12);
        assert!((initialized.mu_phi.0[0] - 0.55_f64.ln()).abs() < 1e-12);
        assert!((population.mu()[0] - 0.55_f64.ln()).abs() < 1e-12);
        assert_eq!(initialized.omega_phi[(0, 1)], 0.0);
        assert_eq!(initialized.omega_phi[(1, 0)], 0.0);
    }
}
