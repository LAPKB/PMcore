use anyhow::{bail, Result};

use crate::model::ParameterScale;

use super::transforms::{phi_to_psi, psi_to_phi};

pub(crate) fn population_phi(
    population_psi: &[f64],
    scales: &[ParameterScale],
) -> Result<Vec<f64>> {
    validate_widths(
        population_psi.len(),
        scales.len(),
        "population parameters",
        "scales",
    )?;
    Ok(population_psi
        .iter()
        .zip(scales.iter())
        .map(|(psi, scale)| psi_to_phi(*psi, *scale))
        .collect())
}

pub(crate) fn population_psi(
    population_phi: &[f64],
    scales: &[ParameterScale],
) -> Result<Vec<f64>> {
    validate_widths(
        population_phi.len(),
        scales.len(),
        "population phi",
        "scales",
    )?;
    Ok(population_phi
        .iter()
        .zip(scales.iter())
        .map(|(phi, scale)| phi_to_psi(*phi, *scale))
        .collect())
}

pub(crate) fn individual_phi(
    population_psi: &[f64],
    scales: &[ParameterScale],
    random_effect_indices: &[usize],
    eta: &[f64],
) -> Result<Vec<f64>> {
    validate_widths(
        population_psi.len(),
        scales.len(),
        "population parameters",
        "scales",
    )?;
    validate_widths(
        random_effect_indices.len(),
        eta.len(),
        "random-effect indices",
        "eta",
    )?;

    let mut phi = population_phi(population_psi, scales)?;
    let mut seen = vec![false; population_psi.len()];
    for (eta_index, parameter_index) in random_effect_indices.iter().copied().enumerate() {
        if parameter_index >= population_psi.len() {
            bail!(
                "random-effect parameter index {parameter_index} exceeds parameter width {}",
                population_psi.len()
            );
        }
        if seen[parameter_index] {
            bail!("random-effect parameter index {parameter_index} is duplicated");
        }
        seen[parameter_index] = true;
        phi[parameter_index] += eta[eta_index];
    }
    Ok(phi)
}

pub(crate) fn individual_psi(
    population_parameters: &[f64],
    scales: &[ParameterScale],
    random_effect_indices: &[usize],
    eta: &[f64],
) -> Result<Vec<f64>> {
    let phi = individual_phi(population_parameters, scales, random_effect_indices, eta)?;
    population_psi(&phi, scales)
}

/// Construct one occasion's ψ-space parameters by adding subject η and
/// occasion κ in transformed φ-space.
pub(crate) fn occasion_psi(
    population_parameters: &[f64],
    scales: &[ParameterScale],
    random_effect_indices: &[usize],
    eta: &[f64],
    iov_effect_indices: &[usize],
    kappa: &[f64],
) -> Result<Vec<f64>> {
    validate_widths(
        iov_effect_indices.len(),
        kappa.len(),
        "IOV-effect indices",
        "kappa",
    )?;
    let mut phi = individual_phi(population_parameters, scales, random_effect_indices, eta)?;
    let mut seen = vec![false; population_parameters.len()];
    for (kappa_index, parameter_index) in iov_effect_indices.iter().copied().enumerate() {
        if parameter_index >= population_parameters.len() {
            bail!(
                "IOV-effect parameter index {parameter_index} exceeds parameter width {}",
                population_parameters.len()
            );
        }
        if seen[parameter_index] {
            bail!("IOV-effect parameter index {parameter_index} is duplicated");
        }
        seen[parameter_index] = true;
        phi[parameter_index] += kappa[kappa_index];
    }
    population_psi(&phi, scales)
}

/// Add η to an already resolved subject-specific population mean in φ-space.
///
/// The no-covariate helpers above intentionally remain unchanged; covariate
/// execution calls this helper only after resolving the subject mean.
#[allow(dead_code, reason = "N5 subject-mean execution helper")]
pub(crate) fn individual_phi_from_subject_mean(
    subject_mu_phi: &[f64],
    random_effect_indices: &[usize],
    eta: &[f64],
) -> Result<Vec<f64>> {
    validate_widths(
        random_effect_indices.len(),
        eta.len(),
        "random-effect indices",
        "eta",
    )?;
    let mut phi = subject_mu_phi.to_vec();
    let mut seen = vec![false; phi.len()];
    for (eta_index, parameter_index) in random_effect_indices.iter().copied().enumerate() {
        if parameter_index >= phi.len() {
            bail!(
                "random-effect parameter index {parameter_index} exceeds parameter width {}",
                phi.len()
            );
        }
        if seen[parameter_index] {
            bail!("random-effect parameter index {parameter_index} is duplicated");
        }
        seen[parameter_index] = true;
        phi[parameter_index] += eta[eta_index];
    }
    Ok(phi)
}

#[allow(dead_code, reason = "N5 subject-mean execution helper")]
pub(crate) fn individual_psi_from_subject_mean(
    subject_mu_phi: &[f64],
    scales: &[ParameterScale],
    random_effect_indices: &[usize],
    eta: &[f64],
) -> Result<Vec<f64>> {
    validate_widths(subject_mu_phi.len(), scales.len(), "subject mean", "scales")?;
    population_psi(
        &individual_phi_from_subject_mean(subject_mu_phi, random_effect_indices, eta)?,
        scales,
    )
}

#[allow(dead_code, reason = "N5 subject-mean execution helper")]
pub(crate) fn occasion_psi_from_subject_mean(
    subject_mu_phi: &[f64],
    scales: &[ParameterScale],
    random_effect_indices: &[usize],
    eta: &[f64],
    iov_effect_indices: &[usize],
    kappa: &[f64],
) -> Result<Vec<f64>> {
    validate_widths(
        iov_effect_indices.len(),
        kappa.len(),
        "IOV-effect indices",
        "kappa",
    )?;
    let mut phi = individual_phi_from_subject_mean(subject_mu_phi, random_effect_indices, eta)?;
    let mut seen = vec![false; phi.len()];
    for (kappa_index, parameter_index) in iov_effect_indices.iter().copied().enumerate() {
        if parameter_index >= phi.len() {
            bail!(
                "IOV-effect parameter index {parameter_index} exceeds parameter width {}",
                phi.len()
            );
        }
        if seen[parameter_index] {
            bail!("IOV-effect parameter index {parameter_index} is duplicated");
        }
        seen[parameter_index] = true;
        phi[parameter_index] += kappa[kappa_index];
    }
    population_psi(&phi, scales)
}

fn validate_widths(left: usize, right: usize, left_name: &str, right_name: &str) -> Result<()> {
    if left != right {
        bail!("{left_name} has width {left} but {right_name} has width {right}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_is_additive_in_phi_space() {
        let population = vec![0.2, 10.0];
        let scales = vec![ParameterScale::Log, ParameterScale::Log];
        let eta = vec![2.0_f64.ln(), 0.5_f64.ln()];
        let individual = individual_psi(&population, &scales, &[0, 1], &eta).unwrap();

        assert!((individual[0] - 0.4).abs() < 1e-12);
        assert!((individual[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn occasion_parameters_add_eta_and_kappa_in_phi_space() {
        let population = vec![0.2, 10.0];
        let scales = vec![ParameterScale::Log, ParameterScale::Log];
        let parameters = occasion_psi(
            &population,
            &scales,
            &[0],
            &[2.0_f64.ln()],
            &[0, 1],
            &[0.5_f64.ln(), 2.0_f64.ln()],
        )
        .unwrap();

        assert!((parameters[0] - 0.2).abs() < 1e-12);
        assert!((parameters[1] - 20.0).abs() < 1e-12);
    }

    #[test]
    fn subject_means_convert_exactly_across_all_supported_transforms() {
        let mean = vec![2.0, 3.0_f64.ln(), 0.0, 0.0];
        let scales = vec![
            ParameterScale::Identity,
            ParameterScale::Log,
            ParameterScale::Logit {
                lower: -2.0,
                upper: 6.0,
            },
            ParameterScale::Probit {
                lower: 10.0,
                upper: 14.0,
            },
        ];
        let psi = individual_psi_from_subject_mean(&mean, &scales, &[], &[]).unwrap();
        assert!((psi[0] - 2.0).abs() < 1e-12);
        assert!((psi[1] - 3.0).abs() < 1e-12);
        assert!((psi[2] - 2.0).abs() < 1e-12);
        assert!((psi[3] - 12.0).abs() < 1e-12);
        let occasion =
            occasion_psi_from_subject_mean(&mean, &scales, &[], &[], &[0, 1], &[1.5, 2.0_f64.ln()])
                .unwrap();
        assert!((occasion[0] - 3.5).abs() < 1e-12);
        assert!((occasion[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn parameters_without_random_effects_ignore_eta() {
        let population = vec![0.2, 10.0];
        let scales = vec![ParameterScale::Log, ParameterScale::Log];
        let individual = individual_psi(&population, &scales, &[0], &[2.0_f64.ln()]).unwrap();

        assert!((individual[0] - 0.4).abs() < 1e-12);
        assert!((individual[1] - 10.0).abs() < 1e-12);
    }
}
