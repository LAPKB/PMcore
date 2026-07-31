use anyhow::{bail, Result};
use ndarray::Array2;

#[cfg(test)]
use super::covariance::ensure_positive_definite_covariance;

/// SAEM φ sufficient statistics for models without covariate effects.
///
/// This uses first and second φ moments in the no-covariate case:
/// `mean_phi = E[φ]` and `second_moment = E[φφᵀ]`. Ω is then the centered
/// covariance `E[(φ-μ)(φ-μ)ᵀ]`, with a diagonal floor to prevent the early
/// collapse identified by numerical robustness analysis.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PhiSufficientStatistics {
    pub(crate) mean_phi: Vec<f64>,
    pub(crate) second_moment: Array2<f64>,
}

/// Subject-resolved moments used only when a covariate model is active.
///
/// `expected_phi` preserves deterministic subject order while
/// `global_second_moment` averages all subject/chain outer products in the IIV
/// coordinate system.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CovariateSufficientStatistics {
    pub(crate) expected_phi: Vec<Vec<f64>>,
    pub(crate) global_second_moment: Array2<f64>,
}

impl CovariateSufficientStatistics {
    pub(crate) fn from_subject_chains(subject_phi: &[Vec<Vec<f64>>]) -> Result<Self> {
        let Some(first_chain) = subject_phi.first().and_then(|subject| subject.first()) else {
            bail!("cannot build covariate statistics without subjects and chains");
        };
        let width = first_chain.len();
        if subject_phi
            .iter()
            .any(|subject| subject.is_empty() || subject.iter().any(|phi| phi.len() != width))
        {
            bail!("covariate phi statistic dimensions do not match");
        }
        let mut expected_phi = Vec::with_capacity(subject_phi.len());
        let mut global_second_moment = Array2::zeros((width, width));
        let mut samples = 0usize;
        for subject in subject_phi {
            let mut mean = vec![0.0; width];
            for phi in subject {
                samples += 1;
                for row in 0..width {
                    mean[row] += phi[row];
                    for column in 0..width {
                        global_second_moment[[row, column]] += phi[row] * phi[column];
                    }
                }
            }
            let chains = subject.len() as f64;
            mean.iter_mut().for_each(|value| *value /= chains);
            expected_phi.push(mean);
        }
        global_second_moment.mapv_inplace(|value| value / samples as f64);
        Ok(Self {
            expected_phi,
            global_second_moment,
        })
    }

    /// Update both raw moments with one coherent stochastic-approximation gain.
    ///
    /// These moments are combined later to form a centered covariance. Using
    /// different gain histories can make that algebraic combination indefinite
    /// even when every observed moment pair is valid.
    pub(crate) fn stochastic_update(&mut self, observed: &Self, step_size: f64) -> Result<()> {
        if self.expected_phi.len() != observed.expected_phi.len()
            || self.global_second_moment.raw_dim() != observed.global_second_moment.raw_dim()
            || self
                .expected_phi
                .iter()
                .zip(&observed.expected_phi)
                .any(|(left, right)| left.len() != right.len())
        {
            bail!("covariate sufficient statistics dimensions do not match");
        }
        for (current, target) in self.expected_phi.iter_mut().zip(&observed.expected_phi) {
            for (value, observed) in current.iter_mut().zip(target) {
                *value += step_size * (*observed - *value);
            }
        }
        self.global_second_moment = &self.global_second_moment
            + &((&observed.global_second_moment - &self.global_second_moment) * step_size);
        Ok(())
    }
}

impl PhiSufficientStatistics {
    pub(crate) fn from_subject_phi(subject_phi: &[Vec<f64>]) -> Result<Self> {
        let Some(first) = subject_phi.first() else {
            bail!("cannot build phi statistics without subjects");
        };
        let n_subjects = subject_phi.len();
        let n_parameters = first.len();
        if n_parameters == 0 {
            bail!("cannot build phi statistics without parameters");
        }
        if subject_phi.iter().any(|row| row.len() != n_parameters) {
            bail!("all subject phi rows must have the same width");
        }

        let mut mean_phi = vec![0.0; n_parameters];
        let mut second_moment = Array2::zeros((n_parameters, n_parameters));
        for phi in subject_phi {
            for parameter_index in 0..n_parameters {
                mean_phi[parameter_index] += phi[parameter_index];
                for other_index in 0..n_parameters {
                    second_moment[[parameter_index, other_index]] +=
                        phi[parameter_index] * phi[other_index];
                }
            }
        }

        let scale = n_subjects as f64;
        for value in &mut mean_phi {
            *value /= scale;
        }
        second_moment.mapv_inplace(|value| value / scale);

        Ok(Self {
            mean_phi,
            second_moment,
        })
    }

    pub(crate) fn stochastic_update_with_steps(
        &mut self,
        observed: &Self,
        mean_step_size: f64,
        second_moment_step_size: f64,
    ) -> Result<()> {
        if self.mean_phi.len() != observed.mean_phi.len()
            || self.second_moment.raw_dim() != observed.second_moment.raw_dim()
        {
            bail!("phi sufficient statistics dimensions do not match");
        }

        for (current, target) in self.mean_phi.iter_mut().zip(observed.mean_phi.iter()) {
            *current += mean_step_size * (*target - *current);
        }
        self.second_moment = &self.second_moment
            + &((&observed.second_moment - &self.second_moment) * second_moment_step_size);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn omega(&self, minimum_variance: f64) -> Array2<f64> {
        let all_indices = (0..self.mean_phi.len()).collect::<Vec<_>>();
        self.omega_for_indices(&all_indices, minimum_variance)
            .expect("all parameter indices are valid")
    }

    #[cfg(test)]
    pub(crate) fn omega_for_indices(
        &self,
        random_effect_indices: &[usize],
        minimum_variance: f64,
    ) -> Result<Array2<f64>> {
        self.omega_around_mean(random_effect_indices, &self.mean_phi, minimum_variance)
    }

    #[cfg(test)]
    pub(crate) fn omega_around_mean(
        &self,
        random_effect_indices: &[usize],
        population_phi: &[f64],
        minimum_variance: f64,
    ) -> Result<Array2<f64>> {
        let covariance = self.covariance_around_mean(random_effect_indices, population_phi)?;
        Ok(ensure_positive_definite_covariance(
            &covariance,
            minimum_variance,
        ))
    }

    /// Raw centered covariance before structural/fixed Ω constraints and
    /// positive-definite guardrails are applied.
    #[cfg(test)]
    pub(crate) fn covariance_around_mean(
        &self,
        random_effect_indices: &[usize],
        population_phi: &[f64],
    ) -> Result<Array2<f64>> {
        if population_phi.len() != self.mean_phi.len() {
            bail!(
                "population phi has width {} but statistics have width {}",
                population_phi.len(),
                self.mean_phi.len()
            );
        }

        let mut seen = vec![false; self.mean_phi.len()];
        for parameter_index in random_effect_indices.iter().copied() {
            if parameter_index >= self.mean_phi.len() {
                bail!(
                    "random-effect parameter index {parameter_index} exceeds parameter width {}",
                    self.mean_phi.len()
                );
            }
            if seen[parameter_index] {
                bail!("random-effect parameter index {parameter_index} is duplicated");
            }
            seen[parameter_index] = true;
        }

        Ok(Array2::from_shape_fn(
            (random_effect_indices.len(), random_effect_indices.len()),
            |(row, col)| {
                let parameter_row = random_effect_indices[row];
                let parameter_col = random_effect_indices[col];
                self.second_moment[[parameter_row, parameter_col]]
                    - self.mean_phi[parameter_row] * population_phi[parameter_col]
                    - population_phi[parameter_row] * self.mean_phi[parameter_col]
                    + population_phi[parameter_row] * population_phi[parameter_col]
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parametric::covariance::cholesky_lower;

    #[test]
    fn covariate_raw_moments_share_one_gain_and_remain_coherent() {
        let mut statistics =
            CovariateSufficientStatistics::from_subject_chains(&[vec![vec![0.0]]]).unwrap();
        let observed =
            CovariateSufficientStatistics::from_subject_chains(&[vec![vec![10.0]]]).unwrap();

        statistics.stochastic_update(&observed, 0.1).unwrap();

        assert_eq!(statistics.expected_phi, vec![vec![1.0]]);
        assert!((statistics.global_second_moment[[0, 0]] - 10.0).abs() < 1e-12);
        let centered = statistics.global_second_moment[[0, 0]]
            - statistics.expected_phi[0][0] * statistics.expected_phi[0][0];
        assert!((centered - 9.0).abs() < 1e-12);
    }

    #[test]
    fn legacy_split_raw_moment_gains_can_leave_the_gaussian_moment_cone() {
        let initial =
            CovariateSufficientStatistics::from_subject_chains(&[vec![vec![0.0]]]).unwrap();
        let observed =
            CovariateSufficientStatistics::from_subject_chains(&[vec![vec![10.0]]]).unwrap();

        // Reproduce the legacy exploration update: full gain for E[phi] but
        // 0.1 gain for E[phi phi']. Each input moment pair is realizable, but
        // their differently weighted combination is not.
        let legacy_mean = observed.expected_phi[0][0];
        let legacy_second = initial.global_second_moment[[0, 0]]
            + 0.1 * (observed.global_second_moment[[0, 0]] - initial.global_second_moment[[0, 0]]);
        let legacy_centered = legacy_second - legacy_mean * legacy_mean;

        assert_eq!(legacy_centered, -90.0);
        assert!(legacy_centered < 0.0);
    }

    #[test]
    fn phi_statistics_compute_mean_and_floored_covariance() {
        let stats =
            PhiSufficientStatistics::from_subject_phi(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        assert_eq!(stats.mean_phi, vec![2.0, 3.0]);
        let omega = stats.omega(1e-6);
        assert!(omega[[0, 0]] >= 1.0);
        assert!(omega[[1, 1]] >= 1.0);
        assert!((omega[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((omega[[1, 0]] - 1.0).abs() < 1e-12);
        assert!(cholesky_lower(&omega).is_ok());
    }

    #[test]
    fn omega_diagonal_is_floored_when_eta_has_not_moved() {
        let stats =
            PhiSufficientStatistics::from_subject_phi(&[vec![1.0, 2.0], vec![1.0, 2.0]]).unwrap();

        let omega = stats.omega(1e-6);
        assert_eq!(omega[[0, 0]], 1e-6);
        assert_eq!(omega[[1, 1]], 1e-6);
    }

    #[test]
    fn omega_is_regularized_to_positive_definite() {
        let stats =
            PhiSufficientStatistics::from_subject_phi(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let omega = stats.omega(1e-6);
        assert!(cholesky_lower(&omega).is_ok());
    }

    #[test]
    fn omega_uses_only_declared_random_effect_dimensions() {
        let stats = PhiSufficientStatistics::from_subject_phi(&[
            vec![1.0, 10.0, 2.0],
            vec![3.0, 10.0, 6.0],
        ])
        .unwrap();

        let omega = stats.omega_for_indices(&[0, 2], 1e-6).unwrap();
        assert_eq!(omega.dim(), (2, 2));
        assert!((omega[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((omega[[1, 0]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn omega_centers_around_fixed_population_mean() {
        let stats = PhiSufficientStatistics::from_subject_phi(&[vec![2.0], vec![4.0]]).unwrap();

        let omega = stats.omega_around_mean(&[0], &[1.0], 1e-6).unwrap();
        assert!((omega[[0, 0]] - 5.0).abs() < 1e-12);
    }
}
