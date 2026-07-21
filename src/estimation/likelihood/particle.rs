use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum ParticleWeightError {
    #[error("particle likelihood requires at least one particle")]
    Empty,
    #[error(
        "particle likelihood increment count {found} does not match particle count {expected}"
    )]
    IncrementCount { expected: usize, found: usize },
    #[error("particle likelihood increment {index} is NaN or positive infinity")]
    InvalidIncrement { index: usize },
    #[error("all particle likelihood weights are zero")]
    AllImpossible,
    #[error("particle likelihood normalization is non-finite")]
    NormalizationFailure,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum ResamplingError {
    #[error("systematic resampling requires at least one particle")]
    Empty,
    #[error("systematic resampling offset must be in [0, 1/N)")]
    InvalidOffset,
    #[error("systematic resampling weight {index} must be finite and non-negative")]
    InvalidWeight { index: usize },
    #[error("systematic resampling weights must sum to one")]
    InvalidWeightSum,
}

/// Particle likelihood weights stored in normalized log space.
#[derive(Debug, Clone)]
pub(crate) struct ParticleWeights {
    log_weights: Vec<f64>,
}

impl ParticleWeights {
    pub(crate) fn uniform(particle_count: usize) -> Result<Self, ParticleWeightError> {
        if particle_count == 0 {
            return Err(ParticleWeightError::Empty);
        }
        let log_weight = -(particle_count as f64).ln();
        Ok(Self {
            log_weights: vec![log_weight; particle_count],
        })
    }

    pub(crate) fn update(
        &mut self,
        log_likelihood_increments: &[f64],
    ) -> Result<f64, ParticleWeightError> {
        if log_likelihood_increments.len() != self.log_weights.len() {
            return Err(ParticleWeightError::IncrementCount {
                expected: self.log_weights.len(),
                found: log_likelihood_increments.len(),
            });
        }
        if let Some(index) = log_likelihood_increments
            .iter()
            .position(|increment| increment.is_nan() || *increment == f64::INFINITY)
        {
            return Err(ParticleWeightError::InvalidIncrement { index });
        }

        for (log_weight, increment) in self.log_weights.iter_mut().zip(log_likelihood_increments) {
            *log_weight += increment;
        }
        normalize_log_weights(&mut self.log_weights)
    }

    pub(crate) fn effective_sample_size(&self) -> f64 {
        let sum_squared_weights = self
            .log_weights
            .iter()
            .map(|log_weight| (2.0 * log_weight).exp())
            .sum::<f64>();
        1.0 / sum_squared_weights
    }

    pub(crate) fn normalized_weights(&self) -> Vec<f64> {
        self.log_weights
            .iter()
            .map(|log_weight| log_weight.exp())
            .collect()
    }

    pub(crate) fn reset_uniform(&mut self) {
        let log_weight = -(self.log_weights.len() as f64).ln();
        self.log_weights.fill(log_weight);
    }

    pub(crate) fn systematic_ancestors(
        weights: &[f64],
        offset: f64,
    ) -> Result<Vec<usize>, ResamplingError> {
        systematic_ancestors(weights, offset)
    }
}

fn normalize_log_weights(log_weights: &mut [f64]) -> Result<f64, ParticleWeightError> {
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return Err(ParticleWeightError::AllImpossible);
    }
    if !maximum.is_finite() {
        return Err(ParticleWeightError::NormalizationFailure);
    }

    let scaled_sum = log_weights
        .iter()
        .map(|log_weight| (*log_weight - maximum).exp())
        .sum::<f64>();
    let log_normalizer = maximum + scaled_sum.ln();
    if !log_normalizer.is_finite() {
        return Err(ParticleWeightError::NormalizationFailure);
    }
    for log_weight in log_weights {
        *log_weight -= log_normalizer;
    }
    Ok(log_normalizer)
}

/// Select particle ancestors by systematic resampling.
pub(crate) fn systematic_ancestors(
    weights: &[f64],
    offset: f64,
) -> Result<Vec<usize>, ResamplingError> {
    if weights.is_empty() {
        return Err(ResamplingError::Empty);
    }
    let particle_count = weights.len();
    let spacing = 1.0 / particle_count as f64;
    if !offset.is_finite() || !(0.0..spacing).contains(&offset) {
        return Err(ResamplingError::InvalidOffset);
    }
    if let Some(index) = weights
        .iter()
        .position(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(ResamplingError::InvalidWeight { index });
    }
    let weight_sum = weights.iter().sum::<f64>();
    if (weight_sum - 1.0).abs() > 1e-10 {
        return Err(ResamplingError::InvalidWeightSum);
    }

    let mut ancestors = Vec::with_capacity(particle_count);
    let mut cumulative = weights[0];
    let mut ancestor = 0usize;
    for draw_index in 0..particle_count {
        let threshold = offset + draw_index as f64 * spacing;
        while threshold >= cumulative && ancestor + 1 < particle_count {
            ancestor += 1;
            cumulative += weights[ancestor];
        }
        ancestors.push(ancestor);
    }
    Ok(ancestors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_update_returns_log_marginal_and_normalizes_weights() {
        let mut weights = ParticleWeights::uniform(2).unwrap();
        let log_marginal = weights.update(&[0.25_f64.ln(), 0.75_f64.ln()]).unwrap();
        assert!((log_marginal - 0.5_f64.ln()).abs() < 1e-12);
        let normalized = weights.normalized_weights();
        assert!((normalized[0] - 0.25).abs() < 1e-12);
        assert!((normalized[1] - 0.75).abs() < 1e-12);
        assert!((weights.effective_sample_size() - 1.6).abs() < 1e-12);
    }

    #[test]
    fn log_space_update_remains_stable_for_tiny_likelihoods() {
        let mut weights = ParticleWeights::uniform(2).unwrap();
        let log_marginal = weights.update(&[-1_000.0, -1_002.0]).unwrap();
        assert!(log_marginal.is_finite());
        assert!((weights.normalized_weights().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn all_impossible_particles_are_typed() {
        let mut weights = ParticleWeights::uniform(2).unwrap();
        assert_eq!(
            weights.update(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            Err(ParticleWeightError::AllImpossible)
        );
    }

    #[test]
    fn invalid_increment_and_normalization_failures_are_distinct() {
        let mut weights = ParticleWeights::uniform(2).unwrap();
        assert_eq!(
            weights.update(&[f64::NAN, 0.0]),
            Err(ParticleWeightError::InvalidIncrement { index: 0 })
        );
    }

    #[test]
    fn systematic_resampling_is_deterministic_for_supplied_offset() {
        let ancestors = systematic_ancestors(&[0.1, 0.2, 0.7], 0.05).unwrap();
        assert_eq!(ancestors, vec![0, 2, 2]);
    }

    #[test]
    fn resampling_failures_are_typed() {
        assert_eq!(
            systematic_ancestors(&[0.4, 0.4], 0.1),
            Err(ResamplingError::InvalidWeightSum)
        );
    }

    #[test]
    fn resetting_after_resampling_restores_uniform_ess() {
        let mut weights = ParticleWeights::uniform(3).unwrap();
        weights
            .update(&[0.1_f64.ln(), 0.2_f64.ln(), 0.7_f64.ln()])
            .unwrap();
        weights.reset_uniform();
        assert!((weights.effective_sample_size() - 3.0).abs() < 1e-12);
    }
}
