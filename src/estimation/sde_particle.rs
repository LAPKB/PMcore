use pharmsol::equation::SdeSessionError;
use pharmsol::{Parameters, Subject, SDE};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use thiserror::Error;

use crate::estimation::likelihood::observation::{
    assay_error_model_log_likelihood, AssayLikelihoodError,
};
use crate::estimation::likelihood::particle::{ParticleWeightError, ParticleWeights};
use crate::estimation::likelihood::NormalDistributionError;
use crate::{AssayErrorModels, ErrorModelError};

/// Reproducible controls for explicit SDE particle filtering.
#[derive(Clone, Debug, PartialEq)]
pub struct SdeParticleConfig {
    pub particle_count: usize,
    /// Resample when ESS is less than or equal to this fraction of the particle count.
    pub ess_threshold: f64,
    pub process_seed: u64,
    pub resampling_seed: u64,
}

impl SdeParticleConfig {
    pub fn new(particle_count: usize) -> Self {
        Self {
            particle_count,
            ess_threshold: 0.5,
            process_seed: 0,
            resampling_seed: 1,
        }
    }

    pub fn with_ess_threshold(mut self, threshold: f64) -> Self {
        self.ess_threshold = threshold;
        self
    }

    pub fn with_process_seed(mut self, seed: u64) -> Self {
        self.process_seed = seed;
        self
    }

    pub fn with_resampling_seed(mut self, seed: u64) -> Self {
        self.resampling_seed = seed;
        self
    }
}

/// One sequential observation update.
///
/// `normalized_weights` and `effective_sample_size` always describe the same
/// pre-resampling phase. When `resampled` is true, they are the weights and ESS
/// that triggered ancestor selection, not the reset-uniform continuation state.
#[derive(Clone, Debug, PartialEq)]
pub struct SdeParticleRecord {
    pub time: f64,
    pub output: usize,
    pub log_increment: f64,
    pub effective_sample_size: f64,
    pub resampled: bool,
    pub ancestors: Option<Vec<usize>>,
    pub normalized_weights: Vec<f64>,
}

/// Complete sequential particle-filter result for one subject.
#[derive(Clone, Debug, PartialEq)]
pub struct SdeParticleResult {
    pub log_value: f64,
    pub records: Vec<SdeParticleRecord>,
    pub final_normalized_weights: Vec<f64>,
}

/// Contextual failures from explicit SDE particle filtering.
#[derive(Debug, Error)]
pub enum SdeParticleError {
    #[error("particle_count must be greater than zero")]
    InvalidParticleCount,
    #[error("ESS threshold must be finite and in (0, 1]")]
    InvalidEssThreshold,
    #[error(transparent)]
    Session(#[from] SdeSessionError),
    #[error("assay models are invalid for this SDE output context")]
    InvalidAssayModels(#[source] ErrorModelError),
    #[error(
        "particle {particle} has an invalid assay model at time {time}, output {output}: {source}"
    )]
    InvalidParticleModel {
        time: f64,
        output: usize,
        particle: usize,
        #[source]
        source: ErrorModelError,
    },
    #[error("particle {particle} has invalid sigma {sigma} at time {time}, output {output}")]
    InvalidSigma {
        time: f64,
        output: usize,
        particle: usize,
        sigma: f64,
    },
    #[error("particle {particle} produced NaN or positive-infinity score at time {time}, output {output}")]
    InvalidParticleScore {
        time: f64,
        output: usize,
        particle: usize,
    },
    #[error("all particles are impossible at time {time}, output {output}")]
    ImpossibleObservation { time: f64, output: usize },
    #[error("particle weight normalization failed at time {time}, output {output}")]
    NormalizationFailure { time: f64, output: usize },
    #[error("systematic resampling failed at time {time}, output {output}")]
    ResamplingFailure { time: f64, output: usize },
}

/// Observation-conditioned particle filtering for pharmsol SDEs.
pub trait SdeParticleFilter {
    fn particle_filter(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        assay_models: &AssayErrorModels,
        config: &SdeParticleConfig,
    ) -> Result<SdeParticleResult, SdeParticleError>;
}

impl SdeParticleFilter for SDE {
    fn particle_filter(
        &self,
        subject: &Subject,
        parameters: &Parameters,
        assay_models: &AssayErrorModels,
        config: &SdeParticleConfig,
    ) -> Result<SdeParticleResult, SdeParticleError> {
        if config.particle_count == 0 {
            return Err(SdeParticleError::InvalidParticleCount);
        }
        if !config.ess_threshold.is_finite()
            || !(0.0..=1.0).contains(&config.ess_threshold)
            || config.ess_threshold == 0.0
        {
            return Err(SdeParticleError::InvalidEssThreshold);
        }

        let bound_models = if let Some(metadata) = self.metadata() {
            assay_models.bind_outputs(metadata.outputs().iter().map(|output| output.name()))
        } else {
            assay_models.bind_outputs(std::iter::empty::<&str>())
        }
        .map_err(SdeParticleError::InvalidAssayModels)?;

        let mut process_rng = StdRng::seed_from_u64(config.process_seed);
        let mut resampling_rng = StdRng::seed_from_u64(config.resampling_seed);
        let mut session =
            self.particle_session(subject, parameters, config.particle_count, &mut process_rng)?;
        let mut weights = ParticleWeights::uniform(config.particle_count).map_err(|_| {
            SdeParticleError::NormalizationFailure {
                time: 0.0,
                output: 0,
            }
        })?;
        let mut total = 0.0;
        let mut records = Vec::new();

        while let Some(boundary) = session.next_observation()? {
            let time = boundary.time();
            let output = boundary.output_index();
            if boundary.observation().value().is_none() {
                let ess = weights.effective_sample_size();
                let normalized = weights.normalized_weights();
                session.retain_particles()?;
                records.push(SdeParticleRecord {
                    time,
                    output,
                    log_increment: 0.0,
                    effective_sample_size: ess,
                    resampled: false,
                    ancestors: None,
                    normalized_weights: normalized,
                });
                continue;
            }

            let mut increments = Vec::with_capacity(config.particle_count);
            for (particle, prediction) in boundary.predictions().iter().enumerate() {
                match assay_error_model_log_likelihood(prediction, &bound_models) {
                    Ok(increment) => increments.push(increment),
                    Err(AssayLikelihoodError::Impossible) => {
                        // A zero likelihood is valid for one particle; only an
                        // observation where every particle is impossible fails.
                        increments.push(f64::NEG_INFINITY);
                    }
                    Err(AssayLikelihoodError::InvalidScore(_))
                    | Err(AssayLikelihoodError::Distribution(
                        NormalDistributionError::NonFiniteInput,
                    )) => {
                        return Err(SdeParticleError::InvalidParticleScore {
                            time,
                            output,
                            particle,
                        });
                    }
                    Err(AssayLikelihoodError::Distribution(
                        NormalDistributionError::InvalidSigma(sigma),
                    )) => {
                        return Err(SdeParticleError::InvalidSigma {
                            time,
                            output,
                            particle,
                            sigma,
                        });
                    }
                    Err(AssayLikelihoodError::ErrorModel(source)) => {
                        return Err(SdeParticleError::InvalidParticleModel {
                            time,
                            output,
                            particle,
                            source,
                        });
                    }
                }
            }
            if increments.iter().all(|value| *value == f64::NEG_INFINITY) {
                return Err(SdeParticleError::ImpossibleObservation { time, output });
            }
            let log_increment = match weights.update(&increments) {
                Ok(value) => value,
                Err(ParticleWeightError::AllImpossible) => {
                    return Err(SdeParticleError::ImpossibleObservation { time, output });
                }
                Err(_) => {
                    return Err(SdeParticleError::NormalizationFailure { time, output });
                }
            };
            total += log_increment;
            let ess = weights.effective_sample_size();
            let should_resample = ess <= config.ess_threshold * config.particle_count as f64;

            if should_resample {
                let normalized = weights.normalized_weights();
                let spacing = 1.0 / config.particle_count as f64;
                let offset = resampling_rng.random_range(0.0..spacing);
                let ancestors = ParticleWeights::systematic_ancestors(&normalized, offset)
                    .map_err(|_| SdeParticleError::ResamplingFailure { time, output })?;
                session.select_ancestors(&ancestors)?;
                records.push(SdeParticleRecord {
                    time,
                    output,
                    log_increment,
                    effective_sample_size: ess,
                    resampled: true,
                    ancestors: Some(ancestors),
                    normalized_weights: normalized,
                });
                weights.reset_uniform();
            } else {
                let normalized = weights.normalized_weights();
                session.retain_particles()?;
                records.push(SdeParticleRecord {
                    time,
                    output,
                    log_increment,
                    effective_sample_size: ess,
                    resampled: false,
                    ancestors: None,
                    normalized_weights: normalized,
                });
            }
        }

        Ok(SdeParticleResult {
            log_value: total,
            records,
            final_normalized_weights: weights.normalized_weights(),
        })
    }
}
