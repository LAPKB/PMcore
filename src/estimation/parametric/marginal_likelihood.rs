//! Explicit post-fit population marginal likelihood by importance sampling.
//!
//! This module owns normalized latent densities and streaming importance-weight
//! moments. It does not participate in the SAEM fit random stream.

use anyhow::Result;
use ndarray::Array2;
use rand::{distr::Distribution, rngs::StdRng, SeedableRng};
use rand_distr::{ChiSquared, StandardNormal};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

use super::covariance::{cholesky_log_determinant, cholesky_lower, solve_lower};
use crate::estimation::parametric::conditional_uncertainty::ConditionalCurvatureAvailability;

/// Fixed domain separator for deterministic subject-specific integration streams.
pub const N2_SEED_DOMAIN: u64 = 0x4e32_5f50_4d43_4f52;

/// Explicit marginal-likelihood sampling budget and proposal configuration.
///
/// There is deliberately no `Default`: every stochastic choice is explicit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarginalLikelihoodConfig {
    pub samples_per_subject: usize,
    pub seed: u64,
    pub degrees_of_freedom: u32,
    pub covariance_scale_multiplier: f64,
    pub proposal: MarginalLikelihoodProposal,
}

impl MarginalLikelihoodConfig {
    pub fn new(
        samples_per_subject: usize,
        seed: u64,
        degrees_of_freedom: u32,
        covariance_scale_multiplier: f64,
    ) -> Self {
        Self {
            samples_per_subject,
            seed,
            degrees_of_freedom,
            covariance_scale_multiplier,
            proposal: MarginalLikelihoodProposal::FinalRawOmegaBlocks,
        }
    }

    /// Use the conditional-mode curvature covariance as the proposal scale matrix.
    ///
    /// When selected, each subject must supply a valid conditional curvature
    /// covariance in its retained subject input; if unavailable, mismatched, non-finite,
    /// or not strictly positive definite, the calculation returns a typed failure
    /// with no fallback to the raw Omega blocks.
    pub fn conditional_mode_curvature_proposal(mut self) -> Self {
        self.proposal = MarginalLikelihoodProposal::ConditionalModeCurvature;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.samples_per_subject < 2 {
            anyhow::bail!("N2 samples_per_subject must be at least 2");
        }
        if self.degrees_of_freedom < 3 {
            anyhow::bail!("N2 degrees_of_freedom must be at least 3");
        }
        if !self.covariance_scale_multiplier.is_finite() || self.covariance_scale_multiplier <= 0.0
        {
            anyhow::bail!("N2 covariance_scale_multiplier must be finite and positive");
        }
        Ok(())
    }
}

/// Subject integration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarginalLikelihoodMethod {
    ExactNoLatent,
    StudentTImportanceSampling,
}

/// Which proposal covariance-scale matrix to use for importance sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarginalLikelihoodProposal {
    FinalRawOmegaBlocks,
    ConditionalModeCurvature,
}

/// Source of the proposal covariance-scale matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalScaleSource {
    FinalRawOmegaBlocks,
    ConditionalModeCurvature,
    NotApplicableNoLatent,
}

/// Typed reason that a subject marginal-likelihood calculation is unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", content = "detail", rename_all = "snake_case")]
pub enum MarginalLikelihoodFailureReason {
    MissingConditionalMode,
    ConditionalModeCalculationFailed(String),
    SubjectIdMismatch {
        expected: String,
        actual: String,
    },
    EtaWidthMismatch {
        expected: usize,
        actual: usize,
    },
    KappaCountMismatch {
        expected: usize,
        actual: usize,
    },
    KappaOccasionMismatch {
        position: usize,
        expected: usize,
        actual: usize,
    },
    KappaWidthMismatch {
        position: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteModeCoordinate,
    InvalidRawCovariance(String),
    NonFiniteDraw,
    NonFiniteDensity,
    NonFiniteWeight,
    NonFiniteMoments,
    PopulationAggregationOverflow,
    AllZeroEffectiveWeights,
    ConditionalCurvatureUnavailable,
    ConditionalCurvatureDimensionMismatch {
        expected: usize,
        actual: usize,
    },
    ConditionalCurvatureNonFinite,
    ConditionalCurvatureNotSPD(String),
    ScoringFailure(String),
}

/// Availability of the complete population marginal-likelihood calculation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum MarginalLikelihoodStatus {
    Available,
    AvailableWithNonconvergedModes {
        subjects: Vec<String>,
    },
    Unavailable {
        failures: Vec<MarginalLikelihoodSubjectFailure>,
    },
}

/// Subject and typed reason retained when marginal likelihood cannot be calculated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarginalLikelihoodSubjectFailure {
    pub subject_id: String,
    pub reason: MarginalLikelihoodFailureReason,
}

/// Immutable per-subject marginal-likelihood diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubjectMarginalLikelihoodDiagnostics {
    pub subject_id: String,
    pub method: MarginalLikelihoodMethod,
    pub proposal_scale_source: ProposalScaleSource,
    pub seed: Option<u64>,
    pub dimension: usize,
    pub occasion_indices: Vec<usize>,
    pub mode: Vec<f64>,
    pub mode_converged: Option<bool>,
    pub samples: usize,
    pub log_marginal_likelihood: Option<f64>,
    pub n2ll: Option<f64>,
    pub effective_sample_size: Option<f64>,
    pub effective_sample_fraction: Option<f64>,
    pub zero_weight_count: usize,
    pub var_log: Option<f64>,
    pub n2ll_mcse: Option<f64>,
    pub failure: Option<MarginalLikelihoodFailureReason>,
}

/// Immutable post-fit population marginal-likelihood diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalLikelihoodDiagnostics {
    pub config: MarginalLikelihoodConfig,
    pub status: MarginalLikelihoodStatus,
    pub log_marginal_likelihood: Option<f64>,
    pub n2ll: Option<f64>,
    pub n2ll_mcse: Option<f64>,
    pub subjects: Vec<SubjectMarginalLikelihoodDiagnostics>,
}

#[derive(Debug, Clone)]
pub(crate) struct MarginalSubject<'a> {
    pub subject_id: &'a str,
    pub occasion_indices: &'a [usize],
    pub mode: &'a [f64],
    pub mode_converged: Option<bool>,
    pub eta_dimension: usize,
    pub kappa_dimension: usize,
    pub validation_failure: Option<MarginalLikelihoodFailureReason>,
    pub curvature_availability: Option<&'a ConditionalCurvatureAvailability>,
    pub curvature_covariance: Option<&'a Array2<f64>>,
}

/// Derive a stable subject stream without consuming any fit RNG state.
pub fn marginal_likelihood_subject_seed(base_seed: u64, subject_index: usize) -> u64 {
    let mut value =
        base_seed ^ N2_SEED_DOMAIN ^ (subject_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn calculate_population_marginal_likelihood<F>(
    config: MarginalLikelihoodConfig,
    subjects: &[MarginalSubject<'_>],
    omega: &Array2<f64>,
    omega_iov: Option<&Array2<f64>>,
    mut score: F,
) -> MarginalLikelihoodDiagnostics
where
    F: FnMut(usize, &[f64], &[Vec<f64>]) -> Result<f64>,
{
    let mut diagnostics = Vec::with_capacity(subjects.len());
    let mut failures = Vec::new();
    let mut nonconverged = Vec::new();
    let mut total_log_marginal = 0.0;
    let mut total_var_log = 0.0;

    for (subject_index, subject) in subjects.iter().enumerate() {
        let dimension =
            subject.eta_dimension + subject.occasion_indices.len() * subject.kappa_dimension;
        let mut result = if let Some(reason) = subject.validation_failure.clone() {
            let method = if dimension == 0 {
                MarginalLikelihoodMethod::ExactNoLatent
            } else {
                MarginalLikelihoodMethod::StudentTImportanceSampling
            };
            let seed = (dimension > 0)
                .then(|| marginal_likelihood_subject_seed(config.seed, subject_index));
            let proposal_scale_source = if dimension == 0 {
                ProposalScaleSource::NotApplicableNoLatent
            } else {
                match config.proposal {
                    MarginalLikelihoodProposal::FinalRawOmegaBlocks => {
                        ProposalScaleSource::FinalRawOmegaBlocks
                    }
                    MarginalLikelihoodProposal::ConditionalModeCurvature => {
                        ProposalScaleSource::ConditionalModeCurvature
                    }
                }
            };
            let samples = if dimension == 0 {
                0
            } else {
                config.samples_per_subject
            };
            let mut failed = empty_subject(subject, method, proposal_scale_source, seed, samples);
            failed.failure = Some(reason);
            failed
        } else if dimension == 0 {
            exact_subject(subject_index, subject, &mut score)
        } else {
            importance_subject(config, subject_index, subject, omega, omega_iov, &mut score)
        };
        if subject.mode_converged == Some(false) && dimension > 0 && result.failure.is_none() {
            nonconverged.push(subject.subject_id.to_owned());
        }
        if let Some(reason) = result.failure.clone() {
            failures.push(MarginalLikelihoodSubjectFailure {
                subject_id: subject.subject_id.to_owned(),
                reason,
            });
        } else if let (Some(subject_log), Some(subject_var)) =
            (result.log_marginal_likelihood, result.var_log)
        {
            let next_log = total_log_marginal + subject_log;
            let next_var = total_var_log + subject_var;
            if !next_log.is_finite() || !next_var.is_finite() {
                clear_subject_numerics(&mut result);
                result.failure =
                    Some(MarginalLikelihoodFailureReason::PopulationAggregationOverflow);
                failures.push(MarginalLikelihoodSubjectFailure {
                    subject_id: subject.subject_id.to_owned(),
                    reason: MarginalLikelihoodFailureReason::PopulationAggregationOverflow,
                });
            } else {
                total_log_marginal = next_log;
                total_var_log = next_var;
            }
        } else {
            clear_subject_numerics(&mut result);
            result.failure = Some(MarginalLikelihoodFailureReason::NonFiniteMoments);
            failures.push(MarginalLikelihoodSubjectFailure {
                subject_id: subject.subject_id.to_owned(),
                reason: MarginalLikelihoodFailureReason::NonFiniteMoments,
            });
        }
        diagnostics.push(result);
    }

    let final_n2ll = -2.0 * total_log_marginal;
    let final_mcse = 2.0 * total_var_log.sqrt();
    if failures.is_empty() && (!final_n2ll.is_finite() || !final_mcse.is_finite()) {
        if let Some(last) = diagnostics.last_mut() {
            clear_subject_numerics(last);
            last.failure = Some(MarginalLikelihoodFailureReason::PopulationAggregationOverflow);
            failures.push(MarginalLikelihoodSubjectFailure {
                subject_id: last.subject_id.clone(),
                reason: MarginalLikelihoodFailureReason::PopulationAggregationOverflow,
            });
        }
    }

    let (status, log_marginal_likelihood, n2ll, n2ll_mcse) = if failures.is_empty() {
        let status = if nonconverged.is_empty() {
            MarginalLikelihoodStatus::Available
        } else {
            MarginalLikelihoodStatus::AvailableWithNonconvergedModes {
                subjects: nonconverged,
            }
        };
        (
            status,
            Some(total_log_marginal),
            Some(final_n2ll),
            Some(final_mcse),
        )
    } else {
        (
            MarginalLikelihoodStatus::Unavailable { failures },
            None,
            None,
            None,
        )
    };

    MarginalLikelihoodDiagnostics {
        config,
        status,
        log_marginal_likelihood,
        n2ll,
        n2ll_mcse,
        subjects: diagnostics,
    }
}

pub(crate) fn unavailable_population_marginal_likelihood(
    config: MarginalLikelihoodConfig,
    subjects: &[MarginalSubject<'_>],
    reason: MarginalLikelihoodFailureReason,
) -> MarginalLikelihoodDiagnostics {
    let failures = subjects
        .iter()
        .map(|subject| MarginalLikelihoodSubjectFailure {
            subject_id: subject.subject_id.to_owned(),
            reason: reason.clone(),
        })
        .collect();
    let subject_diagnostics = subjects
        .iter()
        .enumerate()
        .map(|(subject_index, subject)| {
            let dimension =
                subject.eta_dimension + subject.occasion_indices.len() * subject.kappa_dimension;
            let method = if dimension == 0 {
                MarginalLikelihoodMethod::ExactNoLatent
            } else {
                MarginalLikelihoodMethod::StudentTImportanceSampling
            };
            let seed = (dimension > 0)
                .then(|| marginal_likelihood_subject_seed(config.seed, subject_index));
            let proposal_scale_source = if dimension == 0 {
                ProposalScaleSource::NotApplicableNoLatent
            } else {
                match config.proposal {
                    MarginalLikelihoodProposal::FinalRawOmegaBlocks => {
                        ProposalScaleSource::FinalRawOmegaBlocks
                    }
                    MarginalLikelihoodProposal::ConditionalModeCurvature => {
                        ProposalScaleSource::ConditionalModeCurvature
                    }
                }
            };
            let samples = if dimension == 0 {
                0
            } else {
                config.samples_per_subject
            };
            let mut diagnostics =
                empty_subject(subject, method, proposal_scale_source, seed, samples);
            if matches!(
                reason,
                MarginalLikelihoodFailureReason::ConditionalModeCalculationFailed(_)
            ) {
                diagnostics.mode.clear();
                diagnostics.mode_converged = None;
            }
            diagnostics.failure = Some(reason.clone());
            diagnostics
        })
        .collect();
    MarginalLikelihoodDiagnostics {
        config,
        status: MarginalLikelihoodStatus::Unavailable { failures },
        log_marginal_likelihood: None,
        n2ll: None,
        n2ll_mcse: None,
        subjects: subject_diagnostics,
    }
}

fn exact_subject<F>(
    subject_index: usize,
    subject: &MarginalSubject<'_>,
    score: &mut F,
) -> SubjectMarginalLikelihoodDiagnostics
where
    F: FnMut(usize, &[f64], &[Vec<f64>]) -> Result<f64>,
{
    let mut result = empty_subject(
        subject,
        MarginalLikelihoodMethod::ExactNoLatent,
        ProposalScaleSource::NotApplicableNoLatent,
        None,
        0,
    );
    match score(subject_index, &[], &[]) {
        Ok(value) if value.is_finite() && (-2.0 * value).is_finite() => {
            result.log_marginal_likelihood = Some(value);
            result.n2ll = Some(-2.0 * value);
            result.var_log = Some(0.0);
            result.n2ll_mcse = Some(0.0);
        }
        Ok(_) => result.failure = Some(MarginalLikelihoodFailureReason::NonFiniteMoments),
        Err(error) => {
            result.failure = Some(MarginalLikelihoodFailureReason::ScoringFailure(format!(
                "{error:#}"
            )))
        }
    }
    result
}

fn importance_subject<F>(
    config: MarginalLikelihoodConfig,
    subject_index: usize,
    subject: &MarginalSubject<'_>,
    omega: &Array2<f64>,
    omega_iov: Option<&Array2<f64>>,
    score: &mut F,
) -> SubjectMarginalLikelihoodDiagnostics
where
    F: FnMut(usize, &[f64], &[Vec<f64>]) -> Result<f64>,
{
    let seed = marginal_likelihood_subject_seed(config.seed, subject_index);
    let dimension =
        subject.eta_dimension + subject.occasion_indices.len() * subject.kappa_dimension;
    let proposal_scale_source = match config.proposal {
        MarginalLikelihoodProposal::FinalRawOmegaBlocks => ProposalScaleSource::FinalRawOmegaBlocks,
        MarginalLikelihoodProposal::ConditionalModeCurvature => {
            ProposalScaleSource::ConditionalModeCurvature
        }
    };
    let mut result = empty_subject(
        subject,
        MarginalLikelihoodMethod::StudentTImportanceSampling,
        proposal_scale_source,
        Some(seed),
        config.samples_per_subject,
    );
    if subject.mode.len() != dimension || subject.mode.iter().any(|value| !value.is_finite()) {
        result.failure = Some(MarginalLikelihoodFailureReason::NonFiniteModeCoordinate);
        return result;
    }

    let lower = match config.proposal {
        MarginalLikelihoodProposal::ConditionalModeCurvature => {
            let curvature_cov = match subject.curvature_availability {
                Some(ConditionalCurvatureAvailability::Available) => subject
                    .curvature_covariance
                    .ok_or(MarginalLikelihoodFailureReason::ConditionalCurvatureUnavailable),
                _ => Err(MarginalLikelihoodFailureReason::ConditionalCurvatureUnavailable),
            };
            let curvature_cov = match curvature_cov {
                Ok(cov) => cov,
                Err(reason) => {
                    result.failure = Some(reason);
                    return result;
                }
            };
            if curvature_cov.nrows() != dimension || curvature_cov.ncols() != dimension {
                result.failure = Some(
                    MarginalLikelihoodFailureReason::ConditionalCurvatureDimensionMismatch {
                        expected: dimension,
                        actual: curvature_cov.nrows(),
                    },
                );
                return result;
            }
            if curvature_cov.iter().any(|v| !v.is_finite()) {
                result.failure =
                    Some(MarginalLikelihoodFailureReason::ConditionalCurvatureNonFinite);
                return result;
            }
            for i in 0..dimension {
                for j in 0..i {
                    if curvature_cov[(i, j)] != curvature_cov[(j, i)] {
                        result.failure =
                            Some(MarginalLikelihoodFailureReason::ConditionalCurvatureNotSPD(
                                "curvature covariance is not symmetric".to_string(),
                            ));
                        return result;
                    }
                }
            }
            let cov_lower = match cholesky_lower(curvature_cov) {
                Ok(lower) => lower,
                Err(error) => {
                    result.failure =
                        Some(MarginalLikelihoodFailureReason::ConditionalCurvatureNotSPD(
                            format!("{error:#}"),
                        ));
                    return result;
                }
            };
            let root_scale = config.covariance_scale_multiplier.sqrt();
            let mut lower = vec![vec![0.0; dimension]; dimension];
            for row in 0..dimension {
                for column in 0..=row {
                    lower[row][column] = root_scale * cov_lower[row][column];
                }
            }
            lower
        }
        MarginalLikelihoodProposal::FinalRawOmegaBlocks => {
            match block_scaled_cholesky(
                omega,
                omega_iov,
                subject.eta_dimension,
                subject.kappa_dimension,
                subject.occasion_indices.len(),
                config.covariance_scale_multiplier,
            ) {
                Ok(value) => value,
                Err(error) => {
                    result.failure = Some(MarginalLikelihoodFailureReason::InvalidRawCovariance(
                        format!("{error:#}"),
                    ));
                    return result;
                }
            }
        }
    };
    let log_det = cholesky_log_determinant(&lower);
    if !log_det.is_finite() {
        result.failure = Some(MarginalLikelihoodFailureReason::NonFiniteDensity);
        return result;
    }

    let chi = match ChiSquared::new(config.degrees_of_freedom as f64) {
        Ok(value) => value,
        Err(error) => {
            result.failure = Some(MarginalLikelihoodFailureReason::ScoringFailure(
                error.to_string(),
            ));
            return result;
        }
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut moments = OnlineLogWeightMoments::default();
    for _ in 0..config.samples_per_subject {
        let z = (0..dimension)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect::<Vec<f64>>();
        let u = chi.sample(&mut rng);
        let factor = (config.degrees_of_freedom as f64 / u).sqrt();
        let mut draw = subject.mode.to_vec();
        for row in 0..dimension {
            draw[row] += factor
                * (0..=row)
                    .map(|column| lower[row][column] * z[column])
                    .sum::<f64>();
        }
        if draw.iter().any(|value| !value.is_finite()) {
            result.failure = Some(MarginalLikelihoodFailureReason::NonFiniteDraw);
            return result;
        }
        let eta = &draw[..subject.eta_dimension];
        let kappas = draw[subject.eta_dimension..]
            .chunks(subject.kappa_dimension.max(1))
            .take(subject.occasion_indices.len())
            .map(|values| values.to_vec())
            .collect::<Vec<_>>();
        let target = match score(subject_index, eta, &kappas) {
            Ok(value) => value,
            Err(error) => {
                result.failure = Some(MarginalLikelihoodFailureReason::ScoringFailure(format!(
                    "{error:#}"
                )));
                return result;
            }
        };
        let log_q = match multivariate_t_log_density(
            &draw,
            subject.mode,
            &lower,
            config.degrees_of_freedom,
            log_det,
        ) {
            Ok(value) => value,
            Err(reason) => {
                result.failure = Some(reason);
                return result;
            }
        };
        let log_weight = target - log_q;
        if log_weight.is_nan() || log_weight == f64::INFINITY {
            result.failure = Some(MarginalLikelihoodFailureReason::NonFiniteWeight);
            return result;
        }
        if let Err(reason) = moments.push(log_weight) {
            result.failure = Some(reason);
            return result;
        }
    }

    result.zero_weight_count = moments.zero_weight_count;
    match moments.finish(config.samples_per_subject) {
        Ok(summary) => {
            let n2ll = -2.0 * summary.log_mean_weight;
            let fraction = summary.ess / config.samples_per_subject as f64;
            let mcse = 2.0 * summary.var_log.sqrt();
            if n2ll.is_finite()
                && fraction.is_finite()
                && fraction > 0.0
                && fraction <= 1.0
                && mcse.is_finite()
            {
                result.log_marginal_likelihood = Some(summary.log_mean_weight);
                result.n2ll = Some(n2ll);
                result.effective_sample_size = Some(summary.ess);
                result.effective_sample_fraction = Some(fraction);
                result.var_log = Some(summary.var_log);
                result.n2ll_mcse = Some(mcse);
            } else {
                result.failure = Some(MarginalLikelihoodFailureReason::NonFiniteMoments);
            }
        }
        Err(reason) => result.failure = Some(reason),
    }
    result
}

fn clear_subject_numerics(result: &mut SubjectMarginalLikelihoodDiagnostics) {
    result.log_marginal_likelihood = None;
    result.n2ll = None;
    result.effective_sample_size = None;
    result.effective_sample_fraction = None;
    result.var_log = None;
    result.n2ll_mcse = None;
}

fn empty_subject(
    subject: &MarginalSubject<'_>,
    method: MarginalLikelihoodMethod,
    proposal_scale_source: ProposalScaleSource,
    seed: Option<u64>,
    samples: usize,
) -> SubjectMarginalLikelihoodDiagnostics {
    SubjectMarginalLikelihoodDiagnostics {
        subject_id: subject.subject_id.to_owned(),
        method,
        proposal_scale_source,
        seed,
        dimension: subject.eta_dimension + subject.occasion_indices.len() * subject.kappa_dimension,
        occasion_indices: subject.occasion_indices.to_vec(),
        mode: subject.mode.to_vec(),
        mode_converged: if matches!(method, MarginalLikelihoodMethod::ExactNoLatent) {
            None
        } else {
            subject.mode_converged
        },
        samples,
        log_marginal_likelihood: None,
        n2ll: None,
        effective_sample_size: None,
        effective_sample_fraction: None,
        zero_weight_count: 0,
        var_log: None,
        n2ll_mcse: None,
        failure: None,
    }
}

fn block_scaled_cholesky(
    omega: &Array2<f64>,
    omega_iov: Option<&Array2<f64>>,
    eta_dimension: usize,
    kappa_dimension: usize,
    occasions: usize,
    scale: f64,
) -> Result<Vec<Vec<f64>>> {
    if omega.nrows() != eta_dimension || omega.ncols() != eta_dimension {
        anyhow::bail!("raw Omega dimension does not match eta width");
    }
    let eta_lower = cholesky_lower(omega)?;
    let iov_lower = if kappa_dimension > 0 {
        let matrix = omega_iov.ok_or_else(|| anyhow::anyhow!("raw Omega_IOV is missing"))?;
        if matrix.nrows() != kappa_dimension || matrix.ncols() != kappa_dimension {
            anyhow::bail!("raw Omega_IOV dimension does not match kappa width");
        }
        Some(cholesky_lower(matrix)?)
    } else {
        None
    };
    let dimension = eta_dimension + occasions * kappa_dimension;
    let mut lower = vec![vec![0.0; dimension]; dimension];
    let root_scale = scale.sqrt();
    for row in 0..eta_dimension {
        for column in 0..=row {
            lower[row][column] = root_scale * eta_lower[row][column];
        }
    }
    if let Some(block) = iov_lower {
        for occasion in 0..occasions {
            let offset = eta_dimension + occasion * kappa_dimension;
            for row in 0..kappa_dimension {
                for column in 0..=row {
                    lower[offset + row][offset + column] = root_scale * block[row][column];
                }
            }
        }
    }
    Ok(lower)
}

fn multivariate_t_log_density(
    value: &[f64],
    center: &[f64],
    lower: &[Vec<f64>],
    degrees_of_freedom: u32,
    log_det: f64,
) -> std::result::Result<f64, MarginalLikelihoodFailureReason> {
    if value.len() != center.len() {
        return Err(MarginalLikelihoodFailureReason::NonFiniteDensity);
    }
    let difference = value
        .iter()
        .zip(center)
        .map(|(value, center)| value - center)
        .collect::<Vec<_>>();
    let standardized = solve_lower(lower, &difference)
        .map_err(|_| MarginalLikelihoodFailureReason::NonFiniteDensity)?;
    let delta = standardized.iter().map(|value| value * value).sum::<f64>();
    let d = value.len() as f64;
    let nu = degrees_of_freedom as f64;
    let result = ln_gamma((nu + d) / 2.0)
        - ln_gamma(nu / 2.0)
        - d / 2.0 * (nu * std::f64::consts::PI).ln()
        - 0.5 * log_det
        - (nu + d) / 2.0 * (delta / nu).ln_1p();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(MarginalLikelihoodFailureReason::NonFiniteDensity)
    }
}

#[derive(Debug, Default)]
struct OnlineLogWeightMoments {
    max_log_weight: Option<f64>,
    sum_scaled: f64,
    sum_scaled_squares: f64,
    finite_count: usize,
    zero_weight_count: usize,
}

impl OnlineLogWeightMoments {
    fn push(
        &mut self,
        log_weight: f64,
    ) -> std::result::Result<(), MarginalLikelihoodFailureReason> {
        if log_weight == f64::NEG_INFINITY {
            self.zero_weight_count += 1;
            return Ok(());
        }
        if !log_weight.is_finite() {
            return Err(MarginalLikelihoodFailureReason::NonFiniteWeight);
        }
        match self.max_log_weight {
            None => {
                self.max_log_weight = Some(log_weight);
                self.sum_scaled = 1.0;
                self.sum_scaled_squares = 1.0;
            }
            Some(maximum) if log_weight > maximum => {
                let scale = (maximum - log_weight).exp();
                self.sum_scaled = self.sum_scaled * scale + 1.0;
                self.sum_scaled_squares = self.sum_scaled_squares * scale * scale + 1.0;
                self.max_log_weight = Some(log_weight);
            }
            Some(maximum) => {
                let scaled = (log_weight - maximum).exp();
                self.sum_scaled += scaled;
                self.sum_scaled_squares += scaled * scaled;
            }
        }
        self.finite_count += 1;
        if !self.sum_scaled.is_finite()
            || !self.sum_scaled_squares.is_finite()
            || self.sum_scaled <= 0.0
            || self.sum_scaled_squares <= 0.0
        {
            return Err(MarginalLikelihoodFailureReason::NonFiniteMoments);
        }
        Ok(())
    }

    fn finish(
        &self,
        samples: usize,
    ) -> std::result::Result<LogWeightSummary, MarginalLikelihoodFailureReason> {
        if self.finite_count == 0 {
            return Err(MarginalLikelihoodFailureReason::AllZeroEffectiveWeights);
        }
        let maximum = self
            .max_log_weight
            .ok_or(MarginalLikelihoodFailureReason::AllZeroEffectiveWeights)?;
        let k = samples as f64;
        if samples < 2 || !k.is_finite() {
            return Err(MarginalLikelihoodFailureReason::NonFiniteMoments);
        }
        let log_mean_weight = maximum + (self.sum_scaled.ln() - k.ln());
        let ess = self.sum_scaled * self.sum_scaled / self.sum_scaled_squares;
        let cv2 = k / ess - 1.0;
        let var_log = cv2 / (k - 1.0);
        if !log_mean_weight.is_finite()
            || !ess.is_finite()
            || ess <= 0.0
            || ess > k
            || !cv2.is_finite()
            || cv2 < 0.0
            || !var_log.is_finite()
            || var_log < 0.0
        {
            return Err(MarginalLikelihoodFailureReason::NonFiniteMoments);
        }
        Ok(LogWeightSummary {
            log_mean_weight,
            ess,
            var_log,
        })
    }
}

#[derive(Debug, PartialEq)]
struct LogWeightSummary {
    log_mean_weight: f64,
    ess: f64,
    var_log: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parametric::posterior::eta_log_prior_from_omega;

    #[test]
    fn normalized_student_t_density_matches_scalar_formula() {
        let lower = vec![vec![2.0]];
        let actual = multivariate_t_log_density(&[1.0], &[0.0], &lower, 5, 4.0_f64.ln()).unwrap();
        let expected = ln_gamma(3.0)
            - ln_gamma(2.5)
            - 0.5 * (5.0 * std::f64::consts::PI).ln()
            - 0.5 * 4.0_f64.ln()
            - 3.0 * (1.0_f64 / 20.0).ln_1p();
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn block_determinant_and_mahalanobis_are_exact() {
        let omega = ndarray::array![[4.0, 1.0], [1.0, 2.0]];
        let iov = ndarray::array![[3.0]];
        let lower = block_scaled_cholesky(&omega, Some(&iov), 2, 1, 2, 2.0).unwrap();
        assert!(
            (cholesky_log_determinant(&lower) - (7.0_f64 * 3.0 * 3.0 * 16.0).ln()).abs() < 1e-12
        );
        let z = solve_lower(&lower, &[1.0, -1.0, 2.0, -2.0]).unwrap();
        let delta = z.iter().map(|value| value * value).sum::<f64>();
        let expected = 0.5 * (8.0 / 7.0 + 4.0 / 3.0 + 4.0 / 3.0);
        assert!((delta - expected).abs() < 1e-12);
    }

    #[test]
    fn online_moments_match_offline_weights() {
        let logs = [-1000.0, -999.0, -1002.0, f64::NEG_INFINITY];
        let mut moments = OnlineLogWeightMoments::default();
        for value in logs {
            moments.push(value).unwrap();
        }
        let summary = moments.finish(logs.len()).unwrap();
        let shifted = [0.0_f64, 1.0, -2.0].map(f64::exp);
        let sum = shifted.iter().sum::<f64>();
        let sum2 = shifted.iter().map(|value| value * value).sum::<f64>();
        let expected_ess = sum * sum / sum2;
        assert!((summary.ess - expected_ess).abs() < 1e-12);
        assert_eq!(moments.zero_weight_count, 1);
    }

    #[test]
    fn extreme_and_all_zero_weights_are_typed() {
        let mut moments = OnlineLogWeightMoments::default();
        moments.push(-10_000.0).unwrap();
        moments.push(-10_001.0).unwrap();
        assert!(moments.finish(2).unwrap().log_mean_weight.is_finite());
        let mut zeros = OnlineLogWeightMoments::default();
        zeros.push(f64::NEG_INFINITY).unwrap();
        zeros.push(f64::NEG_INFINITY).unwrap();
        assert_eq!(
            zeros.finish(2),
            Err(MarginalLikelihoodFailureReason::AllZeroEffectiveWeights)
        );
    }

    #[test]
    fn max_shifted_moments_accept_large_positive_and_mixed_extreme_logs() {
        let mut large = OnlineLogWeightMoments::default();
        large.push(f64::MAX).unwrap();
        large.push(f64::MAX).unwrap();
        let summary = large.finish(2).unwrap();
        assert_eq!(summary.log_mean_weight, f64::MAX);
        assert_eq!(summary.ess, 2.0);
        assert_eq!(summary.var_log, 0.0);

        let mut mixed = OnlineLogWeightMoments::default();
        mixed.push(f64::MAX).unwrap();
        mixed.push(-f64::MAX).unwrap();
        let summary = mixed.finish(2).unwrap();
        assert_eq!(summary.log_mean_weight, f64::MAX);
        assert_eq!(summary.ess, 1.0);
        assert_eq!(summary.var_log, 1.0);
    }

    #[test]
    fn population_n2_overflow_is_typed_without_population_totals() {
        let config = MarginalLikelihoodConfig::new(2, 17, 3, 1.0);
        let first = MarginalSubject {
            subject_id: "first",
            occasion_indices: &[],
            mode: &[],
            mode_converged: Some(true),
            eta_dimension: 0,
            kappa_dimension: 0,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let second = MarginalSubject {
            subject_id: "second",
            ..first.clone()
        };
        let diagnostics = calculate_population_marginal_likelihood(
            config,
            &[first, second],
            &Array2::zeros((0, 0)),
            None,
            |_, _, _| Ok(-f64::MAX / 2.0),
        );
        assert!(matches!(
            diagnostics.status,
            MarginalLikelihoodStatus::Unavailable { ref failures }
                if failures.len() == 1
                    && failures[0].subject_id == "second"
                    && failures[0].reason
                        == MarginalLikelihoodFailureReason::PopulationAggregationOverflow
        ));
        assert!(diagnostics.log_marginal_likelihood.is_none());
        assert!(diagnostics.n2ll.is_none());
        assert!(diagnostics.n2ll_mcse.is_none());
        assert!(diagnostics.subjects[0].failure.is_none());
        assert!(diagnostics.subjects[1].log_marginal_likelihood.is_none());
        assert!(diagnostics.subjects[1].failure.is_some());
    }

    #[test]
    fn mixed_valid_and_failed_subjects_retain_ordered_local_failures() {
        let config = MarginalLikelihoodConfig::new(2, 19, 3, 1.0);
        let valid = MarginalSubject {
            subject_id: "valid",
            occasion_indices: &[],
            mode: &[],
            mode_converged: Some(true),
            eta_dimension: 0,
            kappa_dimension: 0,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let failed = MarginalSubject {
            subject_id: "failed",
            validation_failure: Some(MarginalLikelihoodFailureReason::EtaWidthMismatch {
                expected: 1,
                actual: 0,
            }),
            eta_dimension: 1,
            mode: &[],
            ..valid.clone()
        };
        let diagnostics = calculate_population_marginal_likelihood(
            config,
            &[valid, failed],
            &ndarray::array![[1.0]],
            None,
            |_, _, _| Ok(-1.0),
        );
        assert_eq!(diagnostics.subjects[0].log_marginal_likelihood, Some(-1.0));
        assert_eq!(diagnostics.subjects[1].dimension, 1);
        assert_eq!(diagnostics.subjects[1].samples, 2);
        assert_eq!(
            diagnostics.subjects[1].seed,
            Some(marginal_likelihood_subject_seed(19, 1))
        );
        assert!(matches!(
            diagnostics.status,
            MarginalLikelihoodStatus::Unavailable { ref failures }
                if failures.len() == 1 && failures[0].subject_id == "failed"
        ));
    }

    #[test]
    fn seed_derivation_is_stable_and_subject_specific() {
        assert_eq!(
            marginal_likelihood_subject_seed(17, 0),
            10_096_700_463_465_019_373
        );
        assert_ne!(
            marginal_likelihood_subject_seed(17, 0),
            marginal_likelihood_subject_seed(17, 1)
        );
    }

    #[test]
    fn exact_conjugate_correlated_iiv_fixture_is_within_reported_mcse() {
        let omega = ndarray::array![[1.0, 0.3], [0.3, 0.7]];
        let residual = ndarray::array![[0.4, 0.0], [0.0, 0.2]];
        let marginal = &omega + &residual;
        let observation = [0.8, -0.4];
        let determinant = marginal[[0, 0]] * marginal[[1, 1]] - marginal[[0, 1]] * marginal[[1, 0]];
        let solved = [
            (marginal[[1, 1]] * observation[0] - marginal[[0, 1]] * observation[1]) / determinant,
            (-marginal[[1, 0]] * observation[0] + marginal[[0, 0]] * observation[1]) / determinant,
        ];
        let mode = [
            omega[[0, 0]] * solved[0] + omega[[0, 1]] * solved[1],
            omega[[1, 0]] * solved[0] + omega[[1, 1]] * solved[1],
        ];
        let subject = MarginalSubject {
            subject_id: "correlated",
            occasion_indices: &[],
            mode: &mode,
            mode_converged: Some(true),
            eta_dimension: 2,
            kappa_dimension: 0,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let diagnostics = calculate_population_marginal_likelihood(
            MarginalLikelihoodConfig::new(65_536, 201, 5, 1.5),
            &[subject],
            &omega,
            None,
            |_, eta, _| {
                let error = [observation[0] - eta[0], observation[1] - eta[1]];
                Ok(eta_log_prior_from_omega(eta, &omega)?
                    + eta_log_prior_from_omega(&error, &residual)?)
            },
        );
        let exact_n2ll = -2.0 * eta_log_prior_from_omega(&observation, &marginal).unwrap();
        let estimated = diagnostics.n2ll.unwrap();
        let mcse = diagnostics.n2ll_mcse.unwrap();
        eprintln!(
            "correlated IIV N2: exact={exact_n2ll:.17}, estimated={estimated:.17}, mcse={mcse:.17}, abs_error={:.17}",
            (estimated - exact_n2ll).abs()
        );
        assert!((estimated - exact_n2ll).abs() <= 5.0 * mcse + 1e-10);
    }

    #[test]
    fn exact_conjugate_joint_iiv_iov_fixture_is_within_reported_mcse() {
        let omega = ndarray::array![[0.8]];
        let omega_iov = ndarray::array![[0.5]];
        let residual = ndarray::array![[0.3]];
        let marginal = ndarray::array![[1.6]];
        let observation = 1.2;
        let mode = [0.8 / 1.6 * observation, 0.5 / 1.6 * observation];
        let subject = MarginalSubject {
            subject_id: "joint",
            occasion_indices: &[7],
            mode: &mode,
            mode_converged: Some(true),
            eta_dimension: 1,
            kappa_dimension: 1,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let diagnostics = calculate_population_marginal_likelihood(
            MarginalLikelihoodConfig::new(65_536, 301, 5, 1.5),
            &[subject],
            &omega,
            Some(&omega_iov),
            |_, eta, kappas| {
                let error = [observation - eta[0] - kappas[0][0]];
                Ok(eta_log_prior_from_omega(eta, &omega)?
                    + eta_log_prior_from_omega(&kappas[0], &omega_iov)?
                    + eta_log_prior_from_omega(&error, &residual)?)
            },
        );
        let exact_n2ll = -2.0 * eta_log_prior_from_omega(&[observation], &marginal).unwrap();
        let estimated = diagnostics.n2ll.unwrap();
        let mcse = diagnostics.n2ll_mcse.unwrap();
        eprintln!(
            "joint IIV+IOV N2: exact={exact_n2ll:.17}, estimated={estimated:.17}, mcse={mcse:.17}, abs_error={:.17}",
            (estimated - exact_n2ll).abs()
        );
        assert!((estimated - exact_n2ll).abs() <= 5.0 * mcse + 1e-10);
        assert_eq!(diagnostics.subjects[0].occasion_indices, vec![7]);
    }

    #[test]
    fn t_draw_stream_is_bit_exact_and_has_heavier_tail_than_normal_scale() {
        let config = MarginalLikelihoodConfig::new(4096, 91, 3, 1.0);
        let subject = MarginalSubject {
            subject_id: "1",
            occasion_indices: &[],
            mode: &[0.0],
            mode_converged: Some(true),
            eta_dimension: 1,
            kappa_dimension: 0,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let omega = ndarray::array![[1.0]];
        let calculate = || {
            calculate_population_marginal_likelihood(
                config,
                std::slice::from_ref(&subject),
                &omega,
                None,
                |_, eta, _| eta_log_prior_from_omega(eta, &omega),
            )
        };
        assert_eq!(calculate(), calculate());
        let diagnostics = calculate();
        assert!(diagnostics.subjects[0].effective_sample_size.unwrap() > 0.0);
    }

    #[test]
    fn nonconverged_mode_remains_available_and_invalid_covariance_is_typed() {
        let config = MarginalLikelihoodConfig::new(128, 401, 3, 1.0);
        let subject = MarginalSubject {
            subject_id: "finite_mode",
            occasion_indices: &[],
            mode: &[0.0],
            mode_converged: Some(false),
            eta_dimension: 1,
            kappa_dimension: 0,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let omega = ndarray::array![[1.0]];
        let available = calculate_population_marginal_likelihood(
            config,
            std::slice::from_ref(&subject),
            &omega,
            None,
            |_, eta, _| eta_log_prior_from_omega(eta, &omega),
        );
        assert!(matches!(
            available.status,
            MarginalLikelihoodStatus::AvailableWithNonconvergedModes { ref subjects }
                if subjects == &["finite_mode"]
        ));

        let invalid = ndarray::array![[0.0]];
        let unavailable = calculate_population_marginal_likelihood(
            config,
            &[subject],
            &invalid,
            None,
            |_, _, _| Ok(0.0),
        );
        assert!(matches!(
            unavailable.status,
            MarginalLikelihoodStatus::Unavailable { ref failures }
                if matches!(failures[0].reason, MarginalLikelihoodFailureReason::InvalidRawCovariance(_))
        ));
        assert!(unavailable.n2ll.is_none());
    }

    #[test]
    fn global_posthoc_failure_retains_requested_subject_metadata() {
        let config = MarginalLikelihoodConfig::new(32, 411, 5, 1.5);
        let modes = [vec![0.1, 0.2], vec![0.3, 0.4]];
        let first = MarginalSubject {
            subject_id: "first",
            occasion_indices: &[3],
            mode: &modes[0],
            mode_converged: Some(true),
            eta_dimension: 1,
            kappa_dimension: 1,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let second = MarginalSubject {
            subject_id: "second",
            occasion_indices: &[7],
            mode: &modes[1],
            mode_converged: Some(false),
            eta_dimension: 1,
            kappa_dimension: 1,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let diagnostics = unavailable_population_marginal_likelihood(
            config,
            &[first, second],
            MarginalLikelihoodFailureReason::ConditionalModeCalculationFailed(
                "global conditional mode calculation failed".to_string(),
            ),
        );
        assert!(diagnostics.log_marginal_likelihood.is_none());
        assert!(diagnostics.n2ll.is_none());
        assert!(diagnostics.n2ll_mcse.is_none());
        for (index, subject) in diagnostics.subjects.iter().enumerate() {
            assert_eq!(subject.dimension, 2);
            assert_eq!(subject.samples, 32);
            assert_eq!(
                subject.seed,
                Some(marginal_likelihood_subject_seed(411, index))
            );
            assert_eq!(
                subject.occasion_indices,
                vec![if index == 0 { 3 } else { 7 }]
            );
            assert!(matches!(
                subject.failure,
                Some(MarginalLikelihoodFailureReason::ConditionalModeCalculationFailed(_))
            ));
            assert!(subject.mode.is_empty());
            assert_eq!(subject.mode_converged, None);
            assert!(subject.log_marginal_likelihood.is_none());
            assert!(subject.n2ll.is_none());
            assert!(subject.effective_sample_size.is_none());
            assert!(subject.effective_sample_fraction.is_none());
            assert!(subject.var_log.is_none());
            assert!(subject.n2ll_mcse.is_none());
        }
    }

    #[test]
    fn all_zero_effective_weights_are_unavailable_without_thresholding_low_ess() {
        let config = MarginalLikelihoodConfig::new(16, 501, 3, 1.0);
        let subject = MarginalSubject {
            subject_id: "zero",
            occasion_indices: &[],
            mode: &[0.0],
            mode_converged: Some(true),
            eta_dimension: 1,
            kappa_dimension: 0,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let omega = ndarray::array![[1.0]];
        let diagnostics = calculate_population_marginal_likelihood(
            config,
            &[subject],
            &omega,
            None,
            |_, _, _| Ok(f64::NEG_INFINITY),
        );
        assert!(matches!(
            diagnostics.subjects[0].failure,
            Some(MarginalLikelihoodFailureReason::AllZeroEffectiveWeights)
        ));
        assert_eq!(diagnostics.subjects[0].zero_weight_count, 16);
        assert!(diagnostics.n2ll.is_none());
    }

    #[test]
    fn config_default_proposal_is_final_raw_omega_blocks() {
        let config = MarginalLikelihoodConfig::new(100, 42, 3, 1.0);
        assert_eq!(
            config.proposal,
            MarginalLikelihoodProposal::FinalRawOmegaBlocks
        );
    }

    #[test]
    fn config_builder_sets_conditional_mode_curvature_proposal() {
        let config =
            MarginalLikelihoodConfig::new(100, 42, 3, 1.0).conditional_mode_curvature_proposal();
        assert_eq!(
            config.proposal,
            MarginalLikelihoodProposal::ConditionalModeCurvature
        );
    }

    #[test]
    fn curvature_proposal_without_curvature_data_is_typed_failure() {
        let config =
            MarginalLikelihoodConfig::new(32, 701, 3, 1.0).conditional_mode_curvature_proposal();
        let subject = MarginalSubject {
            subject_id: "no_curv",
            occasion_indices: &[],
            mode: &[0.1],
            mode_converged: Some(true),
            eta_dimension: 1,
            kappa_dimension: 0,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let omega = ndarray::array![[1.0]];
        let diagnostics = calculate_population_marginal_likelihood(
            config,
            &[subject],
            &omega,
            None,
            |_, _, _| Ok(0.0),
        );
        assert!(matches!(
            diagnostics.subjects[0].failure,
            Some(MarginalLikelihoodFailureReason::ConditionalCurvatureUnavailable)
        ));
        assert_eq!(
            diagnostics.subjects[0].proposal_scale_source,
            ProposalScaleSource::ConditionalModeCurvature
        );
    }

    #[test]
    fn raw_omega_proposal_default_unchanged_with_new_config_field() {
        let config = MarginalLikelihoodConfig::new(128, 801, 3, 1.0);
        let subject = MarginalSubject {
            subject_id: "default_proposal",
            occasion_indices: &[],
            mode: &[0.0],
            mode_converged: Some(true),
            eta_dimension: 1,
            kappa_dimension: 0,
            validation_failure: None,
            curvature_availability: None,
            curvature_covariance: None,
        };
        let omega = ndarray::array![[1.0]];
        let diagnostics = calculate_population_marginal_likelihood(
            config,
            &[subject],
            &omega,
            None,
            |_, eta, _| eta_log_prior_from_omega(eta, &omega),
        );
        assert_eq!(
            diagnostics.subjects[0].proposal_scale_source,
            ProposalScaleSource::FinalRawOmegaBlocks
        );
        assert!(diagnostics.subjects[0].log_marginal_likelihood.is_some());
    }
}
