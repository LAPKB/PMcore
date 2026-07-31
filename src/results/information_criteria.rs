use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::estimation::parametric::marginal_likelihood::{
    MarginalLikelihoodDiagnostics, MarginalLikelihoodStatus,
};

use super::{InformationCoordinate, InformationCoordinateKind};

/// Availability of post-fit information criteria derived from population marginal N2LL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum InformationCriteriaStatus {
    NotRequested,
    Available,
    AvailableWithNonconvergedModes {
        subjects: Vec<String>,
    },
    Unavailable {
        reason: InformationCriteriaUnavailableReason,
    },
}

/// Typed reason that AIC/BIC cannot be derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", content = "detail", rename_all = "snake_case")]
pub enum InformationCriteriaUnavailableReason {
    ZeroSubjects,
    SubjectCountNotExactlyRepresentable { subject_count: usize },
    ParameterCountOverflow,
    ParameterCountNotExactlyRepresentable { parameter_count: usize },
    InconsistentCoordinateIndices,
    DuplicateCoordinateSource,
    NoncanonicalCoordinateOrder,
    InvalidCovarianceCoordinate { row: usize, column: usize },
    UnknownResidualComponent { component: String },
    SourceMarginalLikelihoodUnavailable,
    ImpossibleMarginalLikelihoodState,
    NonFiniteMarginalN2ll,
    NonFiniteMarginalN2llMcse,
    NegativeMarginalN2llMcse,
    NonFinitePenalty,
    NonFiniteCriterion,
}

/// Deterministic free-population-coordinate count used by both penalties.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InformationCriteriaParameterCount {
    pub population: usize,
    pub covariate: usize,
    pub omega: usize,
    pub omega_iov: usize,
    pub residual: usize,
    pub total: usize,
}

/// Sample-size convention used by BIC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InformationCriteriaSampleSizeConvention {
    IndependentSubjects,
}

/// Immutable AIC/BIC diagnostics derived only from population marginal N2LL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InformationCriteriaDiagnostics {
    pub status: InformationCriteriaStatus,
    pub parameter_count: InformationCriteriaParameterCount,
    pub sample_size_convention: InformationCriteriaSampleSizeConvention,
    pub subject_count: usize,
    pub source_marginal_n2ll: Option<f64>,
    pub source_marginal_n2ll_mcse: Option<f64>,
    pub aic: Option<f64>,
    pub bic: Option<f64>,
    pub aic_mcse: Option<f64>,
    pub bic_mcse: Option<f64>,
}

const MAX_EXACT_INTEGER_F64: usize = 1usize << f64::MANTISSA_DIGITS;

pub(crate) fn derive_information_criteria(
    marginal: Option<&MarginalLikelihoodDiagnostics>,
    coordinates: &[InformationCoordinate],
    subject_count: usize,
) -> InformationCriteriaDiagnostics {
    let counts = match count_parameters(coordinates) {
        Ok(counts) => counts,
        Err(reason) => {
            return unavailable(
                reason,
                InformationCriteriaParameterCount::default(),
                subject_count,
            )
        }
    };
    if subject_count == 0 {
        return unavailable(
            InformationCriteriaUnavailableReason::ZeroSubjects,
            counts,
            subject_count,
        );
    }
    if subject_count > MAX_EXACT_INTEGER_F64 {
        return unavailable(
            InformationCriteriaUnavailableReason::SubjectCountNotExactlyRepresentable {
                subject_count,
            },
            counts,
            subject_count,
        );
    }
    if counts.total > MAX_EXACT_INTEGER_F64 {
        return unavailable(
            InformationCriteriaUnavailableReason::ParameterCountNotExactlyRepresentable {
                parameter_count: counts.total,
            },
            counts,
            subject_count,
        );
    }

    let Some(marginal) = marginal else {
        return InformationCriteriaDiagnostics {
            status: InformationCriteriaStatus::NotRequested,
            parameter_count: counts,
            sample_size_convention: InformationCriteriaSampleSizeConvention::IndependentSubjects,
            subject_count,
            source_marginal_n2ll: None,
            source_marginal_n2ll_mcse: None,
            aic: None,
            bic: None,
            aic_mcse: None,
            bic_mcse: None,
        };
    };

    match &marginal.status {
        MarginalLikelihoodStatus::Unavailable { .. } => {
            if marginal.n2ll.is_some() || marginal.n2ll_mcse.is_some() {
                unavailable(
                    InformationCriteriaUnavailableReason::ImpossibleMarginalLikelihoodState,
                    counts,
                    subject_count,
                )
            } else {
                unavailable(
                    InformationCriteriaUnavailableReason::SourceMarginalLikelihoodUnavailable,
                    counts,
                    subject_count,
                )
            }
        }
        MarginalLikelihoodStatus::Available
        | MarginalLikelihoodStatus::AvailableWithNonconvergedModes { .. } => {
            let (Some(n2ll), Some(mcse)) = (marginal.n2ll, marginal.n2ll_mcse) else {
                return unavailable(
                    InformationCriteriaUnavailableReason::ImpossibleMarginalLikelihoodState,
                    counts,
                    subject_count,
                );
            };
            if !n2ll.is_finite() {
                return unavailable(
                    InformationCriteriaUnavailableReason::NonFiniteMarginalN2ll,
                    counts,
                    subject_count,
                );
            }
            if !mcse.is_finite() {
                return unavailable(
                    InformationCriteriaUnavailableReason::NonFiniteMarginalN2llMcse,
                    counts,
                    subject_count,
                );
            }
            if mcse < 0.0 {
                return unavailable(
                    InformationCriteriaUnavailableReason::NegativeMarginalN2llMcse,
                    counts,
                    subject_count,
                );
            }
            let (aic, bic) =
                match calculate_criteria(n2ll, counts.total as f64, subject_count as f64) {
                    Ok(values) => values,
                    Err(reason) => return unavailable(reason, counts, subject_count),
                };
            let status = match &marginal.status {
                MarginalLikelihoodStatus::Available => InformationCriteriaStatus::Available,
                MarginalLikelihoodStatus::AvailableWithNonconvergedModes { subjects } => {
                    if subjects.is_empty() {
                        return unavailable(
                            InformationCriteriaUnavailableReason::ImpossibleMarginalLikelihoodState,
                            counts,
                            subject_count,
                        );
                    }
                    InformationCriteriaStatus::AvailableWithNonconvergedModes {
                        subjects: subjects.clone(),
                    }
                }
                MarginalLikelihoodStatus::Unavailable { .. } => unreachable!(),
            };
            InformationCriteriaDiagnostics {
                status,
                parameter_count: counts,
                sample_size_convention:
                    InformationCriteriaSampleSizeConvention::IndependentSubjects,
                subject_count,
                source_marginal_n2ll: Some(n2ll),
                source_marginal_n2ll_mcse: Some(mcse),
                aic: Some(aic),
                bic: Some(bic),
                aic_mcse: Some(mcse),
                bic_mcse: Some(mcse),
            }
        }
    }
}

fn calculate_criteria(
    n2ll: f64,
    parameter_count: f64,
    subject_count: f64,
) -> Result<(f64, f64), InformationCriteriaUnavailableReason> {
    let aic_penalty = 2.0 * parameter_count;
    let bic_penalty = subject_count.ln() * parameter_count;
    if !aic_penalty.is_finite() || !bic_penalty.is_finite() {
        return Err(InformationCriteriaUnavailableReason::NonFinitePenalty);
    }
    let aic = n2ll + aic_penalty;
    let bic = n2ll + bic_penalty;
    if !aic.is_finite() || !bic.is_finite() {
        return Err(InformationCriteriaUnavailableReason::NonFiniteCriterion);
    }
    Ok((aic, bic))
}

fn count_parameters(
    coordinates: &[InformationCoordinate],
) -> Result<InformationCriteriaParameterCount, InformationCriteriaUnavailableReason> {
    let mut population_sources = HashSet::new();
    let mut covariate_sources = HashSet::new();
    let mut omega_sources = HashSet::new();
    let mut omega_iov_sources = HashSet::new();
    let mut residual_sources = HashSet::new();
    let mut count = InformationCriteriaParameterCount::default();
    let mut previous_order_key = None;
    for (expected_index, coordinate) in coordinates.iter().enumerate() {
        if coordinate.index != expected_index {
            return Err(InformationCriteriaUnavailableReason::InconsistentCoordinateIndices);
        }
        let order_key = match &coordinate.kind {
            InformationCoordinateKind::Population { parameter_index } => {
                if !population_sources.insert(*parameter_index) {
                    return Err(InformationCriteriaUnavailableReason::DuplicateCoordinateSource);
                }
                count.population = count
                    .population
                    .checked_add(1)
                    .ok_or(InformationCriteriaUnavailableReason::ParameterCountOverflow)?;
                (0, *parameter_index, 0, String::new())
            }
            InformationCoordinateKind::CovariateEffect { effect_index } => {
                if !covariate_sources.insert(*effect_index) {
                    return Err(InformationCriteriaUnavailableReason::DuplicateCoordinateSource);
                }
                count.covariate = count
                    .covariate
                    .checked_add(1)
                    .ok_or(InformationCriteriaUnavailableReason::ParameterCountOverflow)?;
                (1, *effect_index, 0, String::new())
            }
            InformationCoordinateKind::Omega { row, column } => {
                if column > row {
                    return Err(
                        InformationCriteriaUnavailableReason::InvalidCovarianceCoordinate {
                            row: *row,
                            column: *column,
                        },
                    );
                }
                if !omega_sources.insert((*row, *column)) {
                    return Err(InformationCriteriaUnavailableReason::DuplicateCoordinateSource);
                }
                count.omega = count
                    .omega
                    .checked_add(1)
                    .ok_or(InformationCriteriaUnavailableReason::ParameterCountOverflow)?;
                (2, *row, *column, String::new())
            }
            InformationCoordinateKind::OmegaIov { row, column } => {
                if column > row {
                    return Err(
                        InformationCriteriaUnavailableReason::InvalidCovarianceCoordinate {
                            row: *row,
                            column: *column,
                        },
                    );
                }
                if !omega_iov_sources.insert((*row, *column)) {
                    return Err(InformationCriteriaUnavailableReason::DuplicateCoordinateSource);
                }
                count.omega_iov = count
                    .omega_iov
                    .checked_add(1)
                    .ok_or(InformationCriteriaUnavailableReason::ParameterCountOverflow)?;
                (3, *row, *column, String::new())
            }
            InformationCoordinateKind::Residual {
                output_index,
                component,
            } => {
                if !residual_sources.insert((*output_index, component.as_str())) {
                    return Err(InformationCriteriaUnavailableReason::DuplicateCoordinateSource);
                }
                count.residual = count
                    .residual
                    .checked_add(1)
                    .ok_or(InformationCriteriaUnavailableReason::ParameterCountOverflow)?;
                let component_order = match component.as_str() {
                    "sigma" | "additive" => 0,
                    "proportional" => 1,
                    "correlation" => 2,
                    _ => {
                        return Err(
                            InformationCriteriaUnavailableReason::UnknownResidualComponent {
                                component: component.clone(),
                            },
                        )
                    }
                };
                (4, *output_index, component_order, component.clone())
            }
        };
        if previous_order_key
            .as_ref()
            .is_some_and(|previous| previous >= &order_key)
        {
            return Err(InformationCriteriaUnavailableReason::NoncanonicalCoordinateOrder);
        }
        previous_order_key = Some(order_key);
    }
    count.total = count
        .population
        .checked_add(count.covariate)
        .and_then(|value| value.checked_add(count.omega))
        .and_then(|value| value.checked_add(count.omega_iov))
        .and_then(|value| value.checked_add(count.residual))
        .ok_or(InformationCriteriaUnavailableReason::ParameterCountOverflow)?;
    Ok(count)
}

fn unavailable(
    reason: InformationCriteriaUnavailableReason,
    parameter_count: InformationCriteriaParameterCount,
    subject_count: usize,
) -> InformationCriteriaDiagnostics {
    InformationCriteriaDiagnostics {
        status: InformationCriteriaStatus::Unavailable { reason },
        parameter_count,
        sample_size_convention: InformationCriteriaSampleSizeConvention::IndependentSubjects,
        subject_count,
        source_marginal_n2ll: None,
        source_marginal_n2ll_mcse: None,
        aic: None,
        bic: None,
        aic_mcse: None,
        bic_mcse: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parametric::marginal_likelihood::{
        MarginalLikelihoodConfig, MarginalLikelihoodFailureReason, MarginalLikelihoodSubjectFailure,
    };

    fn marginal(
        status: MarginalLikelihoodStatus,
        n2ll: Option<f64>,
        mcse: Option<f64>,
    ) -> MarginalLikelihoodDiagnostics {
        MarginalLikelihoodDiagnostics {
            config: MarginalLikelihoodConfig::new(32, 17, 5, 1.5),
            status,
            log_marginal_likelihood: n2ll.map(|value| -value / 2.0),
            n2ll,
            n2ll_mcse: mcse,
            subjects: Vec::new(),
        }
    }

    fn mixed_coordinates() -> Vec<InformationCoordinate> {
        vec![
            InformationCoordinate {
                index: 0,
                name: "phi:CL".into(),
                kind: InformationCoordinateKind::Population { parameter_index: 0 },
            },
            InformationCoordinate {
                index: 1,
                name: "beta:CL:WT".into(),
                kind: InformationCoordinateKind::CovariateEffect { effect_index: 0 },
            },
            InformationCoordinate {
                index: 2,
                name: "omega:CL:CL".into(),
                kind: InformationCoordinateKind::Omega { row: 0, column: 0 },
            },
            InformationCoordinate {
                index: 3,
                name: "omega_iov:V:V".into(),
                kind: InformationCoordinateKind::OmegaIov { row: 0, column: 0 },
            },
            InformationCoordinate {
                index: 4,
                name: "residual:central:sigma".into(),
                kind: InformationCoordinateKind::Residual {
                    output_index: 0,
                    component: "sigma".into(),
                },
            },
            InformationCoordinate {
                index: 5,
                name: "residual:peripheral:proportional".into(),
                kind: InformationCoordinateKind::Residual {
                    output_index: 1,
                    component: "proportional".into(),
                },
            },
        ]
    }

    #[test]
    fn information_criteria_formula_and_mcse_are_exact() {
        let source = marginal(
            MarginalLikelihoodStatus::Available,
            Some(100.0),
            Some(0.375),
        );
        let result = derive_information_criteria(Some(&source), &mixed_coordinates(), 20);
        assert_eq!(result.status, InformationCriteriaStatus::Available);
        assert_eq!(
            result.parameter_count,
            InformationCriteriaParameterCount {
                population: 1,
                covariate: 1,
                omega: 1,
                omega_iov: 1,
                residual: 2,
                total: 6,
            }
        );
        assert!((result.aic.unwrap() - 112.0).abs() <= 1e-12);
        assert!((result.bic.unwrap() - (100.0 + 6.0 * 20.0_f64.ln())).abs() <= 1e-12);
        assert_eq!(result.aic_mcse.unwrap().to_bits(), 0.375_f64.to_bits());
        assert_eq!(result.bic_mcse.unwrap().to_bits(), 0.375_f64.to_bits());
    }

    #[test]
    fn correlated_residual_coordinates_count_in_canonical_component_order() {
        let coordinates = ["additive", "proportional", "correlation"]
            .into_iter()
            .enumerate()
            .map(|(index, component)| InformationCoordinate {
                index,
                name: format!("residual:cp:{component}"),
                kind: InformationCoordinateKind::Residual {
                    output_index: 0,
                    component: component.to_string(),
                },
            })
            .collect::<Vec<_>>();
        let count = count_parameters(&coordinates).unwrap();
        assert_eq!(count.residual, 3);
        assert_eq!(count.total, 3);

        let mut malformed = coordinates;
        malformed[2].kind = InformationCoordinateKind::Residual {
            output_index: 0,
            component: "unknown".to_string(),
        };
        assert_eq!(
            count_parameters(&malformed),
            Err(
                InformationCriteriaUnavailableReason::UnknownResidualComponent {
                    component: "unknown".to_string()
                }
            )
        );
    }

    #[test]
    fn information_criteria_propagate_all_source_statuses_without_fallback() {
        let subjects = vec!["S2".to_string(), "S7".to_string()];
        let source = marginal(
            MarginalLikelihoodStatus::AvailableWithNonconvergedModes {
                subjects: subjects.clone(),
            },
            Some(42.0),
            Some(0.2),
        );
        let result = derive_information_criteria(Some(&source), &[], 2);
        assert_eq!(
            result.status,
            InformationCriteriaStatus::AvailableWithNonconvergedModes { subjects }
        );
        assert_eq!(result.aic, Some(42.0));
        assert_eq!(result.bic, Some(42.0));

        let unavailable_source = marginal(
            MarginalLikelihoodStatus::Unavailable {
                failures: vec![MarginalLikelihoodSubjectFailure {
                    subject_id: "S1".into(),
                    reason: MarginalLikelihoodFailureReason::AllZeroEffectiveWeights,
                }],
            },
            None,
            None,
        );
        let result = derive_information_criteria(Some(&unavailable_source), &[], 1);
        assert!(matches!(
            result.status,
            InformationCriteriaStatus::Unavailable {
                reason: InformationCriteriaUnavailableReason::SourceMarginalLikelihoodUnavailable
            }
        ));
        assert_eq!((result.aic, result.bic), (None, None));

        let result = derive_information_criteria(None, &[], 1);
        assert_eq!(result.status, InformationCriteriaStatus::NotRequested);
        assert_eq!((result.aic, result.bic), (None, None));
    }

    #[test]
    fn information_criteria_accept_zero_parameters_and_one_subject() {
        let source = marginal(MarginalLikelihoodStatus::Available, Some(12.5), Some(0.0));
        let result = derive_information_criteria(Some(&source), &[], 1);
        assert_eq!(result.parameter_count.total, 0);
        assert_eq!(result.aic, Some(12.5));
        assert_eq!(result.bic, Some(12.5));
    }

    #[test]
    fn information_criteria_fail_closed_for_invalid_inputs() {
        let source = marginal(MarginalLikelihoodStatus::Available, Some(12.5), Some(0.1));
        let assert_unavailable_without_values = |result: InformationCriteriaDiagnostics| {
            assert!(matches!(
                result.status,
                InformationCriteriaStatus::Unavailable { .. }
            ));
            assert_eq!(
                (result.aic, result.bic, result.aic_mcse, result.bic_mcse),
                (None, None, None, None)
            );
        };
        assert_unavailable_without_values(derive_information_criteria(Some(&source), &[], 0));
        assert_unavailable_without_values(derive_information_criteria(
            Some(&source),
            &[],
            MAX_EXACT_INTEGER_F64 + 1,
        ));

        let mut coordinates = mixed_coordinates();
        coordinates[5].index = 4;
        assert_unavailable_without_values(derive_information_criteria(
            Some(&source),
            &coordinates,
            2,
        ));

        for invalid_source in [
            marginal(
                MarginalLikelihoodStatus::Available,
                Some(f64::NAN),
                Some(0.1),
            ),
            marginal(
                MarginalLikelihoodStatus::Available,
                Some(12.5),
                Some(f64::INFINITY),
            ),
            marginal(MarginalLikelihoodStatus::Available, Some(12.5), Some(-0.1)),
            marginal(
                MarginalLikelihoodStatus::AvailableWithNonconvergedModes { subjects: vec![] },
                Some(12.5),
                Some(0.1),
            ),
            marginal(
                MarginalLikelihoodStatus::Unavailable { failures: vec![] },
                None,
                None,
            ),
        ] {
            assert_unavailable_without_values(derive_information_criteria(
                Some(&invalid_source),
                &[],
                1,
            ));
        }

        assert_eq!(
            calculate_criteria(0.0, f64::MAX, 2.0),
            Err(InformationCriteriaUnavailableReason::NonFinitePenalty)
        );
        assert_eq!(
            calculate_criteria(f64::MAX, f64::MAX / 4.0, 1.0),
            Err(InformationCriteriaUnavailableReason::NonFiniteCriterion)
        );

        let mut upper_triangle = mixed_coordinates();
        upper_triangle[2].kind = InformationCoordinateKind::Omega { row: 0, column: 1 };
        assert_unavailable_without_values(derive_information_criteria(
            Some(&source),
            &upper_triangle,
            2,
        ));

        for duplicate_kind in [
            InformationCoordinateKind::Population { parameter_index: 0 },
            InformationCoordinateKind::CovariateEffect { effect_index: 0 },
            InformationCoordinateKind::Omega { row: 0, column: 0 },
            InformationCoordinateKind::OmegaIov { row: 0, column: 0 },
            InformationCoordinateKind::Residual {
                output_index: 0,
                component: "sigma".into(),
            },
        ] {
            let duplicate_sources = vec![
                InformationCoordinate {
                    index: 0,
                    name: "first".into(),
                    kind: duplicate_kind.clone(),
                },
                InformationCoordinate {
                    index: 1,
                    name: "second".into(),
                    kind: duplicate_kind,
                },
            ];
            assert_unavailable_without_values(derive_information_criteria(
                Some(&source),
                &duplicate_sources,
                2,
            ));
        }

        let mut reordered = mixed_coordinates();
        reordered.swap(0, 1);
        for (index, coordinate) in reordered.iter_mut().enumerate() {
            coordinate.index = index;
        }
        assert_unavailable_without_values(derive_information_criteria(
            Some(&source),
            &reordered,
            2,
        ));

        let impossible = marginal(MarginalLikelihoodStatus::Available, None, None);
        assert_unavailable_without_values(derive_information_criteria(Some(&impossible), &[], 1));
    }
}
