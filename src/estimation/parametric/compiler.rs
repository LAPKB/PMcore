use pharmsol::Equation;

use crate::compile::CompiledProblem;
use crate::estimation::parametric::state::{
    CovariateEffectsSnapshot, CovariateState, FixedEffects, ParametricModelState,
    ParametricTransformKind, PsiVector, RandomEffects, ResidualState, TransformSet,
};
use crate::model::{
    CovariateSpec, ParameterDomain, ParameterSpace, ParameterVariability, RandomEffectsSpec,
    VariabilityModel,
};

pub fn compile_model_state<E: Equation>(problem: &CompiledProblem<E>) -> ParametricModelState {
    let parameter_names = problem
        .model
        .parameters
        .iter()
        .map(|parameter| parameter.name.clone())
        .collect::<Vec<_>>();

    let initial_values = problem
        .model
        .parameters
        .iter()
        .map(initial_value)
        .collect::<Vec<_>>();

    let transforms = problem
        .model
        .parameters
        .iter()
        .map(|parameter| ParametricTransformKind::from(&parameter.transform))
        .collect::<Vec<_>>();

    let n_parameters = parameter_names.len();
    let covariance = identity_matrix(n_parameters);
    let standard_deviations = vec![1.0; n_parameters];
    let variability = resolve_variability_model(problem);

    let covariates = match &problem.model.covariates {
        CovariateSpec::InEquation => CovariateState {
            subject_effects: None,
            occasion_effects: None,
        },
        CovariateSpec::Structured(spec) => CovariateState {
            subject_effects: spec.subject_effects.as_ref().map(|model| {
                CovariateEffectsSnapshot::from_model(
                    model,
                    problem
                        .design
                        .structured_covariates
                        .subject_rows
                        .iter()
                        .map(|row| row.values.clone())
                        .collect(),
                )
            }),
            occasion_effects: spec.occasion_effects.as_ref().map(|model| {
                CovariateEffectsSnapshot::from_model(
                    model,
                    problem
                        .design
                        .structured_covariates
                        .occasion_rows
                        .iter()
                        .map(|row| row.values.clone())
                        .collect(),
                )
            }),
        },
    };

    ParametricModelState {
        fixed_effects: FixedEffects {
            parameter_names,
            population_mean: PsiVector(initial_values),
        },
        random_effects: RandomEffects {
            covariance: covariance.clone(),
            standard_deviations,
            correlation: covariance,
        },
        residual: ResidualState { values: Vec::new() },
        transforms: TransformSet { transforms },
        covariates,
        variability,
    }
}

fn resolve_variability_model<E: Equation>(problem: &CompiledProblem<E>) -> VariabilityModel {
    let n_parameters = problem.model.parameters.len();
    let derived_subject = derived_subject_mask(&problem.model.parameters);
    let derived_occasion = derived_occasion_mask(&problem.model.parameters);

    let mut subject = problem.model.variability.subject.clone();
    if subject.enabled_for.len() != n_parameters {
        subject.enabled_for = derived_subject;
    }

    let occasion = match &problem.model.variability.occasion {
        Some(spec) => {
            let mut spec = spec.clone();
            if spec.enabled_for.len() != n_parameters {
                spec.enabled_for = derived_occasion.clone();
            }
            Some(spec)
        }
        None if derived_occasion.iter().any(|enabled| *enabled) => Some(RandomEffectsSpec {
            enabled_for: derived_occasion,
            covariance: subject.covariance.clone(),
        }),
        None => None,
    };

    VariabilityModel { subject, occasion }
}

fn derived_subject_mask(parameter_space: &ParameterSpace) -> Vec<bool> {
    parameter_space
        .iter()
        .map(|parameter| {
            matches!(
                parameter.variability,
                ParameterVariability::Subject | ParameterVariability::SubjectAndOccasion
            )
        })
        .collect()
}

fn derived_occasion_mask(parameter_space: &ParameterSpace) -> Vec<bool> {
    parameter_space
        .iter()
        .map(|parameter| {
            matches!(
                parameter.variability,
                ParameterVariability::Occasion | ParameterVariability::SubjectAndOccasion
            )
        })
        .collect()
}

fn initial_value(parameter: &crate::model::ParameterSpec) -> f64 {
    if let Some(initial) = parameter.initial {
        return initial;
    }

    match parameter.domain {
        ParameterDomain::Bounded { lower, upper } => (lower + upper) / 2.0,
        ParameterDomain::Positive { lower, upper } => match (lower, upper) {
            (Some(lower), Some(upper)) => (lower + upper) / 2.0,
            (Some(lower), None) => lower.max(1.0),
            (None, Some(upper)) => upper / 2.0,
            (None, None) => 1.0,
        },
        ParameterDomain::Unbounded { lower, upper } => match (lower, upper) {
            (Some(lower), Some(upper)) => (lower + upper) / 2.0,
            _ => 0.0,
        },
    }
}

fn identity_matrix(size: usize) -> Vec<Vec<f64>> {
    (0..size)
        .map(|row| {
            (0..size)
                .map(|col| if row == col { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}