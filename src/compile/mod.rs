use anyhow::Result;
use pharmsol::{Data, Equation, Event};

use crate::api::EstimationProblem;
use crate::model::{CovariateSpec, EquationMetadataSource, ParameterSpace};

mod caches;
mod compiled_problem;
mod design_context;
mod observation_index;
mod validation;

pub use caches::ExecutionCaches;
pub use compiled_problem::CompiledProblem;
pub use design_context::{
    DesignContext, OccasionCovariateRow, OccasionDesign, StructuredCovariateDesign,
    SubjectCovariateRow, SubjectDesign,
};
pub use observation_index::{ObservationIndex, ObservationRecord};
pub use validation::validate_problem;

impl<E: Equation + Clone + EquationMetadataSource> EstimationProblem<E> {
    pub fn compile(self) -> Result<CompiledProblem<E>> {
        compile_problem(self)
    }

    pub fn initialize_logs(&self) -> Result<()> {
        crate::output::logging::setup_log_with_options(&self.output, &self.runtime.logging)
    }
}

pub fn compile_problem<E: Equation + Clone + EquationMetadataSource>(
    problem: EstimationProblem<E>,
) -> Result<CompiledProblem<E>> {
    validate_problem(&problem)?;

    let design = build_design_context(
        &problem.model.parameters,
        &problem.model.covariates,
        &problem.data,
    );
    let observation_index = build_observation_index(&problem.data)?;
    let caches = ExecutionCaches {
        prediction_cache_enabled: problem.runtime.cache,
    };

    Ok(CompiledProblem::new(
        problem.model,
        problem.data,
        problem.error_models,
        problem.method,
        problem.output,
        problem.runtime,
        design,
        observation_index,
        caches,
    ))
}

fn build_design_context(
    parameter_space: &ParameterSpace,
    covariates: &CovariateSpec,
    data: &Data,
) -> DesignContext {
    let subjects = data.subjects();

    let subject_design = subjects
        .iter()
        .enumerate()
        .map(|(subject_index, subject)| {
            let occasions = subject.occasions();
            let observation_count = occasions
                .iter()
                .map(|occasion| {
                    occasion
                        .events()
                        .iter()
                        .filter(|event| matches!(event, Event::Observation(_)))
                        .count()
                })
                .sum();

            SubjectDesign {
                subject_index,
                id: subject.id().clone(),
                occasion_count: occasions.len(),
                observation_count,
            }
        })
        .collect::<Vec<_>>();

    let occasion_design = subjects
        .iter()
        .enumerate()
        .flat_map(|(subject_index, subject)| {
            subject.occasions().iter().map(move |occasion| {
                let events = occasion.events();
                let observation_count = events
                    .iter()
                    .filter(|event| matches!(event, Event::Observation(_)))
                    .count();

                OccasionDesign {
                    subject_index,
                    occasion_index: occasion.index(),
                    event_count: events.len(),
                    observation_count,
                }
            })
        })
        .collect::<Vec<_>>();

    let structured_covariates = match covariates {
        CovariateSpec::InEquation => StructuredCovariateDesign::default(),
        CovariateSpec::Structured(spec) => build_structured_covariate_design(
            &spec.subject_columns(),
            &spec.occasion_columns(),
            data,
        ),
    };

    DesignContext {
        parameter_names: parameter_space
            .iter()
            .map(|item| item.name.clone())
            .collect(),
        subjects: subject_design,
        occasions: occasion_design,
        structured_covariates,
    }
}

fn build_structured_covariate_design(
    subject_columns: &[String],
    occasion_columns: &[String],
    data: &Data,
) -> StructuredCovariateDesign {
    let subject_rows = data
        .subjects()
        .iter()
        .enumerate()
        .map(|(subject_index, subject)| {
            let anchor_time = subject_anchor_time(subject);
            let values = subject_columns
                .iter()
                .map(|name| subject_covariate_value(subject, name))
                .collect();

            SubjectCovariateRow {
                subject_index,
                id: subject.id().clone(),
                anchor_time,
                values,
            }
        })
        .collect();

    let occasion_rows = data
        .subjects()
        .iter()
        .enumerate()
        .flat_map(|(subject_index, subject)| {
            subject.occasions().iter().map(move |occasion| {
                let anchor_time = occasion_anchor_time(occasion);
                let values = occasion_columns
                    .iter()
                    .map(|name| {
                        occasion
                            .covariates()
                            .get_covariate(name)
                            .and_then(|covariate| covariate.interpolate(anchor_time).ok())
                    })
                    .collect();

                OccasionCovariateRow {
                    subject_index,
                    occasion_index: occasion.index(),
                    anchor_time,
                    values,
                }
            })
        })
        .collect();

    StructuredCovariateDesign {
        subject_columns: subject_columns.to_vec(),
        subject_rows,
        occasion_columns: occasion_columns.to_vec(),
        occasion_rows,
    }
}

fn subject_anchor_time(subject: &pharmsol::Subject) -> f64 {
    subject
        .occasions()
        .iter()
        .find_map(|occasion| occasion.events().first().map(|event| event.time()))
        .unwrap_or(0.0)
}

fn subject_covariate_value(subject: &pharmsol::Subject, name: &str) -> Option<f64> {
    subject.occasions().iter().find_map(|occasion| {
        let anchor_time = occasion_anchor_time(occasion);
        occasion
            .covariates()
            .get_covariate(name)
            .and_then(|covariate| covariate.interpolate(anchor_time).ok())
    })
}

fn occasion_anchor_time(occasion: &pharmsol::Occasion) -> f64 {
    occasion
        .events()
        .first()
        .map(|event| event.time())
        .unwrap_or(0.0)
}

fn build_observation_index(data: &Data) -> Result<ObservationIndex> {
    let records =
        data.subjects()
            .iter()
            .enumerate()
            .flat_map(|(subject_index, subject)| {
                subject.occasions().iter().flat_map(move |occasion| {
                    occasion
                        .events()
                        .iter()
                        .enumerate()
                        .filter_map(move |(event_index, event)| match event {
                            Event::Observation(observation) => Some(
                                observation
                                    .outeq_index()
                                    .map(|outeq| ObservationRecord {
                                        subject_index,
                                        occasion_index: occasion.index(),
                                        event_index,
                                        outeq,
                                        time: observation.time(),
                                    })
                                    .ok_or_else(|| {
                                        anyhow::anyhow!(
                                            "Compilation requires numeric observation output labels; got `{}`",
                                            observation.outeq()
                                        )
                                    }),
                            ),
                            _ => None,
                        })
                })
            })
            .collect::<Result<Vec<_>>>()?;

    Ok(ObservationIndex { records })
}
