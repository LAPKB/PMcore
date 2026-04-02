use pharmsol::{Data, Equation, Event};

use crate::estimation::parametric::ParametricWorkspace;
use crate::results::{FitSummary, IndividualSummary, ParameterSummary, PopulationSummary};

pub fn fit_summary<E: Equation>(result: &ParametricWorkspace<E>) -> FitSummary {
    FitSummary {
        objective_function: result.objf(),
        converged: result.converged(),
        iterations: result.iterations(),
        subject_count: result.data().subjects().len(),
        observation_count: count_observations(result.data()),
        parameter_count: result.population().npar(),
        algorithm: format!("{:?}", result.algorithm()),
    }
}

pub fn population_summary<E: Equation>(result: &ParametricWorkspace<E>) -> PopulationSummary {
    let names = result.population().param_names();
    let sds = result.standard_deviations();
    let cvs = result.cv_percent();

    let parameters = names
        .into_iter()
        .enumerate()
        .map(|(index, name)| ParameterSummary {
            name,
            mean: result.mu()[index],
            median: result.mu()[index],
            sd: sds[index],
            cv_percent: cvs[index],
        })
        .collect();

    PopulationSummary { parameters }
}

pub fn individual_summaries<E: Equation>(
    result: &ParametricWorkspace<E>,
) -> Vec<IndividualSummary> {
    let parameter_names = result.population().param_names();

    result
        .individual_estimates()
        .iter()
        .map(|individual| IndividualSummary {
            id: individual.subject_id().to_string(),
            parameter_names: parameter_names.clone(),
            estimates: individual.psi().iter().copied().collect(),
            standard_errors: individual
                .standard_errors()
                .map(|errors| errors.iter().copied().collect()),
        })
        .collect()
}

fn count_observations(data: &Data) -> usize {
    data.subjects()
        .iter()
        .flat_map(|subject| subject.occasions())
        .flat_map(|occasion| occasion.events())
        .filter(|event| matches!(event, Event::Observation(_)))
        .count()
}
