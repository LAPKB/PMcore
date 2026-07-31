use ndarray::{Array1, Array2};
use pharmsol::{Data, Equation, Event};

use crate::estimation::nonparametric::NonParametricResult;
use crate::estimation::nonparametric::{population_mean_median, posterior_mean_median};
use crate::results::{FitSummary, IndividualSummary, ParameterSummary, PopulationSummary};

pub fn fit_summary<E: Equation>(result: &NonParametricResult<E>) -> FitSummary {
    FitSummary {
        objective_function: result.objf(),
        converged: result.converged(),
        iterations: result.cycles(),
        subject_count: result.data().subjects().len(),
        observation_count: count_observations(result.data()),
        parameter_count: result.get_theta().parameters().len(),
        marginal_log_likelihood: None,
        marginal_n2ll: None,
        marginal_n2ll_mcse: None,
        marginal_likelihood_status: None,
        information_criteria: None,
    }
}

pub fn population_summary<E: Equation>(result: &NonParametricResult<E>) -> PopulationSummary {
    let theta_matrix = to_ndarray_matrix(result.get_theta().matrix());
    let weights = Array1::from_iter(result.weights().iter());
    let (mean, median) = population_mean_median(&theta_matrix, &weights)
        .expect("population summary should be derivable from theta and weights");

    let parameters = result
        .get_theta()
        .parameters()
        .names()
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let column = theta_matrix.column(index).to_vec();
            let mean_value = mean[index];
            let sd = weighted_sd(&column, &weights, mean_value);
            let cv_percent =
                (mean_value.abs() > f64::EPSILON).then_some((sd / mean_value.abs()) * 100.0);

            ParameterSummary {
                name,
                estimate: mean_value,
                mean: Some(mean_value),
                median: Some(median[index]),
                sd: Some(sd),
                cv_percent,
            }
        })
        .collect();

    PopulationSummary {
        parameters,
        information_criteria: None,
        population_uncertainty: None,
        shrinkage: None,
    }
}

pub fn individual_summaries<E: Equation>(
    result: &NonParametricResult<E>,
) -> Vec<IndividualSummary> {
    let theta_matrix = to_ndarray_matrix(result.get_theta().matrix());
    let psi_matrix = to_ndarray_matrix(result.psi().matrix());
    let weights = Array1::from_iter(result.weights().iter());
    let (means, _) = posterior_mean_median(&theta_matrix, &psi_matrix, &weights)
        .expect("individual summaries should be derivable from theta, psi, and weights");
    let parameter_names = result.get_theta().parameters().names();

    result
        .data()
        .subjects()
        .iter()
        .enumerate()
        .map(|(subject_index, subject)| IndividualSummary {
            id: subject.id().clone(),
            parameter_names: parameter_names.clone(),
            estimates: means.row(subject_index).to_vec(),
            standard_errors: None,
            conditional_uncertainty: None,
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

fn to_ndarray_matrix(matrix: &faer::Mat<f64>) -> Array2<f64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(row, col)| {
        matrix[(row, col)]
    })
}

fn weighted_sd(values: &[f64], weights: &Array1<f64>, mean: f64) -> f64 {
    let variance = values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| weight * (value - mean).powi(2))
        .sum::<f64>();
    variance.sqrt()
}
