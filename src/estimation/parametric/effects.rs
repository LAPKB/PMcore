use std::collections::HashMap;

use anyhow::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Col, Mat};

use crate::compile::StructuredCovariateDesign;
use crate::estimation::parametric::{IndividualEstimates, Population};
use crate::model::CovariateModel;

use super::state::{CovariateEffectsSnapshot, CovariateState};

pub(crate) fn recenter_individual_estimates(
	individual_estimates: &IndividualEstimates,
	subject_means: &[Col<f64>],
) -> Result<IndividualEstimates> {
	let n_params = individual_estimates
		.get(0)
		.map(|individual| individual.npar())
		.unwrap_or(0);
	let mut recentered = Vec::with_capacity(individual_estimates.nsubjects());

	for (subject_index, individual) in individual_estimates.iter().enumerate() {
		let eta = Col::from_fn(n_params, |param_index| {
			individual.psi()[param_index] - subject_means[subject_index][param_index]
		});
		let mut rebuilt = crate::estimation::parametric::Individual::new(
			individual.subject_id().to_string(),
			eta,
			individual.psi().clone(),
		)?;
		if let Some(objf) = individual.objective_function() {
			rebuilt.set_objective_function(objf);
		}
		recentered.push(rebuilt);
	}

	Ok(IndividualEstimates::from_vec(recentered))
}

pub(crate) fn covariance_from_individual_etas(individual_estimates: &IndividualEstimates) -> Mat<f64> {
	let n_subjects = individual_estimates.nsubjects();
	let n_params = individual_estimates
		.get(0)
		.map(|individual| individual.npar())
		.unwrap_or(0);

	if n_subjects == 0 || n_params == 0 {
		return Mat::zeros(n_params, n_params);
	}

	let denom = n_subjects.max(1) as f64;
	Mat::from_fn(n_params, n_params, |row, col| {
		individual_estimates
			.iter()
			.map(|individual| individual.eta()[row] * individual.eta()[col])
			.sum::<f64>()
			/ denom
	})
}

pub(crate) fn subject_covariate_maps(
	structured_covariates: &StructuredCovariateDesign,
) -> Vec<HashMap<String, f64>> {
	covariate_maps(
		&structured_covariates.subject_columns,
		structured_covariates
			.subject_rows
			.iter()
			.map(|row| row.values.as_slice()),
	)
}

pub(crate) fn occasion_covariate_maps(
	structured_covariates: &StructuredCovariateDesign,
) -> Vec<HashMap<String, f64>> {
	covariate_maps(
		&structured_covariates.occasion_columns,
		structured_covariates
			.occasion_rows
			.iter()
			.map(|row| row.values.as_slice()),
	)
}

pub(crate) fn subject_mean_phi(
	population_mean: &Col<f64>,
	n_subjects: usize,
	model: Option<&CovariateModel>,
	subject_covariates: &[HashMap<String, f64>],
) -> Vec<Col<f64>> {
	match model {
		Some(model) => (0..n_subjects)
			.map(|subject_index| {
				let empty = HashMap::new();
				let covariates = subject_covariates.get(subject_index).unwrap_or(&empty);
				model.compute_mu(covariates)
			})
			.collect(),
		None => (0..n_subjects).map(|_| population_mean.clone()).collect(),
	}
}

pub(crate) fn covariate_state(
	subject_model: Option<&CovariateModel>,
	subject_covariates: &[HashMap<String, f64>],
	occasion_model: Option<&CovariateModel>,
	occasion_covariates: &[HashMap<String, f64>],
) -> CovariateState {
	CovariateState {
		subject_effects: covariate_snapshot(subject_model, subject_covariates),
		occasion_effects: covariate_snapshot(occasion_model, occasion_covariates),
	}
}

fn covariate_snapshot(
	model: Option<&CovariateModel>,
	covariates: &[HashMap<String, f64>],
) -> Option<CovariateEffectsSnapshot> {
	model.map(|model| {
		CovariateEffectsSnapshot::from_model(
			model,
			covariates
				.iter()
				.map(|row| {
					model
						.covariate_names()
						.iter()
						.map(|name| row.get(name).copied())
						.collect()
				})
				.collect(),
		)
	})
}

fn covariate_maps<'a>(
	columns: &[String],
	rows: impl Iterator<Item = &'a [Option<f64>]>,
) -> Vec<HashMap<String, f64>> {
	rows.map(|row| {
		columns
			.iter()
			.cloned()
			.zip(row.iter().copied())
			.filter_map(|(name, value)| value.map(|value| (name, value)))
			.collect()
	})
	.collect()
}

pub(crate) fn estimate_beta(
	model: &CovariateModel,
	subject_covariates: &[HashMap<String, f64>],
	individual_estimates: &IndividualEstimates,
) -> Result<Col<f64>> {
	let n_subjects = individual_estimates.nsubjects();
	let n_params = model.n_params();
	let design = model.build_design_matrix(subject_covariates);
	let responses = Col::from_fn(n_subjects * n_params, |row_index| {
		let subject_index = row_index / n_params;
		let param_index = row_index % n_params;
		individual_estimates.get(subject_index).unwrap().psi()[param_index]
	});

	let estimated_indices = model.estimated_beta_indices();
	if estimated_indices.is_empty() {
		return Ok(model.beta().clone());
	}

	let fixed_indices = (0..model.beta().nrows())
		.filter(|index| !estimated_indices.contains(index))
		.collect::<Vec<_>>();
	let mut normal_matrix = Mat::<f64>::zeros(estimated_indices.len(), estimated_indices.len());
	let mut normal_rhs = Col::<f64>::zeros(estimated_indices.len());

	for row_index in 0..design.nrows() {
		let offset = fixed_indices.iter().fold(0.0, |acc, fixed_index| {
			acc + design[(row_index, *fixed_index)] * model.beta()[*fixed_index]
		});
		let adjusted_response = responses[row_index] - offset;

		for (lhs_position, lhs_index) in estimated_indices.iter().enumerate() {
			let lhs_value = design[(row_index, *lhs_index)];
			normal_rhs[lhs_position] += lhs_value * adjusted_response;

			for (rhs_position, rhs_index) in estimated_indices.iter().enumerate() {
				normal_matrix[(lhs_position, rhs_position)] +=
					lhs_value * design[(row_index, *rhs_index)];
			}
		}
	}

	for diagonal in 0..estimated_indices.len() {
		normal_matrix[(diagonal, diagonal)] += 1e-8;
	}

	let solver = normal_matrix
		.llt(faer::Side::Lower)
		.map_err(|_| anyhow::anyhow!("covariate normal equations are singular"))?;
	let inverse = solver.inverse();
	let estimated_beta = Col::from_fn(estimated_indices.len(), |row_index| {
		(0..estimated_indices.len())
			.map(|col_index| inverse[(row_index, col_index)] * normal_rhs[col_index])
			.sum()
	});

	let mut beta = model.beta().clone();
	for (position, beta_index) in estimated_indices.iter().enumerate() {
		beta[*beta_index] = estimated_beta[position];
	}

	Ok(beta)
}

pub(crate) fn blended_subject_covariate_m_step(
	model: &CovariateModel,
	subject_covariates: &[HashMap<String, f64>],
	individual_estimates: &IndividualEstimates,
	population: &Population,
	step_size: f64,
) -> Result<(CovariateModel, Vec<Col<f64>>, Col<f64>, Mat<f64>)> {
	let target_beta = estimate_beta(model, subject_covariates, individual_estimates)?;
	let current_beta = model.beta().clone();
	let updated_beta = Col::from_fn(target_beta.nrows(), |index| {
		current_beta[index] + step_size * (target_beta[index] - current_beta[index])
	});
	let mut updated_model = model.clone();
	updated_model.set_beta(updated_beta)?;

	let subject_means = subject_covariates
		.iter()
		.map(|covariates| updated_model.compute_mu(covariates))
		.collect::<Vec<_>>();
	let mu = Col::from_fn(population.npar(), |index| {
		updated_model
			.intercept(index)
			.unwrap_or(population.mu()[index])
	});
	let omega = covariance_from_subject_means(individual_estimates, &subject_means)?;

	Ok((updated_model, subject_means, mu, omega))
}

pub(crate) fn covariance_from_subject_means(
	individual_estimates: &IndividualEstimates,
	subject_means: &[Col<f64>],
) -> Result<Mat<f64>> {
	let n_subjects = individual_estimates.nsubjects();
	let n_params = individual_estimates
		.get(0)
		.map(|individual| individual.npar())
		.unwrap_or(0);

	if n_subjects == 0 || n_params == 0 {
		return Ok(Mat::zeros(n_params, n_params));
	}

	let mut omega = Mat::<f64>::zeros(n_params, n_params);
	for subject_index in 0..n_subjects {
		let phi = individual_estimates.get(subject_index).unwrap().psi();
		for row in 0..n_params {
			let eta_row = phi[row] - subject_means[subject_index][row];
			for col in 0..n_params {
				let eta_col = phi[col] - subject_means[subject_index][col];
				omega[(row, col)] += eta_row * eta_col;
			}
		}
	}

	let denom = n_subjects as f64;
	Ok(Mat::from_fn(n_params, n_params, |row, col| omega[(row, col)] / denom))
}