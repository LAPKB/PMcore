use anyhow::Result;
use faer::Col;
use pharmsol::{Data, Equation, Event, ResidualErrorModels};

use crate::estimation::parametric::{
    importance_sampling_likelihood_estimates, subject_conditionals_from_eta_samples,
    ImportanceSamplingConfig, IndividualEstimates, LikelihoodEstimates, ParameterTransform,
    ParametricPredictions, ParametricStatistics, ParametricWorkspace, Population,
};

pub fn cache_predictions<E: Equation>(
    result: &mut ParametricWorkspace<E>,
    idelta: f64,
    tad: f64,
) -> Result<()> {
    let sigma_val = result.sigma().additive.or(result.sigma().proportional);
    let predictions = ParametricPredictions::calculate(
        result.equation(),
        result.data(),
        result.population(),
        result.individual_estimates(),
        sigma_val,
        idelta,
        tad,
    )?;
    result.set_predictions(predictions);
    Ok(())
}

pub fn statistics<E: Equation>(result: &ParametricWorkspace<E>) -> ParametricStatistics {
    let n_observations = result
        .data()
        .subjects()
        .iter()
        .flat_map(|subject| subject.occasions())
        .flat_map(|occasion| occasion.events())
        .filter(|event| matches!(event, Event::Observation(_)))
        .count();

    ParametricStatistics::from_result(
        result.population(),
        result.individual_estimates(),
        result.objf(),
        result.iterations(),
        result.converged(),
        result.data().len(),
        n_observations,
        result.likelihoods().ll_importance_sampling,
        result.likelihoods().ll_linearization,
        result.likelihoods().ll_gaussian_quadrature,
        result.sigma().as_vec(),
    )
}

pub fn write_statistics<E: Equation>(result: &ParametricWorkspace<E>) -> Result<()> {
    let stats = statistics(result);
    stats.write(result.output_folder())?;
    stats.write_shrinkage(result.output_folder(), &result.population().param_names())?;
    Ok(())
}

pub(crate) fn saem_posthoc_likelihood<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &ResidualErrorModels,
    transforms: &[ParameterTransform],
    population: &Population,
    individual_estimates: &IndividualEstimates,
    eta_samples_by_subject: &[Vec<Col<f64>>],
    seed: u64,
) -> Result<(LikelihoodEstimates, f64)> {
    let conditionals = subject_conditionals_from_eta_samples(
        eta_samples_by_subject,
        Some(individual_estimates),
        population.mu(),
        population.omega(),
    );
    let likelihood_estimates = importance_sampling_likelihood_estimates(
        equation,
        data.subjects().iter().copied().collect(),
        error_models,
        transforms,
        population.mu(),
        population.omega(),
        &conditionals,
        ImportanceSamplingConfig::saemix_defaults()
            .with_n_samples(10000)
            .with_seed(seed + 12345),
    )?;
    let minus2ll = likelihood_estimates
        .best_objf()
        .unwrap_or(f64::NEG_INFINITY);

    Ok((likelihood_estimates, minus2ll))
}

pub fn shrinkage<E: Equation>(result: &ParametricWorkspace<E>) -> Option<Col<f64>> {
    let n = result.population().npar();
    let pop_var = Col::from_fn(n, |index| result.population().omega()[(index, index)]);
    result.individual_estimates().shrinkage(&pop_var)
}

pub fn aic<E: Equation>(result: &ParametricWorkspace<E>) -> f64 {
    let n_params = result.population().npar();
    let n_fixed = n_params;
    let n_random = n_params * (n_params + 1) / 2;
    let k = n_fixed + n_random;
    result.best_objf() + 2.0 * k as f64
}

pub fn bic<E: Equation>(result: &ParametricWorkspace<E>) -> f64 {
    let n_subjects = result.data().subjects().len();
    let n_params = result.population().npar();
    let n_fixed = n_params;
    let n_random = n_params * (n_params + 1) / 2;
    let k = n_fixed + n_random;
    result.best_objf() + (k as f64) * (n_subjects as f64).ln()
}