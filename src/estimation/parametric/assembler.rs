use anyhow::Result;
use pharmsol::{Data, Equation};

use crate::algorithms::Status;
use crate::estimation::parametric::{
    phi_to_psi_vec, ChainState, CovariateState, Individual, IndividualEffectsState,
    IndividualEstimates, LikelihoodEstimates, ParameterTransform, ParametricIterationLog,
    ParametricModelState, ParametricWorkspace, Population, ResidualErrorEstimates,
    UncertaintyEstimates,
};
use crate::output::shared::RunConfiguration;

use super::posthoc::{eta_samples_by_subject, saem_posthoc_likelihood};

pub(crate) struct ParametricResultInput<'a, E: Equation> {
    pub equation: &'a E,
    pub data: &'a Data,
    pub population: &'a Population,
    pub individual_estimates: &'a IndividualEstimates,
    pub objf: f64,
    pub iterations: usize,
    pub status: &'a Status,
    pub run_configuration: RunConfiguration,
    pub iteration_log: ParametricIterationLog,
    pub likelihood_estimates: LikelihoodEstimates,
    pub uncertainty_estimates: UncertaintyEstimates,
    pub sigma: ResidualErrorEstimates,
    pub transforms: &'a [ParameterTransform],
    pub covariates: Option<CovariateState>,
}

pub(crate) struct SaemFinalizeInput<'a> {
    pub chain_states: &'a [Vec<ChainState>],
    pub residual_error_models: &'a pharmsol::ResidualErrorModels,
    pub seed: u64,
}

pub(crate) fn assemble_parametric_result<E: Equation + Clone>(
    input: ParametricResultInput<'_, E>,
) -> Result<ParametricWorkspace<E>> {
    assemble_parametric_workspace(ParametricWorkspaceInput {
        equation: input.equation,
        data: input.data,
        population: input.population,
        individual_estimates: input.individual_estimates,
        objf: input.objf,
        iterations: input.iterations,
        status: input.status,
        run_configuration: input.run_configuration,
        iteration_log: input.iteration_log,
        likelihood_estimates: input.likelihood_estimates,
        uncertainty_estimates: input.uncertainty_estimates,
        sigma: input.sigma,
        transforms: input.transforms,
        covariates: input.covariates,
    })
}

pub(crate) fn finalize_saem_result<E: Equation + Clone>(
    input: ParametricResultInput<'_, E>,
    finalize: SaemFinalizeInput<'_>,
) -> Result<ParametricWorkspace<E>> {
    let eta_samples = eta_samples_by_subject(finalize.chain_states);
    let (likelihood_estimates, minus2ll) = saem_posthoc_likelihood(
        input.equation,
        input.data,
        finalize.residual_error_models,
        input.transforms,
        input.population,
        input.individual_estimates,
        &eta_samples,
        finalize.seed,
    )?;
    tracing::info!("-2LL computed by importance sampling: {:.4}", minus2ll);

    assemble_parametric_result(ParametricResultInput {
        equation: input.equation,
        data: input.data,
        population: input.population,
        individual_estimates: input.individual_estimates,
        objf: minus2ll,
        iterations: input.iterations,
        status: input.status,
        run_configuration: input.run_configuration,
        iteration_log: input.iteration_log,
        likelihood_estimates,
        uncertainty_estimates: input.uncertainty_estimates,
        sigma: input.sigma,
        transforms: input.transforms,
        covariates: input.covariates,
    })
}

struct ParametricWorkspaceInput<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    population: &'a Population,
    individual_estimates: &'a IndividualEstimates,
    objf: f64,
    iterations: usize,
    status: &'a Status,
    run_configuration: RunConfiguration,
    iteration_log: ParametricIterationLog,
    likelihood_estimates: LikelihoodEstimates,
    uncertainty_estimates: UncertaintyEstimates,
    sigma: ResidualErrorEstimates,
    transforms: &'a [ParameterTransform],
    covariates: Option<CovariateState>,
}

fn assemble_parametric_workspace<E: Equation + Clone>(
    input: ParametricWorkspaceInput<'_, E>,
) -> Result<ParametricWorkspace<E>> {
    let population_psi = build_population_in_psi_space(input.population, input.transforms)?;
    let individual_estimates_psi =
        build_individual_estimates_in_psi_space(input.individual_estimates, input.transforms)?;
    let mut state = ParametricModelState::from_population_and_sigma(&population_psi, &input.sigma);
    if let Some(covariates) = input.covariates {
        state.covariates = covariates;
    }
    let individuals = IndividualEffectsState::from_individual_estimates(&individual_estimates_psi);

    Ok(ParametricWorkspace::new(
        state,
        individuals,
        input.equation.clone(),
        input.data.clone(),
        population_psi,
        individual_estimates_psi,
        input.objf,
        input.iterations,
        input.status.clone(),
        input.run_configuration,
        input.iteration_log,
        input.likelihood_estimates,
        input.uncertainty_estimates,
        input.sigma,
        None,
    ))
}

fn build_population_in_psi_space(
    population: &Population,
    transforms: &[ParameterTransform],
) -> Result<Population> {
    let mean_psi = phi_to_psi_vec(transforms, population.mu());
    Population::new(
        mean_psi,
        population.omega().clone(),
        population.parameters().clone(),
    )
}

fn build_individual_estimates_in_psi_space(
    individual_estimates: &IndividualEstimates,
    transforms: &[ParameterTransform],
) -> Result<IndividualEstimates> {
    let individuals = individual_estimates
        .iter()
        .map(|individual| {
            let psi = phi_to_psi_vec(transforms, individual.psi());
            let mut rebuilt = Individual::new(
                individual.subject_id().to_string(),
                individual.eta().clone(),
                psi,
            )?;
            if let Some(objf) = individual.objective_function() {
                rebuilt.set_objective_function(objf);
            }
            Ok(rebuilt)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(IndividualEstimates::from_vec(individuals))
}

#[cfg(test)]
mod tests {
    use super::{assemble_parametric_result, ParametricResultInput};
    use anyhow::Result;
    use faer::{Col, Mat};
    use pharmsol::{Data, Subject};

    use crate::algorithms::{Status, StopReason};
    use crate::api::{
        AlgorithmTuning, ConvergenceOptions, LoggingLevel, LoggingOptions, OutputPlan,
        RuntimeOptions,
    };
    use crate::estimation::parametric::{
        Individual, IndividualEstimates, LikelihoodEstimates, ParameterTransform, Population,
        ResidualErrorEstimates, UncertaintyEstimates,
    };
    use crate::model::{ParameterSpace, ParameterSpec};
    use crate::output::shared::RunConfiguration;
    use crate::prelude::*;

    fn equation() -> equation::ODE {
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0] / v;
            },
        )
    }

    fn data() -> Data {
        let subject = Subject::builder("1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .build();

        Data::new(vec![subject])
    }

    #[test]
    fn saem_assembler_converts_phi_space_outputs_to_psi_space() -> Result<()> {
        let parameters = ParameterSpace::new()
            .add(ParameterSpec::bounded("ke", 0.1, 1.0))
            .add(ParameterSpec::bounded("v", 1.0, 20.0));
        let population = Population::new(
            Col::from_fn(2, |index| if index == 0 { 0.0 } else { 2.0 }),
            Mat::from_fn(2, 2, |row, col| if row == col { 0.25 } else { 0.0 }),
            parameters,
        )?;
        let individual = Individual::new(
            "1",
            Col::from_fn(2, |_| 0.0),
            Col::from_fn(2, |index| if index == 0 { 0.1 } else { 2.1 }),
        )?;
        let run_configuration = RunConfiguration::new(
            Algorithm::SAEM,
            &OutputPlan::disabled(),
            &RuntimeOptions {
                cycles: 100,
                cache: true,
                progress: false,
                idelta: 0.12,
                tad: 0.0,
                prior: None,
                logging: LoggingOptions {
                    initialize: false,
                    level: LoggingLevel::Info,
                    write: false,
                    stdout: false,
                },
                convergence: ConvergenceOptions::default(),
                tuning: AlgorithmTuning::default(),
            },
            vec!["ke".to_string(), "v".to_string()],
        );
        let equation = equation();
        let data = data();
        let individual_estimates = IndividualEstimates::from_vec(vec![individual]);

        let result = assemble_parametric_result(ParametricResultInput {
            equation: &equation,
            data: &data,
            population: &population,
            individual_estimates: &individual_estimates,
            objf: 120.0,
            iterations: 12,
            status: &Status::Stop(StopReason::Converged),
            run_configuration,
            iteration_log: {
                let mut log = ParametricIterationLog::new();
                log.log_iteration(1, 130.0, &population, &Status::Continue);
                log
            },
            likelihood_estimates: LikelihoodEstimates {
                ll_importance_sampling: Some(-60.0),
                is_n_samples: Some(1000),
                ..LikelihoodEstimates::new()
            },
            uncertainty_estimates: UncertaintyEstimates::new(),
            sigma: ResidualErrorEstimates::additive(0.5),
            transforms: &[ParameterTransform::LogNormal, ParameterTransform::LogNormal],
            covariates: None,
        })?;

        assert!((result.mu()[0] - 1.0).abs() < 1e-12);
        assert!((result.mu()[1] - 7.38905609893065).abs() < 1e-12);
        assert_eq!(result.iteration_log().len(), 1);
        assert_eq!(result.likelihoods().is_n_samples, Some(1000));
        assert_eq!(result.sigma().additive, Some(0.5));
        Ok(())
    }
}
