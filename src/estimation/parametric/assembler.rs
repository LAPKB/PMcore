use anyhow::Result;
use faer::{Col, Mat};
use pharmsol::{Data, Equation};

use crate::algorithms::Status;
use crate::estimation::parametric::{
    phi_to_psi_vec, CovariateState, Individual, IndividualEffectsState, IndividualEstimates,
    LikelihoodEstimates, ParameterTransform, ParametricIterationLog, ParametricModelState,
    ParametricWorkspace, Population, ResidualErrorEstimates, UncertaintyEstimates,
};
use crate::output::shared::RunConfiguration;

pub(crate) struct SaemResultInput<'a, E: Equation> {
    pub equation: &'a E,
    pub data: &'a Data,
    pub population: &'a Population,
    pub individual_estimates: &'a IndividualEstimates,
    pub objf: f64,
    pub iterations: usize,
    pub status: &'a Status,
    pub run_configuration: RunConfiguration,
    pub likelihood_estimates: LikelihoodEstimates,
    pub param_history: &'a [Vec<f64>],
    pub objf_history: &'a [f64],
    pub burn_in_iterations: usize,
    pub sigma_sq: f64,
    pub transforms: &'a [ParameterTransform],
    pub covariates: Option<CovariateState>,
}

pub(crate) struct FoceiResultInput<'a, E: Equation> {
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
    pub transforms: &'a [ParameterTransform],
    pub covariates: Option<CovariateState>,
}

pub(crate) fn assemble_saem_result<E: Equation + Clone>(
    input: SaemResultInput<'_, E>,
) -> Result<ParametricWorkspace<E>> {
    let iteration_log = build_iteration_log(
        input.population,
        input.param_history,
        input.objf_history,
        input.objf,
        input.status,
        input.burn_in_iterations,
    );

    assemble_parametric_workspace(ParametricWorkspaceInput {
        equation: input.equation,
        data: input.data,
        population: input.population,
        individual_estimates: input.individual_estimates,
        objf: input.objf,
        iterations: input.iterations,
        status: input.status,
        run_configuration: input.run_configuration,
        iteration_log,
        likelihood_estimates: input.likelihood_estimates,
        uncertainty_estimates: UncertaintyEstimates::new(),
        sigma: ResidualErrorEstimates::additive(input.sigma_sq.sqrt()),
        transforms: input.transforms,
        covariates: input.covariates,
    })
}

pub(crate) fn assemble_focei_result<E: Equation + Clone>(
    input: FoceiResultInput<'_, E>,
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
        uncertainty_estimates: UncertaintyEstimates::new(),
        sigma: ResidualErrorEstimates::default(),
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

fn build_iteration_log(
    population: &Population,
    param_history: &[Vec<f64>],
    objf_history: &[f64],
    fallback_objf: f64,
    final_status: &Status,
    burn_in_iterations: usize,
) -> ParametricIterationLog {
    let mut iteration_log = ParametricIterationLog::new();
    let n_parameters = population.npar();

    for (index, parameters) in param_history.iter().enumerate() {
        let mu_phi = Col::from_fn(n_parameters, |j| parameters.get(j).copied().unwrap_or(0.0));
        let omega = Mat::from_fn(n_parameters, n_parameters, |row, col| {
            if row == col {
                parameters.get(n_parameters + row).copied().unwrap_or(0.0)
            } else {
                0.0
            }
        });

        if let Ok(population_snapshot) =
            Population::new(mu_phi, omega, population.parameters().clone())
        {
            let status = if index < burn_in_iterations {
                Status::Continue
            } else {
                final_status.clone()
            };
            let iteration_objf = objf_history.get(index).copied().unwrap_or(fallback_objf);
            iteration_log.log_iteration(index + 1, iteration_objf, &population_snapshot, &status);
        }
    }

    iteration_log
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
    use super::{assemble_saem_result, SaemResultInput};
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

        let result = assemble_saem_result(SaemResultInput {
            equation: &equation,
            data: &data,
            population: &population,
            individual_estimates: &individual_estimates,
            objf: 120.0,
            iterations: 12,
            status: &Status::Stop(StopReason::Converged),
            run_configuration,
            likelihood_estimates: LikelihoodEstimates {
                ll_importance_sampling: Some(-60.0),
                is_n_samples: Some(1000),
                ..LikelihoodEstimates::new()
            },
            param_history: &[vec![0.0, 2.0, 0.25, 0.25]],
            objf_history: &[130.0],
            burn_in_iterations: 5,
            sigma_sq: 0.25,
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
