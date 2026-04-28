use anyhow::Result;
use pharmsol::{AssayErrorModel, ErrorPoly, ResidualErrorModel, ResidualErrorModels};
use pmcore::prelude::*;

fn simple_equation() -> equation::ODE {
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

fn simple_data() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .build();

    Data::new(vec![subject])
}

fn structured_parametric_data() -> Data {
    let first = Subject::builder("1")
        .covariate("wt", 0.0, 60.0)
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 12.0, 0)
        .observation(2.0, 8.5, 0)
        .observation(4.0, 4.8, 0)
        .build();

    let second = Subject::builder("2")
        .covariate("wt", 0.0, 90.0)
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 9.5, 0)
        .observation(2.0, 6.4, 0)
        .observation(4.0, 3.1, 0)
        .build();

    Data::new(vec![first, second])
}

fn structured_multi_occasion_parametric_data() -> Data {
    let subject = Subject::builder("1")
        .covariate("wt", 0.0, 70.0)
        .covariate("study_day", 0.0, 1.0)
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .reset()
        .covariate("wt", 0.0, 70.0)
        .covariate("study_day", 0.0, 2.0)
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 8.0, 0)
        .build();

    Data::new(vec![subject])
}

fn assert_subject_covariate_snapshot<E: pharmsol::Equation>(result: &ParametricWorkspace<E>) {
    let covariates = result
        .state()
        .covariates
        .subject_effects
        .as_ref()
        .expect("structured subject covariates should be preserved in the fitted state");

    assert!(result.objf().is_finite());
    assert_eq!(covariates.parameter_names, vec!["ke", "v"]);
    assert_eq!(covariates.column_names, vec!["wt"]);
    assert_eq!(covariates.covariate_mask, vec![vec![true], vec![false]]);
    assert_eq!(covariates.values, vec![vec![Some(60.0)], vec![Some(90.0)]]);
}

#[test]
fn test_model_definition_builder() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

    assert_eq!(model.parameters.len(), 2);
    assert_eq!(model.observations.channels.len(), 1);
    Ok(())
}

#[test]
fn test_unified_fit_nonparametric_smoke() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

    let result = EstimationProblem::builder(model, simple_data())
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 1,
            cache: true,
            progress: false,
            idelta: 0.12,
            tad: 0.0,
            prior: None,
            ..RuntimeOptions::default()
        })
        .run()?;

    assert!(result.objf().is_finite());
    assert_eq!(result.summary().parameter_count, 2);
    assert_eq!(result.population_summary().parameters.len(), 2);
    assert_eq!(result.individual_summaries().len(), 1);
    Ok(())
}

#[test]
fn test_parametric_problem_requires_residual_error_models() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

    let problem = EstimationProblem::builder(model, simple_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Saem(
            SaemOptions,
        )))
        .output(OutputPlan::disabled())
        .build()?;

    assert!(problem.run().is_err());
    Ok(())
}

#[test]
fn test_parametric_problem_accepts_residual_error_models() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
        .with_residual_error_models(residual_error);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

    let problem = EstimationProblem::builder(model, simple_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Saem(
            SaemOptions,
        )))
        .output(OutputPlan::disabled())
        .build()?;

    let compiled = problem.compile()?;
    assert!(compiled.model.observations.residual_error_models.is_some());
    Ok(())
}

#[test]
fn test_unified_fit_parametric_structured_covariates_smoke() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
        .with_residual_error_models(residual_error);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 5.0, 20.0)),
        )
        .observations(observations)
        .covariates(CovariateSpec::Structured(CovariateEffectsSpec {
            subject_effects: Some(CovariateModel::new(
                vec!["ke", "v"],
                vec!["wt"],
                vec![vec![true], vec![false]],
            )?),
            occasion_effects: None,
        }))
        .build()?;

    let result = EstimationProblem::builder(model, structured_parametric_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Saem(
            SaemOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            progress: false,
            tuning: AlgorithmTuning {
                saem: SaemConfig {
                    k1_iterations: 2,
                    k2_iterations: 1,
                    burn_in: 1,
                    mcmc_iterations: 1,
                    n_kernels: 1,
                    compute_map: false,
                    compute_fim: false,
                    compute_ll_is: false,
                    compute_ll_gq: false,
                    n_mc_is: 32,
                    ..SaemConfig::default()
                },
                ..AlgorithmTuning::default()
            },
            ..RuntimeOptions::default()
        })
        .run()?;

    let result = result
        .as_parametric()
        .expect("SAEM should yield a parametric result");
    assert_subject_covariate_snapshot(result);
    assert_eq!(
        result
            .state()
            .covariates
            .subject_effects
            .as_ref()
            .expect("structured subject covariates should be preserved in the fitted state")
            .coefficients
            .len(),
        3
    );
    Ok(())
}

#[test]
fn test_unified_fit_parametric_focei_smoke() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
        .with_residual_error_models(residual_error);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

    let result = EstimationProblem::builder(model, simple_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Focei(
            FoceiOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 3,
            progress: false,
            ..RuntimeOptions::default()
        })
        .run()?;

    let result = result
        .as_parametric()
        .expect("FOCEI should yield a parametric result");

    assert!(result.objf().is_finite());
    assert_eq!(result.population().param_names(), vec!["ke", "v"]);
    assert_eq!(result.individual_estimates().nsubjects(), 1);
    assert_eq!(result.sigma().combined, Some((0.5, 0.1)));
    assert!(has_fim(result));
    assert_eq!(fim_method(result), Some(FimMethod::Linearization));
    Ok(())
}

#[test]
fn test_unified_fit_parametric_focei_structured_covariates_smoke() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
        .with_residual_error_models(residual_error);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 5.0, 20.0)),
        )
        .observations(observations)
        .covariates(CovariateSpec::Structured(CovariateEffectsSpec {
            subject_effects: Some(CovariateModel::new(
                vec!["ke", "v"],
                vec!["wt"],
                vec![vec![true], vec![false]],
            )?),
            occasion_effects: None,
        }))
        .build()?;

    let result = EstimationProblem::builder(model, structured_parametric_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Focei(
            FoceiOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 3,
            progress: false,
            ..RuntimeOptions::default()
        })
        .run()?;

    let result = result
        .as_parametric()
        .expect("FOCEI should yield a parametric result");
    assert_subject_covariate_snapshot(result);
    assert_eq!(result.individual_estimates().nsubjects(), 2);
    Ok(())
}

#[test]
fn test_unified_fit_parametric_focei_preserves_occasion_covariates() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
        .with_residual_error_models(residual_error);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 5.0, 20.0)),
        )
        .observations(observations)
        .covariates(CovariateSpec::Structured(CovariateEffectsSpec {
            subject_effects: Some(CovariateModel::new(
                vec!["ke", "v"],
                vec!["wt"],
                vec![vec![true], vec![false]],
            )?),
            occasion_effects: Some(CovariateModel::new(
                vec!["ke", "v"],
                vec!["study_day"],
                vec![vec![true], vec![false]],
            )?),
        }))
        .build()?;

    let result = EstimationProblem::builder(model, structured_multi_occasion_parametric_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Focei(
            FoceiOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 2,
            progress: false,
            ..RuntimeOptions::default()
        })
        .run()?;

    let result = result
        .as_parametric()
        .expect("FOCEI should yield a parametric result");
    let occasion = result
        .state()
        .covariates
        .occasion_effects
        .as_ref()
        .expect("occasion covariates should be preserved in the fitted state");

    assert!(result.objf().is_finite());
    assert_eq!(occasion.column_names, vec!["study_day"]);
    assert_eq!(occasion.parameter_names, vec!["ke", "v"]);
    assert_eq!(occasion.values, vec![vec![Some(1.0)], vec![Some(2.0)]]);
    Ok(())
}

#[test]
fn test_unified_fit_parametric_saem_preserves_occasion_covariates() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
        .with_residual_error_models(residual_error);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec {
                    name: "ke".to_string(),
                    domain: ParameterDomain::Bounded {
                        lower: 0.1,
                        upper: 1.0,
                    },
                    transform: ModelParameterTransform::Identity,
                    initial: Some(0.4),
                    estimate: true,
                    variability: ParameterVariability::SubjectAndOccasion,
                })
                .add(ParameterSpec::bounded("v", 5.0, 20.0)),
        )
        .observations(observations)
        .covariates(CovariateSpec::Structured(CovariateEffectsSpec {
            subject_effects: Some(CovariateModel::new(
                vec!["ke", "v"],
                vec!["wt"],
                vec![vec![true], vec![false]],
            )?),
            occasion_effects: Some(CovariateModel::new(
                vec!["ke", "v"],
                vec!["study_day"],
                vec![vec![true], vec![false]],
            )?),
        }))
        .build()?;

    let result = EstimationProblem::builder(model, structured_multi_occasion_parametric_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Saem(
            SaemOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            progress: false,
            tuning: AlgorithmTuning {
                saem: SaemConfig {
                    k1_iterations: 2,
                    k2_iterations: 1,
                    burn_in: 1,
                    mcmc_iterations: 1,
                    n_kernels: 1,
                    compute_map: false,
                    compute_fim: false,
                    compute_ll_is: false,
                    compute_ll_gq: false,
                    n_mc_is: 32,
                    ..SaemConfig::default()
                },
                ..AlgorithmTuning::default()
            },
            ..RuntimeOptions::default()
        })
        .run()?;

    let result = result
        .as_parametric()
        .expect("SAEM should yield a parametric result");
    let occasion = result
        .state()
        .covariates
        .occasion_effects
        .as_ref()
        .expect("occasion covariates should be preserved in the fitted state");
    let occasion_kappa = result
        .individuals()
        .occasion_kappa
        .as_ref()
        .expect("occasion effect slots should exist for occasion-enabled SAEM models");

    assert!(result.objf().is_finite());
    assert_eq!(occasion.column_names, vec!["study_day"]);
    assert_eq!(occasion.parameter_names, vec!["ke", "v"]);
    assert_eq!(occasion.values, vec![vec![Some(1.0)], vec![Some(2.0)]]);
    assert_eq!(occasion_kappa.0.len(), 2);
    assert_eq!(occasion_kappa.0[0].subject_index, 0);
    assert_eq!(occasion_kappa.0[0].occasion_index, 0);
    assert_eq!(occasion_kappa.0[1].occasion_index, 1);
    assert_eq!(occasion_kappa.0[0].values.0, vec![0.0, 0.0]);
    assert_eq!(
        result
            .state()
            .variability
            .occasion
            .as_ref()
            .expect("occasion variability should be present")
            .enabled_for,
        vec![true, false]
    );
    Ok(())
}

#[test]
fn test_problem_compile_preserves_runtime_configuration() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

    let runtime = RuntimeOptions {
        cycles: 7,
        cache: false,
        progress: false,
        idelta: 0.5,
        tad: 24.0,
        prior: None,
        logging: LoggingOptions {
            initialize: false,
            level: LoggingLevel::Debug,
            write: true,
            stdout: false,
        },
        convergence: ConvergenceOptions {
            likelihood: 1e-5,
            pyl: 5e-3,
            eps: 2e-3,
        },
        tuning: AlgorithmTuning {
            min_distance: 2e-4,
            nm_steps: 222,
            tolerance: 3e-6,
            saem: SaemConfig {
                k1_iterations: 111,
                k2_iterations: 22,
                ..SaemConfig::default()
            },
        },
    };

    let compiled = EstimationProblem::builder(model, simple_data())
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(runtime)
        .build()?
        .compile()?;

    assert_eq!(compiled.method().algorithm(), Algorithm::NPAG);
    assert!(!compiled.output_plan().write);
    assert_eq!(compiled.runtime_options().cycles, 7);
    assert!(!compiled.runtime_options().cache);
    assert!(!compiled.runtime_options().progress);
    assert_eq!(compiled.runtime_options().idelta, 0.5);
    assert_eq!(compiled.runtime_options().tad, 24.0);
    assert_eq!(
        compiled.runtime_options().logging.level,
        LoggingLevel::Debug
    );
    assert!(compiled.runtime_options().logging.write);
    assert!(!compiled.runtime_options().logging.stdout);
    assert_eq!(compiled.runtime_options().convergence.likelihood, 1e-5);
    assert_eq!(compiled.runtime_options().convergence.pyl, 5e-3);
    assert_eq!(compiled.runtime_options().convergence.eps, 2e-3);
    assert_eq!(compiled.runtime_options().tuning.min_distance, 2e-4);
    assert_eq!(compiled.runtime_options().tuning.nm_steps, 222);
    assert_eq!(compiled.runtime_options().tuning.tolerance, 3e-6);
    assert_eq!(compiled.runtime_options().tuning.saem.k1_iterations, 111);
    assert_eq!(compiled.runtime_options().tuning.saem.k2_iterations, 22);
    Ok(())
}

#[test]
fn test_problem_can_initialize_logs_without_old_settings_api() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

    let problem = EstimationProblem::builder(model, simple_data())
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            logging: LoggingOptions {
                initialize: true,
                level: LoggingLevel::Info,
                write: false,
                stdout: false,
            },
            ..RuntimeOptions::default()
        })
        .build()?;

    problem.initialize_logs()?;
    Ok(())
}
