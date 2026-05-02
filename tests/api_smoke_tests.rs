use anyhow::Result;
use pharmsol::{AssayErrorModel, ErrorPoly};
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
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("simple_equation")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["0"])
            .route(equation::Route::bolus("0").to_state("central")),
    )
    .expect("metadata attachment should validate")
}

fn simple_data() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .build();

    Data::new(vec![subject])
}

fn metadata_equation() -> equation::ODE {
    ode! {
        name: "metadata_derived_outputs",
        params: [ke, v],
        states: [central],
        outputs: [1],
        routes: [
            infusion(1) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[1] = x[central] / v;
        },
    }
}

#[test]
fn test_model_definition_builder() -> Result<()> {
    let model = ModelDefinition::builder(simple_equation())
        .parameter(Parameter::bounded("ke", 0.1, 1.0))?
        .parameter(Parameter::bounded("v", 1.0, 20.0))?
        .build()?;

    assert_eq!(model.parameters.len(), 2);
    assert_eq!(model.parameter_count(), 2);
    assert_eq!(model.parameter_name(0), Some("ke"));
    assert_eq!(model.output_count(), 1);
    assert_eq!(model.output_index("0"), Some(0));
    Ok(())
}

#[test]
fn test_model_definition_rejects_unknown_parameter_name() {
    let err = ModelDefinition::builder(simple_equation())
        .parameter(Parameter::bounded("clearance", 0.1, 1.0))
        .err()
        .expect("parameter name should be validated against equation metadata");

    assert!(err
        .to_string()
        .contains("unknown equation parameter: clearance"));
}

#[test]
fn test_model_definition_derives_observations_from_equation_metadata() -> Result<()> {
    let model = ModelDefinition::builder(metadata_equation())
        .parameter(Parameter::bounded("ke", 0.1, 1.0))?
        .parameter(Parameter::bounded("v", 1.0, 20.0))?
        .build()?;

    assert_eq!(model.output_count(), 1);
    assert_eq!(model.output_name(0), Some("1"));
    assert_eq!(model.output_index("1"), Some(0));

    Ok(())
}

#[test]
fn test_unified_fit_nonparametric_smoke() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let result = EstimationProblem::builder(simple_equation(), simple_data())
        .parameter(Parameter::bounded("ke", 0.1, 1.0))?
        .parameter(Parameter::bounded("v", 1.0, 20.0))?
        .method(Npag::new())
        .error("0", assay_error)?
        .cycles(1)
        .progress(false)
        .fit()?;

    assert!(result.objf().is_finite());
    assert_eq!(result.summary().parameter_count, 2);
    assert_eq!(result.population_summary().parameters.len(), 2);
    assert_eq!(result.individual_summaries().len(), 1);
    Ok(())
}

#[test]
fn test_problem_compile_preserves_runtime_configuration() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let convergence = ConvergenceOptions {
        likelihood: 1e-5,
        pyl: 5e-3,
        eps: 2e-3,
    };
    let tuning = AlgorithmTuning {
        min_distance: 2e-4,
        nm_steps: 222,
        tolerance: 3e-6,
        saem: SaemConfig {
            k1_iterations: 111,
            k2_iterations: 22,
            ..SaemConfig::default()
        },
    };

    let compiled = EstimationProblem::builder(simple_equation(), simple_data())
        .parameter(Parameter::bounded("ke", 0.1, 1.0))?
        .parameter(Parameter::bounded("v", 1.0, 20.0))?
        .method(Npag::new())
        .error("0", assay_error)?
        .cycles(7)
        .cache(false)
        .progress(false)
        .idelta(0.5)
        .tad(24.0)
        .log_level(LoggingLevel::Debug)
        .write_logs(true)
        .stdout_logs(false)
        .convergence(convergence)
        .tuning(tuning)
        .build()?
        .compile()?;

    assert_eq!(compiled.algorithm(), Algorithm::NPAG);
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
    let problem = EstimationProblem::builder(simple_equation(), simple_data())
        .parameter(Parameter::bounded("ke", 0.1, 1.0))?
        .parameter(Parameter::bounded("v", 1.0, 20.0))?
        .method(Npag::new())
        .error("0", assay_error)?
        .initialize_logs()
        .log_level(LoggingLevel::Info)
        .write_logs(false)
        .stdout_logs(false)
        .build()?;

    problem.initialize_logs()?;
    Ok(())
}
