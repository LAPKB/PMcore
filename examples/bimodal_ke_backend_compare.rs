use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use pharmsol::dsl::{compile_module_source_to_runtime, CompiledRuntimeModel, RuntimeOdeModel};
use pharmsol::Cache;
use pmcore::prelude::*;

const DATA_PATH: &str = "examples/bimodal_ke/bimodal_ke.csv";

const BIMODAL_KE_DSL: &str = r#"
name = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = outeq_1

infusion(input_1) -> central

dx(central) = -ke * central

out(outeq_1) = central / v
"#;

#[derive(Debug, Clone)]
struct ComparisonResult {
    label: &'static str,
    compile_time: Duration,
    fit_time: Duration,
    total_time: Duration,
    objf: f64,
}

fn main() -> Result<()> {
    let results = vec![run_legacy()?, run_macro()?, run_dsl()?];

    print_summary(&results)?;

    Ok(())
}

fn legacy_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _bolus, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("bimodal_ke_legacy")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["outeq_1"])
            .route(
                equation::Route::infusion("input_1")
                    .to_state("central")
                    .inject_input_to_destination(),
            ),
    )
    .expect("legacy bimodal_ke metadata should validate")
    .enable_cache()
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45))
}

fn macro_equation() -> equation::ODE {
    ode! {
        name: "bimodal_ke",
        params: [ke, v],
        states: [central],
        outputs: [outeq_1],
        routes: [
            infusion(input_1) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / v;
        },
    }
    .enable_cache()
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45))
}

fn dsl_equation() -> Result<RuntimeOdeModel> {
    let compiled = compile_module_source_to_runtime(BIMODAL_KE_DSL, Some("bimodal_ke"), |_, _| {})
        .map_err(|e| anyhow!("failed to compile DSL model: {e}"))?;

    match compiled {
        CompiledRuntimeModel::Ode(model) => Ok(model
            .enable_cache()
            .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45))),
        other => Err(anyhow!("expected an ODE model, got {:?}", other.kind())),
    }
}

fn run_case<E: pharmsol::Equation + Clone + Send + 'static + EquationMetadataSource>(
    label: &'static str,
    compile_time: Duration,
    equation: E,
) -> Result<ComparisonResult> {
    let data = data::read_pmetrics(DATA_PATH)?;
    let fit_started = Instant::now();
    let parameters = ParameterSpace::bounded()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);
    let prior = Theta::sobol_default(&parameters)?;
    let error_models = AssayErrorModels::new().add(
        "outeq_1",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
    )?;
    let result = EstimationProblem::nonparametric(equation, data, prior, error_models)?
        .fit_with(NonParametricAlgorithm::npag())?;
    let fit_time = fit_started.elapsed();

    Ok(ComparisonResult {
        label,
        compile_time,
        fit_time,
        total_time: compile_time + fit_time,
        objf: result.objf(),
    })
}

fn run_legacy() -> Result<ComparisonResult> {
    let started = Instant::now();
    let equation = legacy_equation();
    run_case("handwritten", started.elapsed(), equation)
}

fn run_macro() -> Result<ComparisonResult> {
    let started = Instant::now();
    let equation = macro_equation();
    run_case("macro-ode", started.elapsed(), equation)
}

fn run_dsl() -> Result<ComparisonResult> {
    let started = Instant::now();
    let equation = dsl_equation()?;
    run_case("dsl-runtime", started.elapsed(), equation)
}

fn print_summary(results: &[ComparisonResult]) -> Result<()> {
    let baseline = results
        .iter()
        .find(|result| result.label == "handwritten")
        .ok_or_else(|| anyhow!("missing handwritten baseline"))?;

    println!("bimodal_ke NPAG backend comparison");
    println!("dataset: {DATA_PATH}");
    println!("cache: on");
    println!();
    println!(
        "{:<18} {:>12} {:>12} {:>12} {:>14} {:>14}",
        "representation", "compile s", "fit s", "total s", "objf diff", "objf"
    );

    for result in results {
        println!(
            "{:<18} {:>12.3} {:>12.3} {:>12.3} {:>14.6} {:>14.6}",
            result.label,
            result.compile_time.as_secs_f64(),
            result.fit_time.as_secs_f64(),
            result.total_time.as_secs_f64(),
            (result.objf - baseline.objf).abs(),
            result.objf,
        );
    }

    Ok(())
}
