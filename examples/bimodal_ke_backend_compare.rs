use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use pharmsol::dsl::{self, CompiledRuntimeModel, RuntimeBackend, RuntimeCompilationTarget};
use pharmsol::Cache;
use pmcore::prelude::*;

const DATA_PATH: &str = "examples/bimodal_ke/bimodal_ke.csv";
const NPAG_CYCLES: usize = 1000;
const BIMODAL_KE_DSL: &str = r#"
name = bimodal_ke
kind = ode

params = ke, v
states = central
outputs = 1

infusion(1) -> central

dx(central) = -ke * central

out(1) = central / v
"#;

#[derive(Debug, Clone)]
struct ComparisonResult {
    label: &'static str,
    compile_time: Duration,
    fit_time: Duration,
    total_time: Duration,
    objf: f64,
    cycles: usize,
}

fn main() -> Result<()> {
    let results = vec![
        run_legacy()?,
        run_macro()?,
        run_runtime_jit()?,
        run_runtime_native_aot()?,
        // run_runtime_wasm()?,
    ];

    print_summary(&results)?;

    Ok(())
}

fn parameter_space() -> ParameterSpace {
    ParameterSpace::new()
        .add(ParameterSpec::bounded("ke", 0.001, 3.0))
        .add(ParameterSpec::bounded("v", 25.0, 250.0))
}

fn observations() -> ObservationSpec {
    ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(
            AssayErrorModels::new()
                .add(
                    0,
                    AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
                )
                .unwrap(),
        )
}

fn runtime_options() -> RuntimeOptions {
    RuntimeOptions {
        cycles: NPAG_CYCLES,
        cache: true,
        progress: false,
        logging: LoggingOptions {
            initialize: false,
            ..LoggingOptions::default()
        },
        ..RuntimeOptions::default()
    }
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
            .outputs(["1"])
            .route(
                equation::Route::infusion("1")
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
    .enable_cache()
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45))
}

fn artifact_root(name: &str) -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before unix epoch")?
        .as_millis();
    let path = std::env::current_dir()?
        .join("target")
        .join("bimodal_ke_backend_compare")
        .join(format!("{name}-{stamp}-{}", std::process::id()));
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn compile_runtime_ode(
    label: &'static str,
    expected_backend: RuntimeBackend,
    target: RuntimeCompilationTarget,
) -> Result<(pharmsol::dsl::RuntimeOdeModel, Duration)> {
    let started = Instant::now();
    let compiled = dsl::compile_module_source_to_runtime(
        BIMODAL_KE_DSL,
        Some("bimodal_ke"),
        target,
        |_, _| {},
    )?;
    let compile_time = started.elapsed();

    let model = match compiled {
        CompiledRuntimeModel::Ode(model) => model
            .enable_cache()
            .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45)),
        CompiledRuntimeModel::Analytical(_) => {
            bail!("{label} compiled bimodal_ke as analytical instead of ode")
        }
        CompiledRuntimeModel::Sde(_) => bail!("{label} compiled bimodal_ke as sde instead of ode"),
    };

    if model.backend() != expected_backend {
        bail!(
            "{label} compiled to {:?}, expected {:?}",
            model.backend(),
            expected_backend
        );
    }

    Ok((model, compile_time))
}

fn run_case<E: pharmsol::Equation + Clone + Send + 'static>(
    label: &'static str,
    compile_time: Duration,
    equation: E,
) -> Result<ComparisonResult> {
    let model = ModelDefinition::builder(equation)
        .parameters(parameter_space())
        .observations(observations())
        .build()?;

    let data = data::read_pmetrics(DATA_PATH)?;
    let fit_started = Instant::now();
    let result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions::default(),
        )))
        .output(OutputPlan {
            write: false,
            path: None,
        })
        .runtime(runtime_options())
        .run()?;
    let fit_time = fit_started.elapsed();

    summarize_result(label, compile_time, fit_time, &result)
}

fn summarize_result<E: pharmsol::Equation>(
    label: &'static str,
    compile_time: Duration,
    fit_time: Duration,
    result: &FitResult<E>,
) -> Result<ComparisonResult> {
    let workspace = result
        .as_nonparametric()
        .ok_or_else(|| anyhow!("expected nonparametric result for {label}"))?;

    Ok(ComparisonResult {
        label,
        compile_time,
        fit_time,
        total_time: compile_time + fit_time,
        objf: result.objf(),
        cycles: workspace.cycles(),
    })
}

fn run_legacy() -> Result<ComparisonResult> {
    let started = Instant::now();
    let equation = legacy_equation();
    run_case("legacy-handwritten", started.elapsed(), equation)
}

fn run_macro() -> Result<ComparisonResult> {
    let started = Instant::now();
    let equation = macro_equation();
    run_case("macro-ode", started.elapsed(), equation)
}

fn run_runtime_jit() -> Result<ComparisonResult> {
    let (equation, compile_time) = compile_runtime_ode(
        "dsl-jit",
        RuntimeBackend::Jit,
        RuntimeCompilationTarget::Jit,
    )?;
    run_case("dsl-jit", compile_time, equation)
}

fn run_runtime_native_aot() -> Result<ComparisonResult> {
    let root = artifact_root("native-aot")?;
    let (equation, compile_time) = compile_runtime_ode(
        "dsl-native-aot",
        RuntimeBackend::NativeAot,
        RuntimeCompilationTarget::NativeAot(
            dsl::NativeAotCompileOptions::new(root.join("build"))
                .with_output(root.join("bimodal_ke_runtime_aot.pkm")),
        ),
    )?;
    run_case("dsl-native-aot", compile_time, equation)
}

fn run_runtime_wasm() -> Result<ComparisonResult> {
    let (equation, compile_time) = compile_runtime_ode(
        "dsl-wasm",
        RuntimeBackend::Wasm,
        RuntimeCompilationTarget::Wasm,
    )?;
    run_case("dsl-wasm", compile_time, equation)
}

fn print_summary(results: &[ComparisonResult]) -> Result<()> {
    let baseline = results
        .iter()
        .find(|result| result.label == "legacy-handwritten")
        .ok_or_else(|| anyhow!("missing legacy-handwritten baseline"))?;

    println!("bimodal_ke NPAG backend comparison");
    println!("dataset: {DATA_PATH}");
    println!("cycles: {NPAG_CYCLES}, cache: on");
    println!();
    println!(
        "{:<18} {:>12} {:>12} {:>12} {:>12} {:>14} {:>14}",
        "representation", "compile s", "fit s", "total s", "cycles", "objf diff", "objf"
    );

    for result in results {
        println!(
            "{:<18} {:>12.3} {:>12.3} {:>12.3} {:>12} {:>14.6} {:>14.6}",
            result.label,
            result.compile_time.as_secs_f64(),
            result.fit_time.as_secs_f64(),
            result.total_time.as_secs_f64(),
            result.cycles,
            (result.objf - baseline.objf).abs(),
            result.objf,
        );
    }

    Ok(())
}
