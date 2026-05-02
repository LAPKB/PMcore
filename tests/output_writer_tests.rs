use anyhow::Result;
use pharmsol::{AssayErrorModel, ErrorPoly};
use pmcore::prelude::*;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

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
        equation::metadata::new("output_writer")
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

fn temp_output_dir() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("pmcore-output-writer-{unique}"))
}

#[test]
fn test_fit_result_writes_shared_output_files() -> Result<()> {
    let output_dir = temp_output_dir();
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let result = EstimationProblem::builder(simple_equation(), simple_data())
        .parameter(Parameter::bounded("ke", 0.1, 1.0))?
        .parameter(Parameter::bounded("v", 1.0, 20.0))?
        .method(Npag::new())
        .error("0", assay_error)?
        .output_dir(output_dir.to_string_lossy().to_string())
        .cycles(1)
        .progress(false)
        .fit()?;

    assert!(output_dir.join("settings.json").exists());
    assert!(output_dir.join("summary.json").exists());
    assert!(output_dir.join("summary.csv").exists());
    assert!(output_dir.join("diagnostics.json").exists());
    assert!(output_dir.join("predictions.csv").exists());
    assert!(output_dir.join("iterations.csv").exists());
    assert!(!output_dir.join("pred.csv").exists());
    assert!(!output_dir.join("cycles.csv").exists());
    assert!(!output_dir.join("covs.csv").exists());

    let artifacts = result.artifacts();
    assert!(artifacts.files.iter().any(|file| file == "settings.json"));
    assert!(artifacts.files.iter().any(|file| file == "predictions.csv"));
    assert!(artifacts
        .expected_files
        .iter()
        .any(|file| file == "settings.json"));
    assert!(artifacts
        .shared_expected_files
        .iter()
        .any(|file| file == "settings.json"));
    assert!(artifacts
        .method_specific_expected_files
        .iter()
        .any(|file| file == "iterations.csv"));
    assert!(artifacts.missing_files.is_empty());

    let predictions = result.predictions();
    assert!(predictions.available);
    assert_eq!(predictions.artifact.as_deref(), Some("predictions.csv"));
    assert_eq!(predictions.source.as_deref(), Some("in_memory"));

    let diagnostics = result.diagnostics();
    assert!(diagnostics.estimator_metadata.contains_key("algorithm"));

    let _ = std::fs::remove_dir_all(output_dir);
    Ok(())
}
