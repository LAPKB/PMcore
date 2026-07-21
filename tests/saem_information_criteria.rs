use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use pharmsol::prelude::*;
use pmcore::prelude::*;

static EQUATION_CALLS: AtomicUsize = AtomicUsize::new(0);

fn exact_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "n3_exact_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    exact_problem_from_equation(equation)
}

fn instrumented_exact_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "n3_instrumented_exact_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            EQUATION_CALLS.fetch_add(1, Ordering::SeqCst);
            y[cp] = x[central] / v;
        },
    };
    exact_problem_from_equation(equation)
}

fn exact_problem_from_equation(
    equation: pharmsol::equation::Analytical,
) -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let data = Data::new(vec![
        Subject::builder("s1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.8, "cp")
            .build(),
        Subject::builder("s2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.1, "cp")
            .build(),
    ]);
    EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.25)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new())
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.35)).fixed(),
        )
        .build()
        .expect("exact N3 fixture")
}

fn mixed_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "n3_mixed_count_fixture",
        params: [ke, v, bio],
        states: [central],
        outputs: [cp, amount],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = bio * x[central] / v;
            y[amount] = x[central];
        },
    };
    let subject = |id: &str, shift: f64| {
        Subject::builder(id)
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.9 + shift, "cp")
            .observation(1.0, 98.0 + shift, "amount")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 4.7 + shift, "cp")
            .observation(13.0, 96.0 + shift, "amount")
            .build()
    };
    EstimationProblem::parametric(
        equation,
        Data::new(vec![subject("one", 0.0), subject("two", 0.2)]),
    )
    .parameter(Parameter::log("ke").with_initial(0.25))
    .parameter(Parameter::log("v").with_initial(20.0).fixed())
    .parameter(
        Parameter::log("bio")
            .with_initial(1.0)
            .fixed()
            .without_random_effect(),
    )
    .omega(
        Omega::new()
            .variance("ke", 0.09)
            .fixed_variance("v", 0.16)
            .covariance("ke", "v", 0.02),
    )
    .iov(
        Iov::new()
            .variance("ke", 0.04)
            .fixed_variance("v", 0.09)
            .fixed_covariance("ke", "v", 0.01),
    )
    .error_model(
        "cp",
        ParametricErrorModel::new(ResidualErrorModel::combined(0.3, 0.05))
            .fixed_combined_proportional(),
    )
    .error_model(
        "amount",
        ParametricErrorModel::new(ResidualErrorModel::proportional(0.05)).fixed(),
    )
    .build()
    .expect("mixed N3 count fixture")
}

fn config(seed: u64) -> SaemConfig {
    SaemConfig::new()
        .seed(seed)
        .n_chains(1)
        .mcmc_iterations(1)
        .burn_in(0)
        .k1_iterations(1)
        .k2_iterations(0)
        .compute_map(true)
}

fn n2(seed: u64) -> MarginalLikelihoodConfig {
    MarginalLikelihoodConfig::new(64, seed, 5, 1.5)
}

fn temp_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "pmcore-n3-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ))
}

fn replace_json_string(value: &mut serde_json::Value, old: &str, new: &str) {
    match value {
        serde_json::Value::String(text) if text == old => *text = new.to_string(),
        serde_json::Value::Array(values) => {
            for value in values {
                replace_json_string(value, old, new);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values_mut() {
                replace_json_string(value, old, new);
            }
        }
        _ => {}
    }
}

#[test]
fn exact_n2_drives_aic_bic_and_not_requested_is_explicit() {
    let result = exact_problem()
        .fit_with(config(91).marginal_likelihood(n2(901)))
        .expect("exact N3 fit");
    let n2ll = result.marginal_n2ll().expect("exact marginal N2LL");
    let mcse = result.marginal_n2ll_mcse().expect("exact N2LL MCSE");
    assert_eq!(result.free_parameter_count(), 0);
    assert_eq!(result.aic(), Some(n2ll));
    assert_eq!(result.bic(), Some(n2ll));
    assert_eq!(result.aic_mcse().unwrap().to_bits(), mcse.to_bits());
    assert_eq!(result.bic_mcse().unwrap().to_bits(), mcse.to_bits());
    assert_eq!(
        result.information_criteria().status,
        InformationCriteriaStatus::Available
    );
    assert_eq!(
        result.information_criteria().sample_size_convention,
        InformationCriteriaSampleSizeConvention::IndependentSubjects
    );
    assert_eq!(result.information_criteria().subject_count, 2);
    assert_eq!(
        result.summary().information_criteria,
        Some(result.information_criteria().clone())
    );
    assert_eq!(
        result.population_summary().information_criteria,
        Some(result.information_criteria().clone())
    );

    let disabled = exact_problem()
        .fit_with(config(92))
        .expect("N2-disabled N3 fit");
    assert_eq!(
        disabled.information_criteria().status,
        InformationCriteriaStatus::NotRequested
    );
    assert_eq!((disabled.aic(), disabled.bic()), (None, None));
    assert!(disabled.marginal_likelihood_diagnostics().is_none());
}

#[test]
fn mixed_real_metadata_counts_each_free_coordinate_once() {
    let result = mixed_problem()
        .fit_with(config(93).marginal_likelihood(n2(903)))
        .expect("mixed count N3 fit");
    assert_eq!(
        result.information_criteria().parameter_count,
        InformationCriteriaParameterCount {
            population: 1,
            covariate: 0,
            omega: 2,
            omega_iov: 1,
            residual: 1,
            total: 5,
        }
    );
    let source_subjects = match result.marginal_likelihood_status().unwrap() {
        MarginalLikelihoodStatus::AvailableWithNonconvergedModes { subjects } => subjects.clone(),
        status => panic!("expected nonconverged-mode source status, got {status:?}"),
    };
    assert_eq!(
        result.information_criteria().status,
        InformationCriteriaStatus::AvailableWithNonconvergedModes {
            subjects: source_subjects,
        }
    );
    let n2ll = result.marginal_n2ll().expect("mixed marginal N2LL");
    assert!((result.aic().unwrap() - (n2ll + 10.0)).abs() <= 1e-12);
    assert!((result.bic().unwrap() - (n2ll + 5.0 * 2.0_f64.ln())).abs() <= 1e-12);
}

#[test]
fn criteria_access_is_pure_and_seeded_fit_boundary_is_bit_exact() {
    // The criteria API takes neither an equation nor an RNG. Repeating the
    // complete seeded fit/N2 boundary bit-exactly is the observable RNG-
    // isolation regression; no production RNG hook is needed.
    EQUATION_CALLS.store(0, Ordering::SeqCst);
    let first = instrumented_exact_problem()
        .fit_with(config(94).marginal_likelihood(n2(904)))
        .expect("first instrumented N3 fit");
    let first_fit_calls = EQUATION_CALLS.load(Ordering::SeqCst);
    assert!(first_fit_calls > 0);
    let first_tables = first.tables(0.0, 0.0).expect("first tables");
    for _ in 0..10 {
        let _ = first.information_criteria();
        let _ = (
            first.aic(),
            first.bic(),
            first.aic_mcse(),
            first.bic_mcse(),
            first.summary(),
            first.population_summary(),
        );
    }
    let first_directory = temp_dir("isolation-first");
    first
        .write_outputs(&first_directory, 0.0, 0.0)
        .expect("write first complete outputs");
    let first_record = ParametricResultRecord::read_json(first_directory.join("result.json"))
        .expect("read first schema-six output");
    let _ = (
        &first_record.information_criteria,
        &first_record.tables.information_criteria,
    );
    assert_eq!(EQUATION_CALLS.load(Ordering::SeqCst), first_fit_calls);

    EQUATION_CALLS.store(0, Ordering::SeqCst);
    let second = instrumented_exact_problem()
        .fit_with(config(94).marginal_likelihood(n2(904)))
        .expect("repeated instrumented N3 fit");
    let second_fit_calls = EQUATION_CALLS.load(Ordering::SeqCst);
    assert_eq!(second_fit_calls, first_fit_calls);
    assert!(second_fit_calls > 0);
    let second_tables = second.tables(0.0, 0.0).expect("second tables");
    let second_directory = temp_dir("isolation-second");
    second
        .write_outputs(&second_directory, 0.0, 0.0)
        .expect("write second complete outputs");
    let second_record = ParametricResultRecord::read_json(second_directory.join("result.json"))
        .expect("read second schema-six output");
    let _ = &second_record.information_criteria;
    assert_eq!(EQUATION_CALLS.load(Ordering::SeqCst), second_fit_calls);

    assert_eq!(
        second.conditional_n2ll().to_bits(),
        first.conditional_n2ll().to_bits()
    );
    assert_eq!(
        second.population_parameters().len(),
        first.population_parameters().len()
    );
    for (second_estimate, first_estimate) in second
        .population_parameters()
        .iter()
        .zip(first.population_parameters())
    {
        assert_eq!(second_estimate.to_bits(), first_estimate.to_bits());
    }
    assert_eq!(
        serde_json::to_vec(second.cycle_diagnostics()).unwrap(),
        serde_json::to_vec(first.cycle_diagnostics()).unwrap()
    );
    assert_eq!(
        serde_json::to_vec(&second.marginal_likelihood_diagnostics()).unwrap(),
        serde_json::to_vec(&first.marginal_likelihood_diagnostics()).unwrap()
    );
    assert_eq!(
        serde_json::to_vec(second.information_criteria()).unwrap(),
        serde_json::to_vec(first.information_criteria()).unwrap()
    );
    assert_eq!(
        serde_json::to_vec(&second_tables).unwrap(),
        serde_json::to_vec(&first_tables).unwrap()
    );
    assert_eq!(
        fs::read(second_directory.join("result.json")).unwrap(),
        fs::read(first_directory.join("result.json")).unwrap()
    );
    for file in [
        "population.csv",
        "omega.csv",
        "residual_error.csv",
        "individual_effects.csv",
        "individual_parameters.csv",
        "iterations.csv",
        "statistics.csv",
        "marginal_likelihood.csv",
        "information_criteria.csv",
        "predictions.csv",
        "covariate_effects.csv",
        "subject_covariates.csv",
        "subject_population_parameters.csv",
        "manifest.json",
    ] {
        assert_eq!(
            fs::read(second_directory.join(file)).unwrap(),
            fs::read(first_directory.join(file)).unwrap(),
            "deterministic output mismatch in {file}"
        );
    }
    fs::remove_dir_all(first_directory).expect("remove first isolation directory");
    fs::remove_dir_all(second_directory).expect("remove second isolation directory");
}

#[test]
fn schema_six_outputs_validate_and_reject_tampering_and_old_schemas() {
    let result = mixed_problem()
        .fit_with(config(95).marginal_likelihood(n2(905)))
        .expect("schema N3 fit");
    let directory = temp_dir("schema");
    result
        .write_outputs(&directory, 0.0, 0.0)
        .expect("write schema-six outputs");
    let result_path = directory.join("result.json");
    let record = ParametricResultRecord::read_json(&result_path).expect("load schema six");
    assert_eq!(record.schema_version, 9);
    assert_eq!(
        record.information_criteria.status,
        result.information_criteria().status
    );
    assert_eq!(
        record.information_criteria.parameter_count,
        result.information_criteria().parameter_count
    );
    let persisted_bic = record.information_criteria.bic.unwrap();
    let in_memory_bic = result.information_criteria().bic.unwrap();
    assert!(
        (persisted_bic - in_memory_bic).abs()
            <= 64.0 * f64::EPSILON * persisted_bic.abs().max(in_memory_bic.abs()).max(1.0)
    );
    assert_eq!(record.tables.information_criteria.len(), 1);
    let csv = fs::read_to_string(directory.join("information_criteria.csv"))
        .expect("read information criteria CSV");
    let mut reader = csv::Reader::from_reader(csv.as_bytes());
    let headers = reader
        .headers()
        .expect("information criteria headers")
        .clone();
    let rows = reader
        .records()
        .collect::<Result<Vec<_>, _>>()
        .expect("parse information criteria CSV");
    assert_eq!(rows.len(), 1);
    let column = |name: &str| headers.iter().position(|header| header == name).unwrap();
    assert_eq!(
        &rows[0][column("status")],
        record.tables.information_criteria[0].status
    );
    assert_eq!(
        rows[0][column("free_parameter_count")]
            .parse::<usize>()
            .unwrap(),
        record.tables.information_criteria[0].free_parameter_count
    );
    let csv_bic = rows[0][column("bic")].parse::<f64>().unwrap();
    let json_bic = record.tables.information_criteria[0].bic.unwrap();
    assert!(
        (csv_bic - json_bic).abs()
            <= 64.0 * f64::EPSILON * csv_bic.abs().max(json_bic.abs()).max(1.0)
    );
    let manifest: serde_json::Value =
        serde_json::from_reader(fs::File::open(directory.join("manifest.json")).expect("manifest"))
            .expect("parse manifest");
    assert_eq!(manifest["schema_version"], 9);
    assert!(manifest["files"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!("information_criteria.csv")));

    let original: serde_json::Value =
        serde_json::from_reader(fs::File::open(&result_path).unwrap()).unwrap();
    assert_eq!(record.source_metadata.parameters.len(), 3);
    assert_eq!(record.source_metadata.random_effects.len(), 2);
    assert_eq!(record.source_metadata.omega.dimension, 2);
    assert_eq!(record.source_metadata.iov_effects.len(), 2);
    assert_eq!(
        record.source_metadata.omega_iov.as_ref().unwrap().dimension,
        2
    );
    assert_eq!(record.source_metadata.residual_outputs.len(), 2);

    let mut missing = original.clone();
    missing
        .as_object_mut()
        .unwrap()
        .remove("information_criteria");
    fs::write(&result_path, serde_json::to_vec_pretty(&missing).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&result_path).is_err());

    let mut tampered = original.clone();
    tampered["information_criteria"]["aic"] = serde_json::json!(123.0);
    fs::write(&result_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
    assert!(ParametricResultRecord::read_json(&result_path).is_err());

    let reject = |value: &serde_json::Value| {
        fs::write(&result_path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
        assert!(ParametricResultRecord::read_json(&result_path).is_err());
    };

    // Coordinated final/table/statistic changes still cannot alter immutable
    // fixed declarations captured before cycle one.
    let mut fixed_theta = original.clone();
    let changed_theta = fixed_theta["source_metadata"]["parameters"][1]["estimate"]
        .as_f64()
        .unwrap()
        + 1.0;
    fixed_theta["source_metadata"]["parameters"][1]["estimate"] = serde_json::json!(changed_theta);
    fixed_theta["tables"]["population"][1]["estimate"] = serde_json::json!(changed_theta);
    for row in fixed_theta["tables"]["statistics"].as_array_mut().unwrap() {
        if row["kind"] == "theta" && row["name"] == "v" {
            row["value"] = serde_json::json!(changed_theta);
        }
    }
    reject(&fixed_theta);

    let mut fixed_omega = original.clone();
    let changed_omega = fixed_omega["source_metadata"]["omega"]["values"][1][1]
        .as_f64()
        .unwrap()
        + 0.01;
    fixed_omega["source_metadata"]["omega"]["values"][1][1] = serde_json::json!(changed_omega);
    fixed_omega["tables"]["omega"][2]["estimate"] = serde_json::json!(changed_omega);
    for row in fixed_omega["tables"]["statistics"].as_array_mut().unwrap() {
        if row["kind"] == "omega" && row["row"] == "v" && row["column"] == "v" {
            row["value"] = serde_json::json!(changed_omega);
        }
    }
    reject(&fixed_omega);

    let mut fixed_omega_iov = original.clone();
    let changed_omega_iov = fixed_omega_iov["source_metadata"]["omega_iov"]["values"][1][1]
        .as_f64()
        .unwrap()
        + 0.01;
    fixed_omega_iov["source_metadata"]["omega_iov"]["values"][1][1] =
        serde_json::json!(changed_omega_iov);
    fixed_omega_iov["tables"]["omega_iov"][2]["estimate"] = serde_json::json!(changed_omega_iov);
    for row in fixed_omega_iov["tables"]["statistics"]
        .as_array_mut()
        .unwrap()
    {
        if row["kind"] == "omega_iov" && row["row"] == "v" && row["column"] == "v" {
            row["value"] = serde_json::json!(changed_omega_iov);
        }
    }
    reject(&fixed_omega_iov);

    let mut fixed_combined_residual = original.clone();
    let changed_proportional = fixed_combined_residual["source_metadata"]["residual_outputs"][0]
        ["values"][1]
        .as_f64()
        .unwrap()
        + 0.01;
    fixed_combined_residual["source_metadata"]["residual_outputs"][0]["values"][1] =
        serde_json::json!(changed_proportional);
    fixed_combined_residual["tables"]["residual_error"][1]["estimate"] =
        serde_json::json!(changed_proportional);
    for row in fixed_combined_residual["tables"]["statistics"]
        .as_array_mut()
        .unwrap()
    {
        if row["kind"] == "residual" && row["name"] == "cp" && row["component"] == "proportional" {
            row["value"] = serde_json::json!(changed_proportional);
        }
    }
    reject(&fixed_combined_residual);

    for field in ["initial_values", "initial_estimated_mask"] {
        let mut missing_residual_initial = original.clone();
        missing_residual_initial["source_metadata"]["residual_outputs"][0]
            .as_object_mut()
            .unwrap()
            .remove(field);
        reject(&missing_residual_initial);
    }

    let mut non_spd_initial = original.clone();
    non_spd_initial["source_metadata"]["omega"]["initial_values"][0][1] = serde_json::json!(10.0);
    non_spd_initial["source_metadata"]["omega"]["initial_values"][1][0] = serde_json::json!(10.0);
    reject(&non_spd_initial);

    let mut nonzero_initial_structural_zero = original.clone();
    for (row, column) in [(0, 1), (1, 0)] {
        nonzero_initial_structural_zero["source_metadata"]["omega"]["structural_mask"][row]
            [column] = serde_json::json!(false);
        nonzero_initial_structural_zero["source_metadata"]["omega"]["estimated_mask"][row]
            [column] = serde_json::json!(false);
        nonzero_initial_structural_zero["source_metadata"]["omega"]["values"][row][column] =
            serde_json::json!(0.0);
    }
    nonzero_initial_structural_zero["tables"]["omega"][1]["structural"] = serde_json::json!(false);
    nonzero_initial_structural_zero["tables"]["omega"][1]["estimated"] = serde_json::json!(false);
    nonzero_initial_structural_zero["tables"]["omega"][1]["estimate"] = serde_json::json!(0.0);
    reject(&nonzero_initial_structural_zero);

    let mut missing_source = original.clone();
    missing_source
        .as_object_mut()
        .unwrap()
        .remove("source_metadata");
    reject(&missing_source);

    let mut malformed_source = original.clone();
    malformed_source["source_metadata"]["omega"]["estimated_mask"][0]
        .as_array_mut()
        .unwrap()
        .pop();
    reject(&malformed_source);

    let mut asymmetric_source = original.clone();
    asymmetric_source["source_metadata"]["omega"]["estimated_mask"][0][1] =
        serde_json::json!(false);
    asymmetric_source["source_metadata"]["omega"]["estimated_mask"][1][0] = serde_json::json!(true);
    reject(&asymmetric_source);

    let mut unordered_source = original.clone();
    unordered_source["source_metadata"]["random_effects"][1]["parameter_index"] =
        serde_json::json!(0);
    reject(&unordered_source);

    let mut inconsistent_iov_source = original.clone();
    inconsistent_iov_source["source_metadata"]["omega_iov"] = serde_json::Value::Null;
    reject(&inconsistent_iov_source);

    let mut malformed_residual_source = original.clone();
    malformed_residual_source["source_metadata"]["residual_outputs"][0]["estimated_mask"] =
        serde_json::json!([]);
    reject(&malformed_residual_source);

    for pointer in [
        "/source_metadata/omega/values",
        "/source_metadata/omega/names",
        "/source_metadata/residual_outputs/0/values",
    ] {
        let mut missing_value_snapshot = original.clone();
        let (parent, field) = pointer.rsplit_once('/').unwrap();
        missing_value_snapshot
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(field);
        reject(&missing_value_snapshot);
    }

    let mut nonsymmetric_covariance = original.clone();
    nonsymmetric_covariance["source_metadata"]["omega"]["values"][0][1] =
        serde_json::json!(0.012345);
    reject(&nonsymmetric_covariance);

    let mut non_spd_covariance = original.clone();
    non_spd_covariance["source_metadata"]["omega"]["values"][0][1] = serde_json::json!(10.0);
    non_spd_covariance["source_metadata"]["omega"]["values"][1][0] = serde_json::json!(10.0);
    reject(&non_spd_covariance);

    let mut residual_value_mismatch = original.clone();
    residual_value_mismatch["source_metadata"]["residual_outputs"][0]["values"][0] =
        serde_json::json!(0.123456);
    reject(&residual_value_mismatch);

    // Finite coordinated table/statistic tampering cannot replace the
    // independently bound covariance source snapshot.
    let mut table_and_statistics = original.clone();
    let changed = table_and_statistics["tables"]["omega"][0]["estimate"]
        .as_f64()
        .unwrap()
        + 0.001;
    table_and_statistics["tables"]["omega"][0]["estimate"] = serde_json::json!(changed);
    for row in table_and_statistics["tables"]["statistics"]
        .as_array_mut()
        .unwrap()
    {
        if row["kind"] == "omega" && row["row"] == "ke" && row["column"] == "ke" {
            row["value"] = serde_json::json!(changed);
        }
    }
    reject(&table_and_statistics);

    // Conversely, coordinated source/statistic tampering cannot replace the
    // unchanged structured covariance table.
    let mut source_and_statistics = original.clone();
    source_and_statistics["source_metadata"]["omega"]["values"][0][0] = serde_json::json!(changed);
    for row in source_and_statistics["tables"]["statistics"]
        .as_array_mut()
        .unwrap()
    {
        if row["kind"] == "omega" && row["row"] == "ke" && row["column"] == "ke" {
            row["value"] = serde_json::json!(changed);
        }
    }
    reject(&source_and_statistics);

    // Coordinated mutable population derivatives are changed together: the
    // population free flag moves to a fixed covariance diagonal, coordinates
    // and criteria count families follow, while canonical source metadata is
    // deliberately untouched.
    let mut coordinated_population = original.clone();
    coordinated_population["tables"]["population"][0]["estimated"] = serde_json::json!(false);
    coordinated_population["tables"]["omega"][2]["estimated"] = serde_json::json!(true);
    let coordinates = coordinated_population["information_diagnostics"]["coordinates"]
        .as_array_mut()
        .unwrap();
    coordinates.remove(0);
    coordinates.insert(
        2,
        serde_json::json!({
            "index": 2,
            "name": "omega:v:v",
            "kind": {"Omega": {"row": 1, "column": 1}}
        }),
    );
    for (index, coordinate) in coordinates.iter_mut().enumerate() {
        coordinate["index"] = serde_json::json!(index);
    }
    coordinated_population["information_criteria"]["parameter_count"]["population"] =
        serde_json::json!(0);
    coordinated_population["information_criteria"]["parameter_count"]["omega"] =
        serde_json::json!(3);
    coordinated_population["tables"]["information_criteria"][0]["population_parameter_count"] =
        serde_json::json!(0);
    coordinated_population["tables"]["information_criteria"][0]["omega_parameter_count"] =
        serde_json::json!(3);
    for row in coordinated_population["tables"]["statistics"]
        .as_array_mut()
        .unwrap()
    {
        match row["name"].as_str() {
            Some("population_parameter_count") => row["value"] = serde_json::json!(0.0),
            Some("omega_parameter_count") => row["value"] = serde_json::json!(3.0),
            _ => {}
        }
    }
    reject(&coordinated_population);

    // Covariance structure/free status and every mutable coordinate derivative
    // move together; the independent source masks still reject the record.
    let mut coordinated_covariance = original.clone();
    coordinated_covariance["tables"]["omega"][1]["structural"] = serde_json::json!(false);
    coordinated_covariance["tables"]["omega"][1]["estimated"] = serde_json::json!(false);
    coordinated_covariance["tables"]["omega"][1]["estimate"] = serde_json::json!(0.0);
    coordinated_covariance["tables"]["omega"][2]["estimated"] = serde_json::json!(true);
    coordinated_covariance["information_diagnostics"]["coordinates"][2] = serde_json::json!({
        "index": 2,
        "name": "omega:v:v",
        "kind": {"Omega": {"row": 1, "column": 1}}
    });
    replace_json_string(
        &mut coordinated_covariance["tables"]["statistics"],
        "omega:v:ke",
        "omega:v:v",
    );
    reject(&coordinated_covariance);

    // Residual family/component declarations and all mutable labels are
    // synchronized, but the untouched canonical family/component mask wins.
    let mut coordinated_residual = original.clone();
    coordinated_residual["tables"]["residual_error"]
        .as_array_mut()
        .unwrap()
        .remove(1);
    coordinated_residual["tables"]["residual_error"][0]["family"] = serde_json::json!("constant");
    coordinated_residual["tables"]["residual_error"][0]["component"] = serde_json::json!("sigma");
    coordinated_residual["information_diagnostics"]["coordinates"][4]["name"] =
        serde_json::json!("residual:cp:sigma");
    coordinated_residual["information_diagnostics"]["coordinates"][4]["kind"]["Residual"]
        ["component"] = serde_json::json!("sigma");
    replace_json_string(
        &mut coordinated_residual["tables"]["statistics"],
        "additive",
        "sigma",
    );
    replace_json_string(
        &mut coordinated_residual["tables"]["statistics"],
        "residual:cp:additive",
        "residual:cp:sigma",
    );
    reject(&coordinated_residual);

    let mut coordinate = original.clone();
    coordinate["information_diagnostics"]["coordinates"][0]["index"] = serde_json::json!(7);
    reject(&coordinate);

    let mut duplicate_population = original.clone();
    duplicate_population["tables"]["population"][1]["name"] =
        duplicate_population["tables"]["population"][0]["name"].clone();
    reject(&duplicate_population);

    let mut reordered_population = original.clone();
    reordered_population["tables"]["population"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    reject(&reordered_population);

    let mut upper_triangle = original.clone();
    let lower_row = upper_triangle["tables"]["omega"][1]["row"].clone();
    let lower_column = upper_triangle["tables"]["omega"][1]["column"].clone();
    upper_triangle["tables"]["omega"][1]["row"] = lower_column;
    upper_triangle["tables"]["omega"][1]["column"] = lower_row;
    reject(&upper_triangle);

    let mut duplicate_covariance = original.clone();
    duplicate_covariance["tables"]["omega"][1] = duplicate_covariance["tables"]["omega"][0].clone();
    reject(&duplicate_covariance);

    let mut fixed_source = original.clone();
    fixed_source["tables"]["population"][0]["estimated"] = serde_json::json!(false);
    reject(&fixed_source);

    let mut free_structural_zero = original.clone();
    free_structural_zero["tables"]["omega"][1]["structural"] = serde_json::json!(false);
    free_structural_zero["tables"]["omega"][1]["estimated"] = serde_json::json!(true);
    reject(&free_structural_zero);

    let mut unknown_component = original.clone();
    unknown_component["tables"]["residual_error"][0]["component"] = serde_json::json!("unknown");
    unknown_component["information_diagnostics"]["coordinates"][4]["kind"]["component"] =
        serde_json::json!("unknown");
    reject(&unknown_component);

    let mut duplicate_residual = original.clone();
    let residual = duplicate_residual["tables"]["residual_error"][0].clone();
    duplicate_residual["tables"]["residual_error"]
        .as_array_mut()
        .unwrap()
        .push(residual);
    reject(&duplicate_residual);

    for field in [
        "delta",
        "g",
        "expected_complete_hessian",
        "observed_hessian",
        "observed_information",
    ] {
        let mut malformed_shape = original.clone();
        malformed_shape["information_diagnostics"][field]
            .as_array_mut()
            .unwrap()
            .pop();
        reject(&malformed_shape);
    }

    for schema in 1..=8 {
        let mut old = original.clone();
        old["schema_version"] = serde_json::json!(schema);
        fs::write(&result_path, serde_json::to_vec_pretty(&old).unwrap()).unwrap();
        assert!(ParametricResultRecord::read_json(&result_path).is_err());
    }
    fs::remove_dir_all(directory).expect("remove N3 schema directory");
}

#[test]
fn fit_next_recomputes_criteria_from_child_configuration() {
    let parent = exact_problem()
        .fit_with(config(96).marginal_likelihood(n2(906)))
        .expect("parent N3 fit");
    let disabled = parent.fit_next(config(97)).expect("disabled child");
    assert_eq!(
        disabled.information_criteria().status,
        InformationCriteriaStatus::NotRequested
    );
    let enabled = parent
        .fit_next(config(98).marginal_likelihood(n2(908)))
        .expect("enabled child");
    assert_eq!(
        enabled.information_criteria().status,
        InformationCriteriaStatus::Available
    );
    assert_ne!(
        enabled
            .marginal_likelihood_diagnostics()
            .unwrap()
            .config
            .seed,
        parent
            .marginal_likelihood_diagnostics()
            .unwrap()
            .config
            .seed
    );
}
