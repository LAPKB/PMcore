use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use pharmsol::prelude::*;
use pmcore::prelude::*;

fn equation() -> pharmsol::equation::Analytical {
    analytical! {
        name: "saem_warm_start_fixture",
        params: [ke, volume, fraction, bio],
        states: [central],
        outputs: [cp, prop_cp, log_cp, mixed_cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            let concentration = x[central] / volume;
            y[cp] = ke + fraction * concentration;
            y[prop_cp] = concentration;
            y[log_cp] = bio * concentration;
            y[mixed_cp] = (fraction + bio) * concentration;
        },
    }
}

fn data() -> Data {
    Data::new(vec![
        Subject::builder("s1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 2.4, "cp")
            .observation(1.0, 2.1, "prop_cp")
            .observation(1.0, 1.8, "log_cp")
            .observation(1.0, 3.7, "mixed_cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 2.5, "cp")
            .observation(13.0, 2.2, "prop_cp")
            .observation(13.0, 1.9, "log_cp")
            .observation(13.0, 3.8, "mixed_cp")
            .build(),
        Subject::builder("s2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 2.2, "cp")
            .observation(1.0, 2.0, "prop_cp")
            .observation(1.0, 1.7, "log_cp")
            .observation(1.0, 3.5, "mixed_cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 2.3, "cp")
            .observation(13.0, 2.1, "prop_cp")
            .observation(13.0, 1.8, "log_cp")
            .observation(13.0, 3.6, "mixed_cp")
            .build(),
    ])
}

fn initial_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    EstimationProblem::parametric(equation(), data())
        .parameter(Parameter::real("ke").with_initial(0.2))
        .parameter(Parameter::log("volume").with_initial(20.0).fixed())
        .parameter(Parameter::logit("fraction", 0.0, 1.0).with_initial(0.55))
        .parameter(
            Parameter::probit("bio", 0.0, 1.0)
                .with_initial(0.45)
                .fixed(),
        )
        .omega(
            Omega::new()
                .variance("ke", 0.04)
                .fixed_variance("volume", 0.09)
                .variance("fraction", 0.03)
                .fixed_variance("bio", 0.02)
                .covariance("ke", "volume", 0.01)
                .fixed_covariance("fraction", "bio", 0.005),
        )
        .iov(
            Iov::new()
                .fixed_variance("volume", 0.025)
                .variance("bio", 0.015)
                .covariance("volume", "bio", 0.004)
                .fixed_variance("fraction", 0.02),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.2)).fixed(),
        )
        .error_model(
            "prop_cp",
            ParametricErrorModel::new(ResidualErrorModel::proportional(0.12)),
        )
        .error_model(
            "log_cp",
            ParametricErrorModel::new(ResidualErrorModel::exponential(0.15)),
        )
        .error_model(
            "mixed_cp",
            ParametricErrorModel::new(ResidualErrorModel::combined(0.2, 0.1))
                .fixed_combined_additive(),
        )
        .build()
        .expect("warm-start fixture should build")
}

fn short_config(seed: u64) -> SaemConfig {
    SaemConfig::new()
        .seed(seed)
        .n_chains(1)
        .mcmc_iterations(1)
        .burn_in(0)
        .k1_iterations(1)
        .k2_iterations(0)
        .compute_map(false)
}

fn fitted() -> ParametricResult<pharmsol::equation::Analytical> {
    initial_problem()
        .fit_with(short_config(41))
        .expect("parent fit should complete")
}

fn averaged_fitted() -> ParametricResult<pharmsol::equation::Analytical> {
    initial_problem()
        .fit_with(
            short_config(45)
                .k1_iterations(1)
                .k2_iterations(3)
                .averaged_iterates(0.75),
        )
        .expect("averaged parent fit should complete")
}

fn fixed_parameters(
    builder: pmcore::estimation::ParametricBuilder<pharmsol::equation::Analytical>,
) -> pmcore::estimation::ParametricBuilder<pharmsol::equation::Analytical> {
    builder
        .parameter(Parameter::real("ke").with_initial(0.2).fixed())
        .parameter(Parameter::log("volume").with_initial(20.0).fixed())
        .parameter(
            Parameter::logit("fraction", 0.0, 1.0)
                .with_initial(0.55)
                .fixed(),
        )
        .parameter(
            Parameter::probit("bio", 0.0, 1.0)
                .with_initial(0.45)
                .fixed(),
        )
}

fn output_data(output: &str) -> Data {
    Data::new(vec![Subject::builder("sparse")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 3.5, output)
        .observation(2.0, 2.8, output)
        .build()])
}

fn sparse_second_output_fitted() -> ParametricResult<pharmsol::equation::Analytical> {
    fixed_parameters(EstimationProblem::parametric(
        equation(),
        output_data("prop_cp"),
    ))
    .error_model(
        "prop_cp",
        ParametricErrorModel::new(ResidualErrorModel::constant(0.2)).fixed(),
    )
    .build()
    .expect("sparse second-output problem")
    .fit_with(short_config(42))
    .expect("sparse second-output fit")
}

fn fixed_zero_combined_fitted(
    additive_zero: bool,
) -> ParametricResult<pharmsol::equation::Analytical> {
    let residual = if additive_zero {
        ParametricErrorModel::new(ResidualErrorModel::combined(0.0, 0.1)).fixed_combined_additive()
    } else {
        ParametricErrorModel::new(ResidualErrorModel::combined(0.2, 0.0))
            .fixed_combined_proportional()
    };
    fixed_parameters(EstimationProblem::parametric(
        equation(),
        output_data("mixed_cp"),
    ))
    .error_model("mixed_cp", residual)
    .build()
    .expect("fixed-zero combined problem")
    .fit_with(
        short_config(if additive_zero { 43 } else { 44 })
            .k2_iterations(2)
            .averaged_iterates(0.75),
    )
    .expect("fixed-zero combined fit")
}

fn assert_matrix_close(actual: &ndarray::Array2<f64>, expected: &ndarray::Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_roundoff(*actual, *expected);
    }
}

fn assert_roundoff(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs()
            <= 64.0 * f64::EPSILON * actual.abs().max(expected.abs()).max(1.0),
        "{actual} != {expected}"
    );
}

fn assert_residual_roundoff(actual: &ResidualErrorModel, expected: &ResidualErrorModel) {
    match (actual, expected) {
        (ResidualErrorModel::Constant { a }, ResidualErrorModel::Constant { a: expected })
        | (
            ResidualErrorModel::Proportional { b: a },
            ResidualErrorModel::Proportional { b: expected },
        )
        | (
            ResidualErrorModel::Exponential { sigma: a },
            ResidualErrorModel::Exponential { sigma: expected },
        ) => assert_roundoff(*a, *expected),
        (
            ResidualErrorModel::Combined { a, b },
            ResidualErrorModel::Combined {
                a: expected_a,
                b: expected_b,
            },
        ) => {
            assert_roundoff(*a, *expected_a);
            assert_roundoff(*b, *expected_b);
        }
        _ => panic!("warm-start residual family differs from parent"),
    }
}

fn assert_problem_matches(
    parent: &ParametricResult<pharmsol::equation::Analytical>,
    problem: &EstimationProblem<pharmsol::equation::Analytical, Parametric>,
) {
    let parameters: Vec<_> = problem.parameters().iter().collect();
    assert_eq!(parameters.len(), parent.population_parameters().len());
    for (index, parameter) in parameters.iter().enumerate() {
        assert_eq!(parameter.name, parent.parameter_names()[index]);
        assert_eq!(parameter.scale, parent.parameter_scales()[index]);
        assert_roundoff(
            parameter.initial.expect("warm-start initial is required"),
            parent.population_parameters()[index],
        );
        assert_eq!(parameter.estimate, parent.estimated_parameters()[index]);
        assert_eq!(
            parameter.random_effect,
            parent.random_effect_indices().contains(&index)
        );
    }
    assert_eq!(problem.random_effect_names(), parent.random_effect_names());
    assert_matrix_close(problem.omega(), parent.omega());
    let expected_iov_names =
        (!parent.iov_effect_names().is_empty()).then_some(parent.iov_effect_names());
    assert_eq!(problem.iov_effect_names(), expected_iov_names);
    match (problem.omega_iov(), parent.omega_iov()) {
        (Some(actual), Some(expected)) => assert_matrix_close(actual, expected),
        (None, None) => {}
        _ => panic!("warm-start IOV presence differs from parent"),
    }

    let errors = problem.residual_error_models();
    for estimate in parent.residual_error_estimates() {
        assert_eq!(
            errors.output_name(estimate.output_index),
            Some(estimate.output.as_str())
        );
        assert_residual_roundoff(errors.get(estimate.output_index).unwrap(), &estimate.model);
        assert_eq!(
            errors.is_estimated(estimate.output_index),
            estimate.estimated
        );
        let expected = [
            estimate.combined_additive_estimated.unwrap_or(false),
            estimate.combined_proportional_estimated.unwrap_or(false),
        ];
        if estimate.model.is_combined() {
            assert_eq!(
                errors.combined_component_estimated(estimate.output_index),
                expected
            );
        }
    }
}

#[test]
fn schema_six_averaged_roundtrip_and_warm_start_use_canonical_iov_multi_output_state() {
    let parent = averaged_fitted();
    assert_eq!(
        parent.config().estimator_policy,
        SaemEstimatorPolicy::AveragedIterates { alpha: 0.75 }
    );
    assert!(parent.estimator_metadata().average_applied);
    assert_eq!(parent.estimator_metadata().averaging_start_cycle, Some(2));
    assert_eq!(parent.estimator_metadata().averaged_iterations, 3);

    let directory = std::env::temp_dir().join(format!(
        "pmcore-saem-averaged-roundtrip-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    parent
        .write_outputs(&directory, 0.0, 0.0)
        .expect("write averaged outputs");
    let record = ParametricResultRecord::read_json(directory.join("result.json"))
        .expect("read averaged result");
    assert_eq!(record.schema_version, 9);
    assert_eq!(
        record.config.estimator_policy,
        parent.config().estimator_policy
    );
    assert_eq!(record.estimator_metadata, *parent.estimator_metadata());
    let manifest: serde_json::Value = serde_json::from_reader(
        fs::File::open(directory.join("manifest.json")).expect("open averaged manifest"),
    )
    .expect("parse averaged manifest");
    assert_eq!(manifest["schema_version"], 9);
    assert_eq!(
        manifest["estimator_metadata"]["policy"]["AveragedIterates"]["alpha"],
        0.75
    );
    assert_eq!(manifest["estimator_metadata"]["average_applied"], true);
    assert_eq!(manifest["estimator_metadata"]["averaging_start_cycle"], 2);
    assert_eq!(manifest["estimator_metadata"]["averaged_iterations"], 3);

    let warm = record
        .warm_start_problem(equation(), data())
        .expect("averaged persisted warm start");
    assert_problem_matches(&parent, &warm);
    fs::remove_dir_all(directory).expect("remove averaged output directory");
}

#[test]
fn live_and_json_warm_starts_preserve_typed_scientific_initialization() {
    let parent = fitted();
    let parent_tables = parent.tables(0.0, 0.0).expect("parent tables");
    assert_eq!(
        parent.parameter_scales(),
        [
            ParameterScale::Identity,
            ParameterScale::Log,
            ParameterScale::Logit {
                lower: 0.0,
                upper: 1.0,
            },
            ParameterScale::Probit {
                lower: 0.0,
                upper: 1.0,
            },
        ]
    );

    let live = parent.warm_start_problem().expect("live warm start");
    assert_problem_matches(&parent, &live);
    assert_eq!(
        parent.tables(0.0, 0.0).expect("unchanged parent"),
        parent_tables
    );

    let path = std::env::temp_dir().join(format!(
        "pmcore-saem-warm-start-{}-{}.json",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    parent.write_json(&path, 0.0, 0.0).expect("write result");
    let record = ParametricResultRecord::read_json(&path).expect("read result");
    assert_eq!(record.tables.population, parent_tables.population);
    assert_eq!(record.tables.omega.len(), parent_tables.omega.len());
    for (actual, expected) in record.tables.omega.iter().zip(&parent_tables.omega) {
        assert_eq!(actual.row, expected.row);
        assert_eq!(actual.column, expected.column);
        assert_eq!(actual.structural, expected.structural);
        assert_eq!(actual.estimated, expected.estimated);
        assert_roundoff(actual.estimate, expected.estimate);
    }
    match (&record.tables.omega_iov, &parent_tables.omega_iov) {
        (Some(actual), Some(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.row, expected.row);
                assert_eq!(actual.column, expected.column);
                assert_eq!(actual.structural, expected.structural);
                assert_eq!(actual.estimated, expected.estimated);
                assert_roundoff(actual.estimate, expected.estimate);
            }
        }
        (None, None) => {}
        _ => panic!("persisted Omega_IOV presence changed"),
    }
    assert_eq!(
        record.tables.residual_error.len(),
        parent_tables.residual_error.len()
    );
    for (actual, expected) in record
        .tables
        .residual_error
        .iter()
        .zip(&parent_tables.residual_error)
    {
        assert_eq!(actual.output, expected.output);
        assert_eq!(actual.output_index, expected.output_index);
        assert_eq!(actual.family, expected.family);
        assert_eq!(actual.component, expected.component);
        assert_eq!(actual.estimated, expected.estimated);
        assert_roundoff(actual.estimate, expected.estimate);
    }
    let persisted = record
        .warm_start_problem(equation(), data())
        .expect("persisted warm start");
    assert_problem_matches(&parent, &persisted);
    fs::remove_file(path).expect("remove temporary result");
}

#[test]
fn sparse_second_output_warm_starts_preserve_index_and_name() {
    let parent = sparse_second_output_fitted();
    let rows = &parent.tables(0.0, 0.0).expect("tables").residual_error;
    assert_eq!(rows.len(), 1);
    assert_eq!(
        (rows[0].output_index, rows[0].output.as_str()),
        (1, "prop_cp")
    );

    let live = parent.warm_start_problem().expect("live sparse warm start");
    assert!(live.residual_error_models().get(0).is_none());
    assert_eq!(live.residual_error_models().output_name(1), Some("prop_cp"));

    let path = std::env::temp_dir().join(format!(
        "pmcore-saem-warm-start-sparse-{}.json",
        std::process::id()
    ));
    parent.write_json(&path, 0.0, 0.0).expect("write result");
    let record = ParametricResultRecord::read_json(&path).expect("read result");
    let persisted = record
        .warm_start_problem(equation(), output_data("prop_cp"))
        .expect("JSON sparse warm start");
    assert!(persisted.residual_error_models().get(0).is_none());
    assert_eq!(
        persisted.residual_error_models().output_name(1),
        Some("prop_cp")
    );
    fs::remove_file(path).expect("remove temporary result");
}

fn assert_fixed_zero_combined_warm_starts(additive_zero: bool) {
    let parent = fixed_zero_combined_fitted(additive_zero);
    assert!(parent.estimator_metadata().average_applied);
    assert_eq!(parent.estimator_metadata().averaging_start_cycle, Some(2));
    assert_eq!(parent.estimator_metadata().averaged_iterations, 2);
    let expected_mask = if additive_zero {
        [false, true]
    } else {
        [true, false]
    };
    let expected_free = parent.cycle_diagnostics()[1..]
        .iter()
        .map(|cycle| match cycle.residual_error_estimates[0].model {
            ResidualErrorModel::Combined { a, b } => {
                if additive_zero {
                    b
                } else {
                    a
                }
            }
            _ => panic!("expected combined cycle residual model"),
        })
        .sum::<f64>()
        / 2.0;
    match parent.residual_error_estimates()[0].model {
        ResidualErrorModel::Combined { a, b } => {
            assert!((if additive_zero { b } else { a } - expected_free).abs() < 1e-12);
        }
        _ => panic!("expected combined averaged residual model"),
    }
    let assert_problem = |problem: &EstimationProblem<_, Parametric>| {
        let errors = problem.residual_error_models();
        assert_eq!(errors.output_name(3), Some("mixed_cp"));
        assert_eq!(errors.combined_component_estimated(3), expected_mask);
        match errors.get(3).expect("combined model") {
            ResidualErrorModel::Combined { a, b } => {
                assert_eq!((*a == 0.0, *b == 0.0), (additive_zero, !additive_zero));
            }
            other => panic!("expected combined model, found {other:?}"),
        }
    };

    let live = parent
        .warm_start_problem()
        .expect("live fixed-zero warm start");
    assert_problem(&live);

    let path = std::env::temp_dir().join(format!(
        "pmcore-saem-warm-start-fixed-zero-{}-{}.json",
        additive_zero,
        std::process::id()
    ));
    parent.write_json(&path, 0.0, 0.0).expect("write result");
    let record = ParametricResultRecord::read_json(&path).expect("read result");
    let persisted = record
        .warm_start_problem(equation(), output_data("mixed_cp"))
        .expect("JSON fixed-zero warm start");
    assert_problem(&persisted);
    fs::remove_file(path).expect("remove temporary result");
}

#[test]
fn combined_fixed_zero_components_survive_live_and_json_warm_starts() {
    assert_fixed_zero_combined_warm_starts(true);
    assert_fixed_zero_combined_warm_starts(false);
}

#[test]
fn fit_next_uses_caller_configuration_without_mutating_parent() {
    let parent = fitted();
    let parent_tables = parent.tables(0.0, 0.0).expect("parent tables");
    let child_config = short_config(9_991).k1_iterations(2);
    let child = parent.fit_next(child_config.clone()).expect("child fit");
    assert_eq!(
        serde_json::to_value(child.config()).expect("serialize child config"),
        serde_json::to_value(&child_config).expect("serialize requested config")
    );
    assert_eq!(child.config().seed, 9_991);
    assert_eq!(parent.config().seed, 41);
    assert_eq!(
        parent.tables(0.0, 0.0).expect("unchanged parent"),
        parent_tables
    );
    assert_eq!(child.iterations(), 2);
}

#[test]
fn iov_warm_start_preserves_multidimensional_structure_and_masks() {
    let parent = fitted();
    assert_eq!(parent.iov_effect_names(), ["volume", "bio", "fraction"]);

    let tables = parent.tables(0.0, 0.0).expect("parent tables");
    let omega_masks: Vec<_> = tables
        .omega
        .iter()
        .map(|row| (row.structural, row.estimated))
        .collect();
    assert_eq!(
        omega_masks,
        [
            (true, true),
            (true, true),
            (true, false),
            (false, false),
            (false, false),
            (true, true),
            (false, false),
            (false, false),
            (true, false),
            (true, false),
        ]
    );
    let omega_iov_masks: Vec<_> = tables
        .omega_iov
        .as_ref()
        .expect("IOV table")
        .iter()
        .map(|row| (row.structural, row.estimated))
        .collect();
    assert_eq!(
        omega_iov_masks,
        [
            (true, false),
            (true, true),
            (true, true),
            (false, false),
            (false, false),
            (true, false),
        ]
    );

    let child = parent
        .warm_start_problem()
        .expect("IOV warm start")
        .fit_with(short_config(812))
        .expect("IOV child fit");
    assert_eq!(child.iov_effect_names(), parent.iov_effect_names());
    assert_eq!(
        child.omega_structural_mask(),
        parent.omega_structural_mask()
    );
    assert_eq!(child.omega_estimated_mask(), parent.omega_estimated_mask());
    assert_eq!(
        child.omega_iov_structural_mask(),
        parent.omega_iov_structural_mask()
    );
    assert_eq!(
        child.omega_iov_estimated_mask(),
        parent.omega_iov_estimated_mask()
    );
}

#[test]
fn combined_components_reconstruct_with_independent_masks() {
    let parent = fitted();
    let path = std::env::temp_dir().join(format!(
        "pmcore-saem-warm-start-combined-{}.json",
        std::process::id()
    ));
    parent.write_json(&path, 0.0, 0.0).expect("write result");
    let mut record = ParametricResultRecord::read_json(&path).expect("read result");
    let mut combined: Vec<_> = record
        .tables
        .residual_error
        .iter_mut()
        .filter(|row| row.family == "combined")
        .collect();
    assert_eq!(combined.len(), 2);
    combined[0].estimated = true;
    combined[1].estimated = false;

    assert!(record.warm_start_problem(equation(), data()).is_err());
    fs::remove_file(path).expect("remove temporary result");
}

fn assert_warm_start_rejected(record: &ParametricResultRecord) {
    assert!(record.warm_start_problem(equation(), data()).is_err());
}

#[test]
fn schemas_one_through_eight_and_missing_current_diagnostics_are_rejected() {
    let parent = fitted();
    let path = std::env::temp_dir().join(format!(
        "pmcore-saem-schema-four-required-{}.json",
        std::process::id()
    ));
    parent.write_json(&path, 0.0, 0.0).expect("write result");
    let current: serde_json::Value =
        serde_json::from_reader(fs::File::open(&path).expect("open result")).expect("parse result");

    for schema in 1..=8 {
        let mut legacy = current.clone();
        legacy["schema_version"] = serde_json::json!(schema);
        serde_json::to_writer_pretty(fs::File::create(&path).expect("rewrite"), &legacy)
            .expect("write legacy version");
        assert!(ParametricResultRecord::read_json(&path).is_err());
    }
    for required in [
        "information_diagnostics",
        "markov_simulation_variance",
        "population_uncertainty",
        "conditional_modes",
        "shrinkage",
    ] {
        let mut missing = current.clone();
        missing
            .as_object_mut()
            .expect("record object")
            .remove(required);
        serde_json::to_writer_pretty(fs::File::create(&path).expect("rewrite"), &missing)
            .expect("write missing field");
        assert!(ParametricResultRecord::read_json(&path).is_err());
    }
    fs::remove_file(path).expect("remove temporary result");
}

#[test]
fn malformed_persisted_headers_and_tables_are_rejected() {
    let parent = fitted();
    let path = std::env::temp_dir().join(format!(
        "pmcore-saem-warm-start-malformed-{}.json",
        std::process::id()
    ));
    parent.write_json(&path, 0.0, 0.0).expect("write result");
    let record = ParametricResultRecord::read_json(&path).expect("read result");

    for schema_version in 1..=8 {
        let mut bad = record.clone();
        bad.schema_version = schema_version;
        assert_warm_start_rejected(&bad);
    }
    let mut bad = record.clone();
    bad.fit_family = "nonparametric".to_string();
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.algorithm = "other".to_string();
    assert_warm_start_rejected(&bad);

    let mut bad = record.clone();
    bad.tables.population[1].name = bad.tables.population[0].name.clone();
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.population[0].scale = "logit(1,1)".to_string();
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.omega.swap(0, 1);
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.omega[0].estimate = f64::NAN;
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.omega[3].structural = false;
    bad.tables.omega[3].estimated = true;
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.omega_iov.as_mut().expect("IOV table").swap(0, 1);
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.omega_iov.as_mut().expect("IOV table")[3].estimate = 0.01;
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.residual_error[0].family = "unknown".to_string();
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.residual_error[1].output_index = 7;
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.residual_error[1].output = "cp".to_string();
    assert_warm_start_rejected(&bad);

    let combined_start = record.tables.residual_error.len() - 2;
    let mut bad = record.clone();
    bad.tables.residual_error[combined_start].estimate = 0.0;
    bad.tables.residual_error[combined_start].estimated = true;
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.residual_error[combined_start].estimate = 0.0;
    bad.tables.residual_error[combined_start].estimated = false;
    bad.tables.residual_error[combined_start + 1].estimate = 0.0;
    bad.tables.residual_error[combined_start + 1].estimated = false;
    assert_warm_start_rejected(&bad);
    let mut bad = record.clone();
    bad.tables.residual_error[combined_start].estimate = -0.1;
    bad.tables.residual_error[combined_start].estimated = false;
    assert_warm_start_rejected(&bad);
    let mut bad = record;
    bad.tables.residual_error[combined_start].estimate = f64::INFINITY;
    assert_warm_start_rejected(&bad);

    fs::remove_file(path).expect("remove temporary result");
}
