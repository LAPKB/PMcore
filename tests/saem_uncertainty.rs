use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use pharmsol::prelude::*;
use pmcore::prelude::*;

fn analytical_problem(
    with_iov: bool,
) -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "n6_joint_uncertainty_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let subjects = vec![
        Subject::builder("n6-1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 4.9, "cp")
            .observation(3.0, 3.1, "cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 4.7, "cp")
            .observation(15.0, 2.9, "cp")
            .build(),
        Subject::builder("n6-2")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(1.0, 5.2, "cp")
            .observation(3.0, 3.5, "cp")
            .reset()
            .infusion(12.0, 100.0, "iv", 0.5)
            .observation(13.0, 5.0, "cp")
            .observation(15.0, 3.2, "cp")
            .build(),
    ];
    let problem = EstimationProblem::parametric(equation, Data::new(subjects))
        .parameter(Parameter::log("ke").with_initial(0.25).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new().fixed_variance("ke", 0.09))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.35)).fixed(),
        );
    let problem = if with_iov {
        problem.iov(Iov::new().fixed_variance("ke", 0.04))
    } else {
        problem
    };
    problem.build().expect("N6 analytical fixture")
}

fn fit_config() -> SaemConfig {
    SaemConfig::new()
        .seed(0x6e36_2026)
        .n_chains(2)
        .mcmc_iterations(1)
        .burn_in(0)
        .k1_iterations(2)
        .k2_iterations(1)
        .compute_map(true)
}

fn n2(curvature: bool) -> MarginalLikelihoodConfig {
    let config = MarginalLikelihoodConfig::new(128, 0x6e32_2026, 5, 1.5);
    if curvature {
        config.conditional_mode_curvature_proposal()
    } else {
        config
    }
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * left.abs().max(right.abs()).max(1.0)
}

fn available_shrinkage(value: &ShrinkageValue, expected_count: usize) {
    match value {
        ShrinkageValue::Available {
            value, unit_count, ..
        } => {
            assert!(value.is_finite());
            assert_eq!(*unit_count, expected_count);
        }
        ShrinkageValue::Unavailable { reason } => {
            panic!("expected available shrinkage, got {reason:?}")
        }
    }
}

#[test]
fn analytical_iiv_conditional_uncertainty_and_fixed_summary_are_wired() {
    let result = analytical_problem(false).fit_with(fit_config()).unwrap();
    assert_eq!(result.conditional_modes().len(), 2);
    for mode in result.conditional_modes() {
        assert!(mode.uncertainty.mode_metadata.objective_value.is_finite());
        assert_eq!(mode.uncertainty.coordinates.len(), 1);
        assert!(matches!(
            mode.uncertainty.coordinates[0].kind,
            JointLatentCoordinateKind::Eta { parameter_index: 0 }
        ));
        assert_eq!(
            mode.uncertainty.status,
            ConditionalCurvatureStatus::Available
        );
        assert_eq!(
            mode.uncertainty.regularization,
            ConditionalCurvatureRegularization::None
        );
    }
    available_shrinkage(&result.shrinkage().eta_posterior_mean[0].shrinkage, 2);
    available_shrinkage(&result.shrinkage().eta_map[0].shrinkage, 2);

    let summary = result.population_summary();
    assert!(summary
        .parameters
        .iter()
        .all(|parameter| { parameter.sd.is_none() && parameter.cv_percent.is_none() }));
    assert!(result
        .individual_summaries()
        .iter()
        .all(|summary| summary.conditional_uncertainty.is_some()));
}

#[test]
fn analytical_two_occasion_iov_curvature_proposal_reuses_joint_covariance_and_roundtrips() {
    let default = analytical_problem(true)
        .fit_with(fit_config().marginal_likelihood(n2(false)))
        .unwrap();
    let curvature = analytical_problem(true)
        .fit_with(fit_config().marginal_likelihood(n2(true)))
        .unwrap();

    assert_eq!(
        default.population_parameters(),
        curvature.population_parameters()
    );
    assert_eq!(default.objf().to_bits(), curvature.objf().to_bits());
    assert_eq!(default.cycle_diagnostics(), curvature.cycle_diagnostics());
    assert_eq!(default.eta_chain_means(), curvature.eta_chain_means());
    assert_eq!(default.kappa_chain_means(), curvature.kappa_chain_means());

    for mode in curvature.conditional_modes() {
        assert_eq!(
            mode.uncertainty.status,
            ConditionalCurvatureStatus::Available
        );
        assert_eq!(mode.uncertainty.coordinates.len(), 3);
        assert!(matches!(
            mode.uncertainty.coordinates[0].kind,
            JointLatentCoordinateKind::Eta { parameter_index: 0 }
        ));
        for (position, coordinate) in mode.uncertainty.coordinates[1..].iter().enumerate() {
            assert!(matches!(
                coordinate.kind,
                JointLatentCoordinateKind::Kappa {
                    occasion_index,
                    effect_index: 0,
                    parameter_index: 0,
                } if occasion_index == position
            ));
        }
        let covariance = mode.uncertainty.latent_covariance.as_ref().unwrap();
        for (row, values) in covariance.iter().enumerate() {
            for (column, value) in values.iter().enumerate().take(row) {
                assert_eq!(value.to_bits(), covariance[column][row].to_bits());
            }
        }
    }

    let default_n2 = default.marginal_likelihood_diagnostics().unwrap();
    assert!(default_n2.subjects.iter().all(|subject| {
        subject.failure.is_none()
            && subject.proposal_scale_source == ProposalScaleSource::FinalRawOmegaBlocks
    }));
    let curvature_n2 = curvature.marginal_likelihood_diagnostics().unwrap();
    assert!(curvature_n2.subjects.iter().all(|subject| {
        subject.failure.is_none()
            && subject.proposal_scale_source == ProposalScaleSource::ConditionalModeCurvature
    }));

    available_shrinkage(&curvature.shrinkage().eta_posterior_mean[0].shrinkage, 2);
    available_shrinkage(&curvature.shrinkage().eta_map[0].shrinkage, 2);
    available_shrinkage(&curvature.shrinkage().kappa_posterior_mean[0].shrinkage, 4);
    available_shrinkage(&curvature.shrinkage().kappa_map[0].shrinkage, 4);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("pmcore-n6-{unique}"));
    curvature.write_outputs(&directory, 0.0, 0.0).unwrap();
    let record = ParametricResultRecord::read_json(directory.join("result.json")).unwrap();
    assert_eq!(record.schema_version, 9);
    assert_eq!(
        record.population_uncertainty.status,
        curvature.population_uncertainty().status
    );
    assert_eq!(
        record.population_uncertainty.coordinates,
        curvature.population_uncertainty().coordinates
    );
    assert_eq!(
        record.conditional_modes.len(),
        curvature.conditional_modes().len()
    );
    for (persisted, live) in record
        .conditional_modes
        .iter()
        .zip(curvature.conditional_modes())
    {
        assert_eq!(persisted.subject_id, live.subject_id);
        assert!(persisted
            .eta
            .iter()
            .zip(&live.eta)
            .all(|(left, right)| close(*left, *right)));
        assert_eq!(persisted.kappas.len(), live.kappas.len());
        for (persisted_kappa, live_kappa) in persisted.kappas.iter().zip(&live.kappas) {
            assert_eq!(persisted_kappa.subject_id, live_kappa.subject_id);
            assert_eq!(persisted_kappa.occasion_index, live_kappa.occasion_index);
            assert!(persisted_kappa
                .values
                .iter()
                .zip(&live_kappa.values)
                .all(|(left, right)| close(*left, *right)));
        }
        assert!(persisted
            .parameters
            .iter()
            .zip(&live.parameters)
            .all(|(left, right)| close(*left, *right)));
        assert!(close(persisted.objective, live.objective));
        assert_eq!(persisted.converged, live.converged);
        assert_eq!(persisted.iterations, live.iterations);
        assert_eq!(persisted.termination, live.termination);
        assert_eq!(persisted.uncertainty.status, live.uncertainty.status);
        assert_eq!(
            persisted.uncertainty.coordinates,
            live.uncertainty.coordinates
        );
        let persisted_covariance = persisted.uncertainty.latent_covariance.as_ref().unwrap();
        let live_covariance = live.uncertainty.latent_covariance.as_ref().unwrap();
        assert!(persisted_covariance
            .iter()
            .flatten()
            .zip(live_covariance.iter().flatten())
            .all(|(left, right)| close(*left, *right)));
    }
    assert_eq!(record.shrinkage.eta_posterior_mean.len(), 1);
    assert_eq!(record.shrinkage.eta_map.len(), 1);
    assert_eq!(record.shrinkage.kappa_posterior_mean.len(), 1);
    assert_eq!(record.shrinkage.kappa_map.len(), 1);
    available_shrinkage(&record.shrinkage.eta_posterior_mean[0].shrinkage, 2);
    available_shrinkage(&record.shrinkage.eta_map[0].shrinkage, 2);
    available_shrinkage(&record.shrinkage.kappa_posterior_mean[0].shrinkage, 4);
    available_shrinkage(&record.shrinkage.kappa_map[0].shrinkage, 4);
    assert_eq!(
        record
            .tables
            .statistics
            .iter()
            .filter(|row| row.kind == "conditional_curvature_status")
            .count(),
        2
    );
    assert!(record
        .tables
        .statistics
        .iter()
        .any(|row| row.kind == "population_uncertainty_status"));

    let original: serde_json::Value =
        serde_json::from_reader(fs::File::open(directory.join("result.json")).unwrap()).unwrap();
    let tampered_path = directory.join("tampered.json");

    let mut tampered_coordinate = original.clone();
    tampered_coordinate["conditional_modes"][0]["uncertainty"]["coordinates"][1]["effect_index"] =
        serde_json::json!(1);
    fs::write(
        &tampered_path,
        serde_json::to_vec_pretty(&tampered_coordinate).unwrap(),
    )
    .unwrap();
    assert!(ParametricResultRecord::read_json(&tampered_path).is_err());

    let mut tampered_shrinkage = original.clone();
    tampered_shrinkage["shrinkage"]["eta_posterior_mean"][0]["shrinkage"]["value"] =
        serde_json::json!(999.0);
    fs::write(
        &tampered_path,
        serde_json::to_vec_pretty(&tampered_shrinkage).unwrap(),
    )
    .unwrap();
    assert!(ParametricResultRecord::read_json(&tampered_path).is_err());

    let mut tampered_status = original;
    let status_row = tampered_status["tables"]["statistics"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["kind"] == "population_uncertainty_status")
        .unwrap();
    status_row["status"] = serde_json::json!("tampered");
    fs::write(
        &tampered_path,
        serde_json::to_vec_pretty(&tampered_status).unwrap(),
    )
    .unwrap();
    assert!(ParametricResultRecord::read_json(&tampered_path).is_err());

    fs::remove_dir_all(directory).unwrap();
}
