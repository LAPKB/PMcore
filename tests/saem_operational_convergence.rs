use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use pharmsol::prelude::*;
use pmcore::algorithms::{Status, StopReason};
use pmcore::prelude::*;
use pmcore::results::ParametricResultRecord;
use tracing::Level;
use tracing_subscriber::fmt::MakeWriter;

#[derive(Clone, Default)]
struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for LogBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("log lock").extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for LogBuffer {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn identifiable_latent_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "saem_operational_convergence_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [bolus(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };
    let data = Data::new(vec![
        Subject::builder("s1")
            .bolus(0.0, 100.0, "iv")
            .observation(0.5, 7.8, "cp")
            .observation(2.0, 5.0, "cp")
            .observation(6.0, 1.7, "cp")
            .build(),
        Subject::builder("s2")
            .bolus(0.0, 100.0, "iv")
            .observation(0.5, 7.2, "cp")
            .observation(2.0, 4.3, "cp")
            .observation(6.0, 1.2, "cp")
            .build(),
        Subject::builder("s3")
            .bolus(0.0, 100.0, "iv")
            .observation(0.5, 8.2, "cp")
            .observation(2.0, 5.5, "cp")
            .observation(6.0, 2.0, "cp")
            .build(),
        Subject::builder("s4")
            .bolus(0.0, 100.0, "iv")
            .observation(0.5, 7.5, "cp")
            .observation(2.0, 4.7, "cp")
            .observation(6.0, 1.5, "cp")
            .build(),
    ]);

    EstimationProblem::parametric(equation, data)
        .parameter(Parameter::log("ke").with_initial(0.25).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(12.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.4)).fixed(),
        )
        .build()
        .expect("identifiable analytical path-test fixture")
}

fn path_test_config() -> SaemConfig {
    // These deliberately generous fixed-width/Newton limits are declared path-test
    // thresholds before execution. They are not production recommendations. The
    // literature rank gates remain Rhat < 1.01, bulk ESS > 400, and average ESS
    // per split chain >= 50.
    let path_test_policy =
        OperationalConvergenceConfig::literature_guided(4, 20, 10.0, 0.95, 100.0, 100.0);
    let frozen_diagnostics = MarkovSimulationVarianceConfig::new(
        0x51a7_2026,
        1024,
        1152,
        96,
        LugsailConfig::over_lugsail_bartlett(),
        4,
        64 * 1024 * 1024,
    );
    SaemConfig::new()
        .seed(0x51a7_0001)
        .n_chains(2)
        .mcmc_iterations(2)
        .burn_in(8)
        .k1_iterations(8)
        .k2_iterations(8)
        .averaged_iterates(0.75)
        .markov_simulation_variance(frozen_diagnostics)
        .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
        .operational_convergence(path_test_policy)
        .compute_map(false)
}

#[test]
fn genuine_operational_pass_sets_converged() {
    let writer = LogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_max_level(Level::INFO)
        .with_writer(writer.clone())
        .finish();
    let mut snapshots = Vec::new();
    let result = tracing::subscriber::with_default(subscriber, || {
        identifiable_latent_problem().fit_with_observer(
            path_test_config(),
            |controller: &ParametricFitController<_>| {
                snapshots.push(controller.snapshot());
                ParametricCycleFlow::Continue
            },
        )
    })
    .expect("operational path-test fit");
    let diagnostics = result.operational_diagnostics();
    let final_check = diagnostics.checks.last().expect("operational checkpoint");
    assert_eq!(
        result.termination_reason(),
        Some(&StopReason::Converged),
        "outcome={:?}; criteria={:?}; information={:?}; markov={:?}",
        final_check.outcome,
        final_check.criteria,
        final_check.information.as_ref().map(|value| &value.status),
        final_check.markov.as_ref().map(|value| (
            &value.status,
            &value.lambda_status,
            &value.xi_status,
            &value.simulation_covariance_status
        ))
    );
    assert!(diagnostics.used_for_termination);
    assert!(matches!(
        final_check.outcome,
        OperationalConvergenceOutcome::Passed
    ));
    assert!(final_check.worst_rhat().expect("worst Rhat") < 1.01);
    assert!(final_check.min_bulk_ess().expect("minimum bulk ESS") > 400.0);
    assert!(diagnostics.warnings()[0].contains("PMcore operational convergence criteria passed"));
    assert!(diagnostics.warnings()[0].contains("not proof of mathematical convergence"));

    let final_snapshot = snapshots.last().expect("observer final snapshot");
    assert_eq!(final_snapshot.status, Status::Stop(StopReason::Converged));
    assert_eq!(final_snapshot.cycle, result.iterations());
    let logs = String::from_utf8(writer.0.lock().expect("log lock").clone())
        .expect("UTF-8 tracing output");
    assert!(
        logs.contains("PMcore operational convergence criteria passed"),
        "{logs}"
    );
    assert!(
        logs.contains("does not prove mathematical convergence"),
        "{logs}"
    );

    let tables = result.tables(0.0, 0.0).expect("path-test tables");
    let outcome_row = tables
        .statistics
        .iter()
        .find(|row| row.kind == "operational_convergence_outcome" && row.name.ends_with(":outcome"))
        .expect("statistics operational outcome");
    assert_eq!(outcome_row.value, Some(1.0));
    let count_kind = |kind: &str| {
        tables
            .statistics
            .iter()
            .filter(|row| row.kind == kind)
            .count()
    };
    assert!(result
        .cycle_diagnostics()
        .iter()
        .all(|cycle| cycle.omega_relative_spd_margin.is_some()));
    assert_eq!(
        count_kind("covariance_stability"),
        result.cycle_diagnostics().len()
    );
    assert_eq!(
        count_kind("operational_convergence_criterion_status"),
        final_check.criteria.len()
    );
    assert_eq!(
        count_kind("operational_convergence_trace_status"),
        final_check.per_trace_diagnostics().len() * 5
    );
    assert_eq!(count_kind("operational_convergence_lrv_chain_status"), 4);
    assert_eq!(
        count_kind("operational_convergence_lrv_aggregate_status"),
        2
    );
    assert_eq!(count_kind("operational_convergence_matrix_status"), 3);

    let json_path = std::env::temp_dir().join(format!(
        "pmcore-operational-convergence-{}.json",
        std::process::id()
    ));
    result
        .write_json(&json_path, 0.0, 0.0)
        .expect("write path-test JSON");
    let record = ParametricResultRecord::read_json(&json_path).expect("read path-test JSON");
    std::fs::remove_file(&json_path).expect("remove path-test JSON");
    assert_eq!(record.termination, Some(StopReason::Converged));
    assert_eq!(record.operational_convergence.config, diagnostics.config);
    assert_eq!(
        record.operational_convergence.final_status,
        diagnostics.final_status
    );
    assert_eq!(
        record.operational_convergence.used_for_termination,
        diagnostics.used_for_termination
    );
    assert_eq!(
        record.operational_convergence.checks.len(),
        diagnostics.checks.len()
    );
}

fn failed_path_test_config() -> SaemConfig {
    let mut config = path_test_config();
    config.operational_convergence = Some(OperationalConvergenceConfig::literature_guided(
        4, 20, 1.0e-6, 0.95, 100.0, 100.0,
    ));
    config
}

#[test]
fn valid_diagnostics_that_miss_predeclared_precision_end_max_cycles() {
    let result = identifiable_latent_problem()
        .fit_with(failed_path_test_config())
        .expect("failed operational control");
    let diagnostics = result.operational_diagnostics();
    let check = diagnostics.checks.last().expect("failed checkpoint");
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
    assert!(matches!(
        check.outcome,
        OperationalConvergenceOutcome::Failed { .. }
    ));
    assert!(check.criteria.iter().all(|criterion| !matches!(
        criterion.status,
        OperationalConvergenceCriterionStatus::Unavailable(_)
    )));
    let fixed_width = check
        .criteria
        .iter()
        .find(|criterion| criterion.name == "relative_fixed_width")
        .expect("fixed-width criterion");
    assert!(matches!(
        fixed_width.status,
        OperationalConvergenceCriterionStatus::NotSatisfied
    ));
    assert!(diagnostics.warnings()[0].contains("evaluated but not satisfied"));
}

#[test]
fn operational_diagnostic_rng_does_not_change_fit_trajectory_or_results() {
    let checked = identifiable_latent_problem()
        .fit_with(failed_path_test_config())
        .expect("checked fit");
    let mut unchecked_config = failed_path_test_config();
    unchecked_config.operational_convergence = None;
    let unchecked = identifiable_latent_problem()
        .fit_with(unchecked_config)
        .expect("otherwise-identical unchecked fit");

    assert_eq!(checked.termination_reason(), Some(&StopReason::MaxCycles));
    assert_eq!(unchecked.termination_reason(), Some(&StopReason::MaxCycles));
    assert_eq!(checked.cycle_diagnostics(), unchecked.cycle_diagnostics());
    assert_eq!(
        checked.population_parameters(),
        unchecked.population_parameters()
    );
    assert_eq!(checked.omega(), unchecked.omega());
    assert_eq!(checked.omega_iov(), unchecked.omega_iov());
    assert_eq!(checked.residual_sigmas(), unchecked.residual_sigmas());
    assert_eq!(checked.conditional_n2ll(), unchecked.conditional_n2ll());
    assert_eq!(checked.eta_chain_means(), unchecked.eta_chain_means());
    assert_eq!(checked.kappa_chain_means(), unchecked.kappa_chain_means());
    assert_eq!(checked.conditional_modes(), unchecked.conditional_modes());
    assert_eq!(
        checked
            .tables(0.0, 0.0)
            .expect("checked tables")
            .predictions,
        unchecked
            .tables(0.0, 0.0)
            .expect("unchecked tables")
            .predictions
    );
}

fn no_latent_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "saem_operational_no_latent_control",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [bolus(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
    };
    let data = Data::new(vec![Subject::builder("fixed")
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 6.5, "cp")
        .build()]);
    EstimationProblem::parametric(equation, data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.25)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(12.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.4)).fixed(),
        )
        .build()
        .expect("no-latent control")
}

#[test]
fn no_latent_control_is_ineligible_and_ends_max_cycles() {
    let result = no_latent_problem()
        .fit_with(path_test_config())
        .expect("ineligible operational control");
    let diagnostics = result.operational_diagnostics();
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
    assert!(matches!(
        diagnostics.final_status,
        Some(OperationalConvergenceOutcome::Ineligible { .. })
    ));
    assert!(diagnostics.warnings()[0].contains("evaluated but were ineligible"));
}

#[test]
fn configured_but_aborted_before_a_checkpoint_is_neutral() {
    let result = identifiable_latent_problem()
        .fit_with_observer(
            path_test_config(),
            |_controller: &ParametricFitController<_>| ParametricCycleFlow::Stop,
        )
        .expect("pre-checkpoint abort result");
    let diagnostics = result.operational_diagnostics();
    assert_eq!(result.termination_reason(), Some(&StopReason::Aborted));
    assert!(diagnostics.checks.is_empty());
    assert!(diagnostics.final_status.is_none());
    let warning = &diagnostics.warnings()[0];
    assert!(warning.contains("no checkpoint was evaluated"));
    assert!(warning.contains("not evaluated or established"));
    assert!(!warning.contains("failed"));
    assert!(!warning.contains("ineligible"));
}
