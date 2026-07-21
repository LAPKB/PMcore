use std::fs;
use std::path::Path;
use std::sync::Mutex;

use pharmsol::prelude::*;
use pmcore::algorithms::{Status, StopReason};
use pmcore::prelude::*;

static STOP_FILE_LOCK: Mutex<()> = Mutex::new(());

struct StopFileCleanup;

impl Drop for StopFileCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_file("stop");
    }
}

fn lifecycle_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    let equation = analytical! {
        name: "saem_lifecycle_fixture",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [infusion(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    };
    let data = Data::new(vec![Subject::builder("s1")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 4.0, "cp")
        .build()]);

    EstimationProblem::parametric(equation, data)
        .parameter(Parameter::log("ke").with_initial(0.3).fixed())
        .parameter(Parameter::log("v").with_initial(20.0).fixed())
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
        .expect("lifecycle fixture should build")
}

fn tiny_config(total_cycles: usize) -> SaemConfig {
    SaemConfig::new()
        .seed(7)
        .n_chains(1)
        .mcmc_iterations(1)
        .burn_in(0)
        .k1_iterations(total_cycles)
        .k2_iterations(0)
        .compute_map(false)
}

fn averaged_config() -> SaemConfig {
    SaemConfig::new()
        .seed(8)
        .n_chains(1)
        .mcmc_iterations(1)
        .burn_in(0)
        .k1_iterations(1)
        .k2_iterations(3)
        .averaged_iterates(0.75)
        .compute_map(false)
}

#[test]
fn snapshot_tracks_initial_and_completed_cycle_state() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    let mut controller = lifecycle_problem()
        .fit_controller(tiny_config(2))
        .expect("controller should build");

    let initial = controller.snapshot();
    assert_eq!(initial.cycle, 0);
    assert_eq!(initial.total_cycles, 2);
    assert_eq!(initial.status, Status::Continue);
    assert_eq!(initial.progress(), 0.0);
    assert!(!initial.is_terminal());
    assert_eq!(initial.conditional_n2ll, controller.n2ll());
    assert_eq!(
        initial.population_parameters,
        controller.population_parameters()
    );
    assert_eq!(initial.omega.as_ref(), controller.omega());
    assert_eq!(initial.omega_iov.as_ref(), controller.omega_iov());
    assert_eq!(initial.residual_sigmas, controller.residual_sigmas());
    assert!(initial.latest_cycle_diagnostics.is_none());
    assert_eq!(controller.total_cycles(), 2);
    assert_eq!(controller.progress(), 0.0);

    assert_eq!(
        controller.step().expect("first cycle should run"),
        Status::Continue
    );
    let completed = controller.snapshot();
    assert_eq!(completed.cycle, 1);
    assert_eq!(completed.total_cycles, 2);
    assert_eq!(completed.status, Status::Continue);
    assert_eq!(completed.progress(), 0.5);
    assert_eq!(controller.progress(), 0.5);
    assert_eq!(
        completed.latest_cycle_diagnostics.as_ref(),
        controller.cycle_diagnostics().last()
    );
    assert_eq!(
        completed
            .latest_cycle_diagnostics
            .as_ref()
            .expect("completed-cycle diagnostics")
            .iteration,
        completed.cycle
    );

    let zero_total = ParametricFitSnapshot {
        total_cycles: 0,
        ..completed
    };
    assert_eq!(zero_total.progress(), 0.0);
}

#[test]
fn observer_receives_matching_snapshots_including_final_cycle() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    let mut snapshots = Vec::new();
    let result = lifecycle_problem()
        .fit_with_observer(tiny_config(2), |controller: &ParametricFitController<_>| {
            snapshots.push(controller.snapshot());
            ParametricCycleFlow::Continue
        })
        .expect("observed fit should complete");

    assert_eq!(snapshots.len(), 2);
    assert_eq!(
        snapshots
            .iter()
            .map(|snapshot| snapshot.cycle)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(snapshots[0].progress(), 0.5);
    assert_eq!(snapshots[1].progress(), 1.0);
    assert!(snapshots[1].is_terminal());
    assert_eq!(snapshots[1].status, Status::Stop(StopReason::MaxCycles));
    assert_eq!(
        snapshots[1].latest_cycle_diagnostics.as_ref(),
        result.cycle_diagnostics().last()
    );
}

#[test]
fn observer_stop_is_aborted_only_while_fit_is_running() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    let result = lifecycle_problem()
        .fit_with_observer(
            tiny_config(3),
            |_controller: &ParametricFitController<_>| ParametricCycleFlow::Stop,
        )
        .expect("observer-aborted fit should return its completed cycle");

    assert_eq!(result.cycle_diagnostics().len(), 1);
    assert_eq!(result.termination_reason(), Some(&StopReason::Aborted));
}

#[test]
fn averaged_observer_abort_before_smoothing_retains_terminal_estimator() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    let result = lifecycle_problem()
        .fit_with_observer(
            averaged_config(),
            |_controller: &ParametricFitController<_>| ParametricCycleFlow::Stop,
        )
        .expect("pre-smoothing observer abort should produce a result");

    assert_eq!(result.termination_reason(), Some(&StopReason::Aborted));
    assert!(!result.converged());
    assert_eq!(result.cycle_diagnostics().len(), 1);
    assert!(!result.estimator_metadata().average_applied);
    assert_eq!(result.estimator_metadata().averaging_start_cycle, None);
    assert_eq!(result.estimator_metadata().averaged_iterations, 0);
}

#[test]
fn averaged_observer_abort_after_smoothing_installs_completed_average() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    let result = lifecycle_problem()
        .fit_with_observer(
            averaged_config(),
            |controller: &ParametricFitController<_>| {
                if controller.cycle() == 2 {
                    ParametricCycleFlow::Stop
                } else {
                    ParametricCycleFlow::Continue
                }
            },
        )
        .expect("post-smoothing observer abort should produce a result");

    assert_eq!(result.termination_reason(), Some(&StopReason::Aborted));
    assert!(!result.converged());
    assert_eq!(result.cycle_diagnostics().len(), 2);
    assert!(result.estimator_metadata().average_applied);
    assert_eq!(result.estimator_metadata().averaging_start_cycle, Some(2));
    assert_eq!(result.estimator_metadata().averaged_iterations, 1);
}

#[test]
fn algorithm_terminal_reason_wins_over_observer_and_user_stop() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    let result = lifecycle_problem()
        .fit_with_observer(
            tiny_config(1),
            |_controller: &ParametricFitController<_>| ParametricCycleFlow::Stop,
        )
        .expect("final observer callback should not replace MaxCycles");
    assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));

    let mut controller = lifecycle_problem()
        .fit_controller(tiny_config(1))
        .expect("controller should build");
    assert_eq!(
        controller.step().expect("final cycle should run"),
        Status::Stop(StopReason::MaxCycles)
    );
    controller.request_stop();
    assert_eq!(controller.status(), &Status::Stop(StopReason::MaxCycles));
}

#[test]
fn stale_stop_file_is_removed_when_controller_is_constructed() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    fs::write("stop", "stale").expect("stale stop file should be writable");

    let controller = lifecycle_problem()
        .fit_controller(tiny_config(2))
        .expect("controller should remove stale stop file");

    assert!(!Path::new("stop").exists());
    assert_eq!(controller.status(), &Status::Continue);
}

#[test]
fn averaged_stop_file_before_smoothing_retains_terminal_estimator() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    let mut controller = lifecycle_problem()
        .fit_controller(averaged_config())
        .expect("controller should build");
    fs::write("stop", "stop").expect("stop file should be writable");
    assert_eq!(
        controller
            .step()
            .expect("exploration cycle should complete"),
        Status::Stop(StopReason::StopFile)
    );
    let result = controller.into_result().expect("stop-file result");

    assert_eq!(result.termination_reason(), Some(&StopReason::StopFile));
    assert!(!result.converged());
    assert!(!result.estimator_metadata().average_applied);
    assert_eq!(result.estimator_metadata().averaging_start_cycle, None);
    assert_eq!(result.estimator_metadata().averaged_iterations, 0);
}

#[test]
fn averaged_stop_file_after_smoothing_installs_completed_average() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    let mut controller = lifecycle_problem()
        .fit_controller(averaged_config())
        .expect("controller should build");
    assert_eq!(
        controller.step().expect("exploration cycle"),
        Status::Continue
    );
    fs::write("stop", "stop").expect("stop file should be writable");
    assert_eq!(
        controller.step().expect("smoothing cycle should complete"),
        Status::Stop(StopReason::StopFile)
    );
    let result = controller.into_result().expect("stop-file result");

    assert_eq!(result.termination_reason(), Some(&StopReason::StopFile));
    assert!(!result.converged());
    assert!(result.estimator_metadata().average_applied);
    assert_eq!(result.estimator_metadata().averaging_start_cycle, Some(2));
    assert_eq!(result.estimator_metadata().averaged_iterations, 1);
}

#[test]
fn stop_file_after_a_completed_cycle_preserves_cycle_and_reason() {
    let _lock = STOP_FILE_LOCK.lock().expect("stop-file test lock");
    let _cleanup = StopFileCleanup;
    let mut controller = lifecycle_problem()
        .fit_controller(tiny_config(2))
        .expect("controller should build");
    fs::write("stop", "stop").expect("stop file should be writable");

    let status = controller
        .step()
        .expect("completed cycle should be retained");
    assert_eq!(status, Status::Stop(StopReason::StopFile));
    assert_eq!(controller.cycle(), 1);
    assert_eq!(controller.cycle_diagnostics().len(), 1);
    assert_eq!(
        controller.snapshot().status,
        Status::Stop(StopReason::StopFile)
    );
    assert!(Path::new("stop").exists());

    let result = controller
        .into_result()
        .expect("stop-file termination should produce a completed fit result");
    assert_eq!(result.termination_reason(), Some(&StopReason::StopFile));
    assert_eq!(result.cycle_diagnostics().len(), 1);
}
