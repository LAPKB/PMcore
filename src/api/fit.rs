use anyhow::Result;
use pharmsol::equation::Equation;

use crate::compile::CompiledProblem;
use crate::api::estimation_problem::EstimationProblem;
use crate::api::progress::{FitControlSource, FitProgress};
use crate::estimation::nonparametric;
use crate::model::EquationMetadataSource;
use crate::results::FitResult;

fn compile_problem<E>(problem: EstimationProblem<E>) -> Result<CompiledProblem<E>>
where
    E: Equation + Clone + Send + 'static + EquationMetadataSource,
{
    if problem.runtime.logging.initialize {
        problem.initialize_logs()?;
    }

    problem.compile()
}

fn finish_fit<E>(mut result: FitResult<E>, write_outputs: bool) -> Result<FitResult<E>>
where
    E: Equation,
{
    if write_outputs {
        result.write_outputs()?;
    }

    Ok(result)
}

pub fn fit_in_memory<E: Equation + Clone + Send + 'static + EquationMetadataSource>(
    problem: EstimationProblem<E>,
) -> Result<FitResult<E>> {
    let compiled = compile_problem(problem)?;
    nonparametric::fit(compiled)
}

pub fn fit<E: Equation + Clone + Send + 'static + EquationMetadataSource>(
    problem: EstimationProblem<E>,
) -> Result<FitResult<E>> {
    finish_fit(fit_in_memory(problem)?, true)
}

pub fn fit_with_progress_in_memory<E, F>(
    problem: EstimationProblem<E>,
    on_progress: F,
) -> Result<FitResult<E>>
where
    E: Equation + Clone + Send + 'static + EquationMetadataSource,
    F: FnMut(FitProgress),
{
    let compiled = compile_problem(problem)?;
    nonparametric::fit_with_progress(compiled, on_progress)
}

pub fn fit_with_progress_and_control_in_memory<E, F, C>(
    problem: EstimationProblem<E>,
    on_progress: F,
    next_control: C,
) -> Result<FitResult<E>>
where
    E: Equation + Clone + Send + 'static + EquationMetadataSource,
    F: FnMut(FitProgress),
    C: FitControlSource,
{
    let compiled = compile_problem(problem)?;
    nonparametric::fit_with_progress_and_control(compiled, on_progress, next_control)
}

pub fn fit_with_progress<E, F>(
    problem: EstimationProblem<E>,
    on_progress: F,
) -> Result<FitResult<E>>
where
    E: Equation + Clone + Send + 'static + EquationMetadataSource,
    F: FnMut(FitProgress),
{
    finish_fit(fit_with_progress_in_memory(problem, on_progress)?, true)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pharmsol::{AssayErrorModel, ErrorPoly, Subject};
    use std::collections::VecDeque;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        fit, fit_in_memory, fit_with_progress, fit_with_progress_and_control_in_memory,
        fit_with_progress_in_memory, FitProgress,
    };
    use crate::algorithms::{Status, StopReason};
    use crate::api::{EstimationProblem, FitControl, LoggingOptions, Npag};
    use crate::estimation::nonparametric::median;
    use crate::prelude::*;

    fn equation() -> equation::ODE {
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
            equation::metadata::new("fit_progress")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["0"])
                .route(equation::Route::bolus("0").to_state("central")),
        )
        .expect("metadata attachment should validate")
    }

    fn test_output_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "pmcore-{label}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("test output directory should be created");
        path
    }

    fn output_problem(output_dir: &Path, cycles: usize) -> Result<EstimationProblem<equation::ODE>> {
        let data = pharmsol::Data::new(vec![Subject::builder("1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 7.0, 0)
            .build()]);

        let assay_error =
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);

        EstimationProblem::builder(equation(), data)
            .output_dir(output_dir.to_string_lossy().into_owned())
            .parameter(Parameter::bounded("ke", 0.05, 1.0))?
            .parameter(Parameter::bounded("v", 5.0, 50.0))?
            .method(Npag::new())
            .error("0", assay_error)?
            .cycles(cycles)
            .progress(false)
            .prior(Prior::sobol(8, 7))
            .log_level(LoggingOptions::default().level)
            .build()
    }

    fn cycle_events(events: &[FitProgress]) -> Vec<&crate::api::NonparametricCycleProgress> {
        events
            .iter()
            .filter_map(|event| match event {
                FitProgress::NonparametricCycle(cycle) => Some(cycle),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn fit_writes_outputs_when_requested() -> Result<()> {
        let output_dir = test_output_dir("fit-with-outputs");
        let result = fit(output_problem(&output_dir, 2)?)?;

        assert!(output_dir.join("iterations.csv").exists());
        assert!(output_dir.join("settings.json").exists());
        assert!(result.as_nonparametric().is_some());

        Ok(())
    }

    #[test]
    fn fit_in_memory_skips_output_writing() -> Result<()> {
        let output_dir = test_output_dir("fit-in-memory");
        let result = fit_in_memory(output_problem(&output_dir, 2)?)?;

        assert!(!output_dir.join("iterations.csv").exists());
        assert!(!output_dir.join("settings.json").exists());
        assert!(result.as_nonparametric().is_some());

        Ok(())
    }

    #[test]
    fn fit_with_progress_reports_cycles_and_writes_outputs() -> Result<()> {
        let output_dir = test_output_dir("fit-with-progress");
        let problem = output_problem(&output_dir, 2)?;

        let mut progress_events = Vec::new();
        let result = fit_with_progress(problem, |event| progress_events.push(event))?;

        let workspace = result
            .as_nonparametric()
            .expect("expected nonparametric fit result");
        let cycle_events = cycle_events(&progress_events);

        assert!(matches!(progress_events.first(), Some(FitProgress::FitStarted)));
        assert_eq!(cycle_events.len(), workspace.cycle_log().cycles().len());
        assert_eq!(cycle_events.len(), workspace.cycles());

        let last_cycle = cycle_events.last().copied().expect("missing last cycle event");
        assert!(matches!(last_cycle.status, Status::Stop(_)));
        assert_eq!(last_cycle.cycle, workspace.cycles());
        assert!(matches!(
            progress_events.last(),
            Some(FitProgress::FitCompleted {
                status: Status::Stop(_),
                ..
            })
        ));
        assert!(output_dir.join("iterations.csv").exists());
        assert!(output_dir.join("settings.json").exists());

        Ok(())
    }

    #[test]
    fn fit_with_progress_in_memory_reports_cycles_without_writing_outputs() -> Result<()> {
        let output_dir = test_output_dir("fit-with-progress-in-memory");
        let problem = output_problem(&output_dir, 2)?;
        let mut progress_events = Vec::new();
        let result = fit_with_progress_in_memory(problem, |event| progress_events.push(event))?;

        let workspace = result
            .as_nonparametric()
            .expect("expected nonparametric fit result");

        assert!(matches!(progress_events.first(), Some(FitProgress::FitStarted)));
        assert_eq!(cycle_events(&progress_events).len(), workspace.cycle_log().cycles().len());
        assert_eq!(cycle_events(&progress_events).len(), workspace.cycles());
        assert!(!output_dir.join("iterations.csv").exists());
        assert!(!output_dir.join("settings.json").exists());

        Ok(())
    }

    #[test]
    fn fit_with_progress_in_memory_matches_cycle_log_snapshot() -> Result<()> {
        let output_dir = test_output_dir("fit-rich-progress");
        let mut progress_events = Vec::new();
        let result = fit_with_progress_in_memory(output_problem(&output_dir, 2)?, |event| {
            progress_events.push(event)
        })?;

        let workspace = result
            .as_nonparametric()
            .expect("expected nonparametric fit result");
        let cycle_events = cycle_events(&progress_events);
        let last_cycle_progress = cycle_events
            .last()
            .copied()
            .expect("missing final cycle progress");
        let last_cycle = workspace
            .cycle_log()
            .cycles()
            .last()
            .expect("missing final cycle log state");

        assert!(cycle_events.first().expect("missing first cycle").objective_delta.is_none());
        assert!(last_cycle_progress.objective_delta.is_some());
        assert_eq!(last_cycle_progress.cycle, last_cycle.cycle());
        assert_eq!(last_cycle_progress.neg2ll, last_cycle.objf());
        assert_eq!(last_cycle_progress.nspp, last_cycle.nspp());
        assert_eq!(last_cycle_progress.status, *last_cycle.status());
        assert_eq!(last_cycle_progress.error_models.len(), 1);
        assert_eq!(
            last_cycle_progress
                .error_models
                .first()
                .expect("missing error model progress")
                .value,
            last_cycle
                .error_models()
                .iter()
                .find_map(|(_, error_model)| match error_model {
                    AssayErrorModel::None => None,
                    _ => Some(error_model.factor().expect("factor should exist")),
                })
                .expect("missing final error model factor")
        );

        let expected_parameters = last_cycle
            .theta()
            .matrix()
            .col_iter()
            .map(|values| {
                let parameter_values: Vec<f64> = values.iter().copied().collect();
                let mean = parameter_values.iter().sum::<f64>() / parameter_values.len() as f64;
                let variance = parameter_values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / (parameter_values.len() as f64 - 1.0);

                (mean, median(&parameter_values), variance.sqrt())
            })
            .collect::<Vec<_>>();

        assert_eq!(last_cycle_progress.parameters.len(), expected_parameters.len());
        for (index, (mean, median_value, sd)) in expected_parameters.into_iter().enumerate() {
            let parameter = &last_cycle_progress.parameters[index];
            assert_eq!(parameter.mean, mean);
            assert_eq!(parameter.median, median_value);
            if sd.is_nan() {
                assert!(parameter.sd.is_nan());
            } else {
                assert_eq!(parameter.sd, sd);
            }
        }

        Ok(())
    }

    #[test]
    fn fit_with_progress_and_control_in_memory_pauses_then_resumes() -> Result<()> {
        let output_dir = test_output_dir("fit-pause-resume");
        let problem = output_problem(&output_dir, 2)?;
        let mut progress_events = Vec::new();
        let mut commands = VecDeque::from(vec![FitControl::PauseAfterCycle, FitControl::Resume]);

        let result = fit_with_progress_and_control_in_memory(
            problem,
            |event| progress_events.push(event),
            || Ok(commands.pop_front()),
        )?;

        let workspace = result
            .as_nonparametric()
            .expect("expected nonparametric fit result");
        let paused_index = progress_events
            .iter()
            .position(|event| matches!(event, FitProgress::Paused { cycle } if *cycle == 1))
            .expect("missing paused event");
        let resumed_index = progress_events
            .iter()
            .position(|event| matches!(event, FitProgress::Resumed { cycle } if *cycle == 1))
            .expect("missing resumed event");

        assert!(paused_index < resumed_index);
        assert_eq!(cycle_events(&progress_events).len(), workspace.cycles());
        assert!(matches!(
            progress_events.last(),
            Some(FitProgress::FitCompleted {
                status: Status::Stop(StopReason::MaxCycles),
                ..
            })
        ));

        Ok(())
    }

    #[test]
    fn fit_with_progress_and_control_in_memory_stops_after_cycle() -> Result<()> {
        let output_dir = test_output_dir("fit-stop-after-cycle");
        let problem = output_problem(&output_dir, 4)?;
        let mut progress_events = Vec::new();
        let mut commands = VecDeque::from(vec![FitControl::StopAfterCycle]);

        let result = fit_with_progress_and_control_in_memory(
            problem,
            |event| progress_events.push(event),
            || Ok(commands.pop_front()),
        )?;

        let workspace = result
            .as_nonparametric()
            .expect("expected nonparametric fit result");

        assert_eq!(cycle_events(&progress_events).len(), 1);
        assert_eq!(workspace.cycles(), 1);
        assert!(!workspace.converged());
        assert!(progress_events.iter().any(
            |event| matches!(event, FitProgress::StopRequested { cycle } if *cycle == 1)
        ));
        assert!(matches!(
            progress_events.last(),
            Some(FitProgress::FitCompleted {
                cycles: 1,
                status: Status::Stop(StopReason::Stopped),
            })
        ));

        Ok(())
    }
}
