use anyhow::Result;
use pharmsol::equation::Equation;

use crate::api::estimation_problem::{EstimationMethod, EstimationProblem};
use crate::api::progress::FitProgress;
use crate::estimation::nonparametric;
use crate::results::FitResult;

pub fn fit<E: Equation + Clone + Send + 'static>(
    problem: EstimationProblem<E>,
) -> Result<FitResult<E>> {
    if problem.runtime.logging.initialize {
        problem.initialize_logs()?;
    }

    let method = problem.method;
    let compiled = problem.compile()?;

    match method {
        EstimationMethod::Nonparametric(_) => nonparametric::fit(compiled),
    }
}

pub fn fit_with_progress<E, F>(
    problem: EstimationProblem<E>,
    mut on_progress: F,
) -> Result<FitResult<E>>
where
    E: Equation + Clone + Send + 'static,
    F: FnMut(FitProgress),
{
    if problem.runtime.logging.initialize {
        problem.initialize_logs()?;
    }

    let compiled = problem.compile()?;
    let method = compiled.method();

    match method {
        EstimationMethod::Nonparametric(_) => nonparametric::fit_with_progress(compiled, |event| {
            on_progress(FitProgress::NonparametricCycle(event));
        }),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use pharmsol::{AssayErrorModel, ErrorPoly, Subject};

    use super::{fit_with_progress, FitProgress};
    use crate::algorithms::Status;
    use crate::api::{
        EstimationMethod, EstimationProblem, LoggingOptions, NonparametricMethod, NpagOptions,
        RuntimeOptions,
    };
    use crate::model::{
        ModelDefinition, ObservationChannel, ObservationSpec, ParameterSpace, ParameterSpec,
    };
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
    }

    #[test]
    fn fit_with_progress_reports_nonparametric_cycles() -> Result<()> {
        let data = pharmsol::Data::new(vec![Subject::builder("1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 7.0, 0)
            .build()]);

        let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
        let observations = ObservationSpec::new()
            .add_channel(ObservationChannel::continuous(0, "cp"))
            .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?);

        let model = ModelDefinition::builder(equation())
            .parameters(
                ParameterSpace::new()
                    .add(ParameterSpec::bounded("ke", 0.05, 1.0))
                    .add(ParameterSpec::bounded("v", 5.0, 50.0)),
            )
            .observations(observations)
            .build()?;

        let runtime = RuntimeOptions {
            cycles: 2,
            progress: false,
            prior: Some(Prior::sobol(8, 7)),
            logging: LoggingOptions {
                initialize: false,
                ..LoggingOptions::default()
            },
            ..RuntimeOptions::default()
        };

        let problem = EstimationProblem::builder(model, data)
            .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
                NpagOptions,
            )))
            .runtime(runtime)
            .build()?;

        let mut progress_events = Vec::new();
        let result = fit_with_progress(problem, |event| progress_events.push(event))?;

        let workspace = result
            .as_nonparametric()
            .expect("expected nonparametric fit result");

        assert!(!progress_events.is_empty());
        assert_eq!(progress_events.len(), workspace.cycle_log().cycles().len());
        assert_eq!(progress_events.len(), workspace.cycles());

        let last_event = progress_events.last().cloned().expect("missing last event");
        let FitProgress::NonparametricCycle(last_cycle) = last_event;
        assert!(matches!(last_cycle.status, Status::Stop(_)));
        assert_eq!(last_cycle.cycle, workspace.cycles());

        Ok(())
    }
}
