use anyhow::Result;
use pharmsol::prelude::simulator::Equation;
use serde::{Deserialize, Serialize};

use super::fit::{CycleOutcome, FitState, Runner};

/// Configuration options for the non-parametric maximum a posteriori (NPMAP) algorithm
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NpmapConfig {}

impl NpmapConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Non-parametric maximum a posteriori (NPMAP) algorithm.
///
/// Reweights the prior support points by their likelihood given the data, in a single pass:
/// the support points themselves are left unchanged.
#[derive(Debug)]
pub(crate) struct Npmap;

impl<E: Equation> Runner<E> for Npmap {
    fn step(&mut self, state: &mut FitState<E>) -> Result<CycleOutcome> {
        let solution = state.estimate(state.solution().theta().clone(), false)?;
        state.set_solution(solution);
        Ok(CycleOutcome::Converged)
    }
}

#[cfg(test)]
mod tests {
    use crate::estimation::nonparametric::burke;
    use crate::prelude::*;

    use pharmsol::{fa, fetch_params, lag, Subject, SubjectBuilderExt};

    fn equation() -> pharmsol::equation::ODE {
        pharmsol::equation::ODE::new(
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
            pharmsol::equation::metadata::new("npmap_test")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["0"])
                .route(pharmsol::equation::Route::bolus("0").to_state("central")),
        )
        .expect("metadata attachment should validate")
    }

    #[test]
    fn npmap_reports_neg2ll_like_every_other_algorithm() -> anyhow::Result<()> {
        let data = Data::new(vec![Subject::builder("1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 10.0, 0)
            .observation(2.0, 8.0, 0)
            .build()]);
        let parameters = ParameterSpace::bounded()
            .add("ke", 0.001, 3.0)
            .add("v", 25.0, 250.0);
        let error_models = AssayErrorModels::new().add(
            "0",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )?;
        let problem = EstimationProblem::nonparametric(
            equation(),
            data,
            Theta::sobol(&parameters, 64)?,
            error_models,
        )?;

        let result = problem.fit_with(NpmapConfig::new())?;

        let (_, log_likelihood) = burke(result.psi())?;
        approx::assert_relative_eq!(result.objf(), -2.0 * log_likelihood, epsilon = 1e-10);
        assert_eq!(result.cycle_log().cycles()[0].objf(), result.objf());
        Ok(())
    }
}
