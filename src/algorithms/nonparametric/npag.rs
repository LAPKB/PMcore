use anyhow::Result;
use pharmsol::prelude::{data::AssayErrorModels, simulator::Equation};
use serde::{Deserialize, Serialize};

use super::error_optim::{ErrorModelOptimizer, ErrorOptimConfig};
use super::fit::{CycleOutcome, FitState, Runner};
use crate::estimation::nonparametric::{adaptative_grid, Theta};

/// Configuration options for the Non-Parametric Adaptive Grid (NPAG) algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NpagConfig {
    pub eps: f64,
    pub min_eps: f64,
    pub objective_tolerance: f64,
    pub pyl_tolerance: f64,
    pub prune_threshold: f64,
    pub qr_tolerance: f64,
    pub grid_tolerance: f64,
    pub error_optim: ErrorOptimConfig,
    pub max_cycles: usize,
    pub progress: bool,
}

impl Default for NpagConfig {
    fn default() -> Self {
        Self {
            eps: 0.2,
            min_eps: 1e-4,
            objective_tolerance: 1e-4,
            pyl_tolerance: 1e-2,
            prune_threshold: 1e-3,
            qr_tolerance: 1e-8,
            grid_tolerance: 1e-4,
            error_optim: ErrorOptimConfig::default(),
            max_cycles: 1000,
            progress: true,
        }
    }
}

impl NpagConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn min_eps(mut self, min_eps: f64) -> Self {
        self.min_eps = min_eps;
        self
    }

    pub fn objective_tolerance(mut self, tolerance: f64) -> Self {
        self.objective_tolerance = tolerance;
        self
    }

    pub fn pyl_tolerance(mut self, tolerance: f64) -> Self {
        self.pyl_tolerance = tolerance;
        self
    }

    pub fn prune_threshold(mut self, threshold: f64) -> Self {
        self.prune_threshold = threshold;
        self
    }

    pub fn qr_tolerance(mut self, tolerance: f64) -> Self {
        self.qr_tolerance = tolerance;
        self
    }

    pub fn grid_tolerance(mut self, tolerance: f64) -> Self {
        self.grid_tolerance = tolerance;
        self
    }

    pub fn error_optim(mut self, config: ErrorOptimConfig) -> Self {
        self.error_optim = config;
        self
    }

    pub fn max_cycles(mut self, cycles: usize) -> Self {
        self.max_cycles = cycles;
        self
    }

    pub fn progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }
}

/// NPAG's own state between cycles: its configuration, the adaptive grid resolution, and the
/// bookkeeping for its convergence test.
#[derive(Debug)]
pub(crate) struct Npag {
    config: NpagConfig,
    /// Bounds of each parameter; the grid is refined within them.
    ranges: Vec<(f64, f64)>,
    /// Grid resolution, as a fraction of each parameter's range.
    eps: f64,
    /// `Σ ln(Ψ·w)` the last time `eps` was refined down to `min_eps`. NPAG has converged when
    /// two such values in a row agree to within `pyl_tolerance`.
    last_fine_objective: f64,
    error_optimizer: ErrorModelOptimizer,
}

impl Npag {
    pub(crate) fn new(config: NpagConfig, prior: &Theta, error_models: &AssayErrorModels) -> Self {
        Self {
            ranges: prior.parameters().finite_ranges(),
            eps: config.eps,
            last_fine_objective: -1e30,
            error_optimizer: ErrorModelOptimizer::new(config.error_optim.clone(), error_models),
            config,
        }
    }

    /// Whether the fit has converged.
    ///
    /// Every time the objective stalls, the grid is refined by halving `eps`. Once `eps`
    /// reaches `min_eps`, the fit has converged if `Σ ln(Ψ·w)` matches its value at the
    /// previous such point; otherwise `eps` is reset to start another round of refinement.
    fn converged<E: Equation>(&mut self, state: &FitState<E>) -> bool {
        tracing::debug!("EPS = {:.4}", self.eps);

        let stalled = state
            .objective_change()
            .is_some_and(|change| change <= self.config.objective_tolerance);
        if !stalled || self.eps <= self.config.min_eps {
            return false;
        }

        self.eps /= 2.0;
        if self.eps > self.config.min_eps {
            return false;
        }

        let solution = state.solution();
        let pyl = solution.psi().matrix() * solution.weights().weights();
        let objective: f64 = pyl.iter().map(|x| x.ln()).sum();
        if (objective - self.last_fine_objective).abs() <= self.config.pyl_tolerance {
            return true;
        }
        self.last_fine_objective = objective;
        self.eps = self.config.eps;
        false
    }
}

impl<E: Equation> Runner<E> for Npag {
    fn step(&mut self, state: &mut FitState<E>) -> Result<CycleOutcome> {
        let mut theta = state.solution().theta().clone();
        if !state.is_first_cycle() {
            adaptative_grid(
                &mut theta,
                self.eps,
                &self.ranges,
                self.config.grid_tolerance,
            )?;
        }

        let estimate = state.estimate(theta, self.config.progress && state.is_first_cycle())?;
        let solution = state.condense(
            estimate,
            self.config.prune_threshold,
            self.config.qr_tolerance,
        )?;
        state.set_solution(solution);
        self.error_optimizer.optimize(state)?;

        Ok(if self.converged(state) {
            CycleOutcome::Converged
        } else {
            CycleOutcome::Continue
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    use pharmsol::{fa, fetch_params, lag, Subject, SubjectBuilderExt};

    fn simple_equation() -> pharmsol::equation::ODE {
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
            pharmsol::equation::metadata::new("npag_settings_test")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["0"])
                .route(pharmsol::equation::Route::bolus("0").to_state("central")),
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

    #[test]
    fn npag_runs_without_error() {
        let parameters = ParameterSpace::bounded()
            .add("ke", 0.001, 3.0)
            .add("v", 25.0, 250.0);
        let prior = Theta::sobol_default(&parameters).expect("Failed to build prior");
        let error_models = AssayErrorModels::new()
            .add(
                "0",
                AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
            )
            .expect("Failed to build error models");
        let problem =
            EstimationProblem::nonparametric(simple_equation(), simple_data(), prior, error_models)
                .expect("Failed to build problem");

        let result = problem.fit_with(NonParametricAlgorithm::npag());

        assert!(
            result.is_ok(),
            "NPAG algorithm should run without error, but got: {:?}",
            result.err()
        );
    }

    #[test]
    fn controller_exposes_a_consistent_solution_between_steps() -> anyhow::Result<()> {
        let parameters = ParameterSpace::bounded()
            .add("ke", 0.001, 3.0)
            .add("v", 25.0, 250.0);
        let error_models = AssayErrorModels::new().add(
            "0",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )?;
        let problem = EstimationProblem::nonparametric(
            simple_equation(),
            simple_data(),
            Theta::sobol_default(&parameters)?,
            error_models,
        )?;

        let mut controller = problem.fit_controller(NpagConfig::new().progress(false))?;
        for _ in 0..3 {
            controller.step()?;
            let nspp = controller.theta().nspp();
            assert_eq!(controller.weights().len(), nspp);
            assert_eq!(controller.psi().matrix().ncols(), nspp);
            assert_eq!(controller.objf(), -2.0 * controller.log_likelihood());
        }
        assert_eq!(controller.cycle(), 3);
        Ok(())
    }
}
