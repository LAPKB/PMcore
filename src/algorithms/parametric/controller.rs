//! Step through a parametric fit cycle by cycle using the same controller API
//! as nonparametric fits.

use std::{fs, path::Path};

use anyhow::{Context, Result};
use ndarray::Array2;
use pharmsol::prelude::simulator::Equation;

use crate::algorithms::{Status, StopReason};
use crate::estimation::{EstimationProblem, Parametric};
use crate::results::{ParametricResult, SaemCycleDiagnostics};

use super::{ParametricAlgorithm, ParametricRunner};

/// A self-contained view of a live parametric fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricFitSnapshot {
    /// Number of completed cycles.
    pub cycle: usize,
    /// Number of cycles in the configured schedule.
    pub total_cycles: usize,
    /// Current lifecycle status.
    pub status: Status,
    /// Current conditional `-2 log likelihood` diagnostic.
    pub conditional_n2ll: f64,
    /// Current population parameter estimates in model-space ψ.
    pub population_parameters: Vec<f64>,
    /// Current covariate coefficients in canonical declaration order.
    pub covariate_betas: Option<Vec<f64>>,
    /// Current random-effect covariance matrix.
    pub omega: Option<Array2<f64>>,
    /// Current inter-occasion covariance matrix.
    pub omega_iov: Option<Array2<f64>>,
    /// Current primary residual-error sigma parameters.
    pub residual_sigmas: Vec<f64>,
    /// Most recent completed-cycle diagnostics, if any.
    pub latest_cycle_diagnostics: Option<SaemCycleDiagnostics>,
}

impl ParametricFitSnapshot {
    /// Completed fraction of the configured schedule, clamped to `0.0..=1.0`.
    pub fn progress(&self) -> f64 {
        if self.total_cycles == 0 {
            0.0
        } else {
            (self.cycle as f64 / self.total_cycles as f64).clamp(0.0, 1.0)
        }
    }

    /// Whether the fit has reached any terminal status.
    pub fn is_terminal(&self) -> bool {
        self.status.is_stop()
    }
}

/// A running parametric fit that can be advanced one cycle at a time.
///
/// The controller is algorithm-neutral and drives whichever parametric runner
/// the selected algorithm provides.
pub struct FitController<E: Equation + Send + 'static> {
    runner: Box<dyn ParametricRunner<E>>,
    termination_logged: bool,
}

impl<E: Equation + Send + 'static> FitController<E> {
    pub(crate) fn new(
        algorithm: ParametricAlgorithm,
        problem: EstimationProblem<E, Parametric>,
    ) -> Result<Self> {
        let stop_path = Path::new("stop");
        if stop_path.exists() {
            tracing::info!("Removing existing stop file prior to parametric run");
            fs::remove_file(stop_path).context("Unable to remove previous stop file")?;
        }

        let runner = algorithm.into_runner(problem)?;
        tracing::info!("Starting SAEM fit");
        tracing::debug!(
            scheduled_cycles = runner.total_iterations(),
            chains = ?runner.n_chains(),
            random_effects = ?runner.random_effect_names(),
            initial_population = ?runner.population_parameters(),
            "SAEM configuration and initial state"
        );
        if let Some(iov_effects) = runner.iov_effect_names() {
            tracing::debug!(iov_effects = ?iov_effects, "SAEM IOV configuration");
        }
        Ok(Self {
            runner,
            termination_logged: false,
        })
    }

    /// Advance one complete parametric estimation cycle and return its status.
    pub fn step(&mut self) -> Result<Status> {
        if self.runner.status().is_stop() {
            return Ok(self.runner.status().clone());
        }

        let next_cycle = self.runner.cycle() + 1;
        let span = tracing::info_span!("", "{}", format!("Cycle {next_cycle}"));
        let _entered = span.enter();
        let previous_cycle = self.runner.cycle();
        match self.runner.step() {
            Ok(_) => {
                if self.runner.cycle() > previous_cycle {
                    self.log_completed_cycle();
                    if self.runner.status().is_continue() && Path::new("stop").exists() {
                        self.runner.request_stop(StopReason::StopFile);
                    }
                }
                let status = self.runner.status().clone();
                if status.is_stop() {
                    self.log_termination(&status, None);
                }
                Ok(status)
            }
            Err(error) => {
                let status = self.runner.status().clone();
                self.log_termination(&status, Some(&error));
                Err(error)
            }
        }
    }

    /// Flag the fit as aborted. Call before [`into_result`](Self::into_result)
    /// when stopping from a debugger or observer.
    pub fn request_stop(&mut self) {
        let was_running = self.runner.status().is_continue();
        self.runner.request_stop(StopReason::Aborted);
        if was_running {
            let status = self.runner.status().clone();
            self.log_termination(&status, None);
        }
    }

    /// Current cycle number, 0 before the first [`step`](Self::step).
    pub fn cycle(&self) -> usize {
        self.runner.cycle()
    }

    /// Current controller status.
    pub fn status(&self) -> &Status {
        self.runner.status()
    }

    /// Self-contained snapshot of the current live fit state.
    pub fn snapshot(&self) -> ParametricFitSnapshot {
        ParametricFitSnapshot {
            cycle: self.runner.cycle(),
            total_cycles: self.runner.total_iterations(),
            status: self.runner.status().clone(),
            conditional_n2ll: self.runner.n2ll(),
            population_parameters: self.runner.population_parameters().to_vec(),
            covariate_betas: self.runner.covariate_betas(),
            omega: self.runner.omega().cloned(),
            omega_iov: self.runner.omega_iov().cloned(),
            residual_sigmas: self.runner.residual_sigmas().to_vec(),
            latest_cycle_diagnostics: self.runner.cycle_diagnostics().last().cloned(),
        }
    }

    /// Number of cycles in the configured schedule.
    pub fn total_cycles(&self) -> usize {
        self.runner.total_iterations()
    }

    /// Completed fraction of the configured schedule, clamped to `0.0..=1.0`.
    pub fn progress(&self) -> f64 {
        let total_cycles = self.runner.total_iterations();
        if total_cycles == 0 {
            0.0
        } else {
            (self.runner.cycle() as f64 / total_cycles as f64).clamp(0.0, 1.0)
        }
    }

    /// Completed SAEM cycle records, suitable for observers and debuggers.
    pub fn cycle_diagnostics(&self) -> &[SaemCycleDiagnostics] {
        self.runner.cycle_diagnostics()
    }

    /// Current log-likelihood from residual scoring.
    pub fn likelihood(&self) -> f64 {
        self.runner.log_likelihood()
    }

    /// Current population parameter estimates in model-space ψ.
    pub fn population_parameters(&self) -> &[f64] {
        self.runner.population_parameters()
    }

    /// Current covariate coefficients in canonical declaration order.
    pub fn covariate_betas(&self) -> Option<Vec<f64>> {
        self.runner.covariate_betas()
    }

    /// Names of parameters with IIV random effects, in η/Ω order.
    pub fn random_effect_names(&self) -> &[String] {
        self.runner.random_effect_names()
    }

    /// Names of parameters with IOV random effects, in κ/Ω_IOV order.
    pub fn iov_effect_names(&self) -> Option<&[String]> {
        self.runner.iov_effect_names()
    }

    /// Current η log-prior under Ω.
    pub fn eta_log_prior(&self) -> f64 {
        self.runner.eta_log_prior()
    }

    /// Current κ log-prior under Ω_IOV.
    pub fn kappa_log_prior(&self) -> f64 {
        self.runner.kappa_log_prior()
    }

    /// Current log posterior, up to the current parameterization.
    pub fn log_posterior(&self) -> f64 {
        self.runner.log_posterior()
    }

    /// Last MCMC proposal acceptance rate, when at least one E-step has run.
    pub fn acceptance_rate(&self) -> Option<f64> {
        self.runner.acceptance_rate()
    }

    /// Ω-scaled η block acceptance rate in the most recent E-step.
    /// Returns `None` when the block kernel is disabled.
    pub fn eta_block_acceptance_rate(&self) -> Option<f64> {
        self.runner.eta_block_acceptance_rate()
    }

    /// κ component proposal acceptance rate in the most recent E-step.
    /// Returns `None` when IOV is not configured.
    pub fn kappa_acceptance_rate(&self) -> Option<f64> {
        self.runner.kappa_acceptance_rate()
    }

    /// Number of rejected proposals in the most recent E-step.
    pub fn rejected_proposals(&self) -> Option<usize> {
        self.runner.rejected_proposals()
    }

    /// Number of proposals rejected for non-finite posterior scores in the
    /// most recent E-step.
    pub fn non_finite_proposals(&self) -> Option<usize> {
        self.runner.non_finite_proposals()
    }

    /// Last MCMC acceptance rates in [`random_effect_names`](Self::random_effect_names) order.
    pub fn parameter_acceptance_rates(&self) -> Option<&[f64]> {
        self.runner.parameter_acceptance_rates()
    }

    /// Current component-wise proposal scales in random-effect order.
    pub fn proposal_step_sizes(&self) -> Option<&[f64]> {
        self.runner.proposal_step_sizes()
    }

    /// Current per-subject Ω-scaled η block multipliers.
    /// Returns `None` when the block kernel is disabled.
    pub fn eta_block_step_sizes(&self) -> Option<&[f64]> {
        self.runner.eta_block_step_sizes()
    }

    /// Last per-subject log acceptance ratios from the proposal-scoring pass.
    pub fn log_acceptance_ratios(&self) -> Option<&[f64]> {
        self.runner.log_acceptance_ratios()
    }

    /// Current negative log-likelihood from residual scoring.
    pub fn negative_log_likelihood(&self) -> f64 {
        self.runner.negative_log_likelihood()
    }

    /// Current `-2 log likelihood`, matching the non-parametric controller name.
    pub fn n2ll(&self) -> f64 {
        self.runner.n2ll()
    }

    /// Current stochastic-approximation step size.
    pub fn step_size(&self) -> f64 {
        self.runner.step_size()
    }

    /// Current random-effect covariance matrix in
    /// [`random_effect_names`](Self::random_effect_names) order.
    pub fn omega(&self) -> Option<&Array2<f64>> {
        self.runner.omega()
    }

    /// Current inter-occasion covariance matrix in
    /// [`iov_effect_names`](Self::iov_effect_names) order.
    pub fn omega_iov(&self) -> Option<&Array2<f64>> {
        self.runner.omega_iov()
    }

    /// Current diagonal of the random-effect covariance matrix, when available.
    pub fn omega_diagonal(&self) -> Option<Vec<f64>> {
        self.runner.omega_diagonal()
    }

    /// Current primary residual-error sigma parameters.
    pub fn residual_sigmas(&self) -> &[f64] {
        self.runner.residual_sigmas()
    }

    /// Number of individual MCMC chains currently scheduled per subject.
    pub fn n_chains(&self) -> Option<usize> {
        self.runner.n_chains()
    }

    /// Number of scheduled SAEM iterations.
    pub fn total_iterations(&self) -> usize {
        self.runner.total_iterations()
    }

    /// Run to completion and return the final parametric result.
    pub fn finish(mut self) -> Result<ParametricResult<E>> {
        while self.status().is_continue() {
            self.step()?;
        }
        self.into_result()
    }

    /// Convert the current state into a parametric result.
    pub fn into_result(mut self) -> Result<ParametricResult<E>> {
        let status = self.runner.status().clone();
        let failure_already_logged =
            self.termination_logged && status.stop_reason() == Some(&StopReason::NumericalFailure);
        if status.is_stop() {
            self.log_termination(&status, None);
        }
        match self.runner.into_result() {
            Ok(result) => {
                tracing::info!("SAEM result assembly complete");
                Ok(result)
            }
            Err(error) => {
                if !failure_already_logged {
                    tracing::error!("{error}");
                }
                Err(error)
            }
        }
    }

    fn log_completed_cycle(&self) {
        let Some(diagnostics) = self.runner.cycle_diagnostics().last() else {
            return;
        };
        tracing::info!(
            "Conditional N2LL = {:.4}",
            2.0 * diagnostics.conditional_negative_log_likelihood
        );
        tracing::debug!(
            phase = ?diagnostics.phase,
            stochastic_approximation_step = diagnostics.stochastic_approximation_step,
            covariance_step = diagnostics.covariance_step,
            population_parameters = ?diagnostics.population_parameters,
            omega = ?diagnostics.omega,
            residual_estimates = ?diagnostics.residual_error_estimates,
            eta_accepted = diagnostics.eta_accepted,
            eta_proposals = diagnostics.eta_proposals,
            eta_acceptance_rate = acceptance_rate(diagnostics.eta_accepted, diagnostics.eta_proposals),
            eta_rejected = diagnostics.eta_rejected,
            eta_non_finite = diagnostics.eta_non_finite,
            kappa_accepted = diagnostics.kappa_accepted,
            kappa_proposals = diagnostics.kappa_proposals,
            kappa_acceptance_rate = acceptance_rate(diagnostics.kappa_accepted, diagnostics.kappa_proposals),
            kappa_rejected = diagnostics.kappa_rejected,
            kappa_non_finite = diagnostics.kappa_non_finite,
            "SAEM cycle state"
        );
        if let Some(omega_iov) = diagnostics.omega_iov.as_ref() {
            tracing::debug!(omega_iov = ?omega_iov, "SAEM IOV cycle state");
        }
        if diagnostics.eta_block_proposals > 0 {
            tracing::debug!(
                eta_block_accepted = diagnostics.eta_block_accepted,
                eta_block_proposals = diagnostics.eta_block_proposals,
                eta_block_acceptance_rate = acceptance_rate(
                    diagnostics.eta_block_accepted,
                    diagnostics.eta_block_proposals,
                ),
                eta_block_rejected = diagnostics.eta_block_rejected,
                eta_block_non_finite = diagnostics.eta_block_non_finite,
                "SAEM block-kernel state"
            );
        }
        self.log_cycle_guardrails(diagnostics);
    }

    fn log_cycle_guardrails(&self, diagnostics: &SaemCycleDiagnostics) {
        if diagnostics.omega_update_rejected {
            tracing::warn!("Omega update rejected");
        }
        if diagnostics.omega_iov_update_rejected {
            tracing::warn!("Omega_IOV update rejected");
        }
        if diagnostics.eta_non_finite > 0 {
            tracing::warn!(
                count = diagnostics.eta_non_finite,
                "Non-finite eta proposals"
            );
        }
        if diagnostics.eta_block_non_finite > 0 {
            tracing::warn!(
                count = diagnostics.eta_block_non_finite,
                "Non-finite eta block proposals"
            );
        }
        if diagnostics.kappa_non_finite > 0 {
            tracing::warn!(
                count = diagnostics.kappa_non_finite,
                "Non-finite kappa proposals"
            );
        }
        for residual in &diagnostics.residual_diagnostics {
            if residual.update_rejected {
                tracing::warn!(output = %residual.output, "Residual update rejected");
            }
            if residual.non_finite_prediction_count > 0 {
                tracing::warn!(
                    output = %residual.output,
                    count = residual.non_finite_prediction_count,
                    "Non-finite residual predictions"
                );
            }
            if residual.exponential_domain_violation_count > 0 {
                tracing::warn!(
                    output = %residual.output,
                    count = residual.exponential_domain_violation_count,
                    "Residual domain violations"
                );
            }
            if residual.proportional_floor_count > 0 {
                tracing::debug!(
                    output = %residual.output,
                    count = residual.proportional_floor_count,
                    "Residual prediction floor applied"
                );
            }
            if residual.combined_additive_collapse_warning {
                tracing::warn!(output = %residual.output, "Combined-family additive residual collapse");
            }
            if residual.optimizer_converged == Some(false) {
                tracing::warn!(output = %residual.output, "Residual optimizer did not converge");
            }
        }
    }

    fn log_termination(&mut self, status: &Status, error: Option<&anyhow::Error>) {
        if self.termination_logged {
            return;
        }
        match status.stop_reason() {
            Some(StopReason::Converged) => tracing::info!(
                "PMcore operational convergence criteria passed; this does not prove mathematical convergence"
            ),
            Some(StopReason::MaxCycles) => {
                tracing::warn!("Maximum SAEM cycles reached; this is not statistical convergence")
            }
            Some(StopReason::StopFile) => tracing::warn!("SAEM stopped: stop file detected"),
            Some(StopReason::Aborted) => tracing::warn!("SAEM aborted"),
            Some(StopReason::NumericalFailure) => {
                if let Some(error) = error {
                    tracing::error!("{error}");
                } else {
                    tracing::error!("SAEM stopped: numerical failure");
                }
            }
            None => {
                if let Some(error) = error {
                    tracing::error!("{error}");
                } else {
                    return;
                }
            }
        }
        self.termination_logged = true;
    }
}

fn acceptance_rate(accepted: usize, proposed: usize) -> f64 {
    if proposed == 0 {
        0.0
    } else {
        accepted as f64 / proposed as f64
    }
}

/// An observer's verdict after a parametric cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleFlow {
    Continue,
    Stop,
}

/// Callback run after every parametric cycle.
pub trait FitObserver<E: Equation + Send + 'static> {
    fn on_cycle(&mut self, controller: &FitController<E>) -> CycleFlow;
}

impl<E, F> FitObserver<E> for F
where
    E: Equation + Send + 'static,
    F: FnMut(&FitController<E>) -> CycleFlow,
{
    fn on_cycle(&mut self, controller: &FitController<E>) -> CycleFlow {
        self(controller)
    }
}

impl<E: Equation + Send + 'static> EstimationProblem<E, Parametric> {
    /// Start a parametric fit that can be driven one cycle at a time.
    pub fn fit_controller(
        self,
        algorithm: impl Into<ParametricAlgorithm>,
    ) -> Result<FitController<E>> {
        FitController::new(algorithm.into(), self)
    }

    /// Run a parametric fit with an observer callback after every cycle.
    ///
    /// The observer sees every completed cycle, including the final cycle. Use
    /// [`FitController::snapshot`] when the callback needs an owned progress
    /// record:
    ///
    /// ```no_run
    /// use pmcore::prelude::*;
    /// # fn run(problem: EstimationProblem<pmcore::prelude::ODE, Parametric>) -> Result<()> {
    /// let result = problem.fit_with_observer(SaemConfig::new(), |controller: &ParametricFitController<_>| {
    ///     let snapshot = controller.snapshot();
    ///     println!("cycle {} | {:.0}%", snapshot.cycle, 100.0 * snapshot.progress());
    ///     ParametricCycleFlow::Continue
    /// })?;
    /// # let _ = result; Ok(())
    /// # }
    /// ```
    pub fn fit_with_observer<O: FitObserver<E>>(
        self,
        algorithm: impl Into<ParametricAlgorithm>,
        mut observer: O,
    ) -> Result<ParametricResult<E>> {
        let mut controller = self.fit_controller(algorithm)?;
        loop {
            let status = controller.step()?;
            let flow = observer.on_cycle(&controller);
            if status.is_stop() {
                break;
            }
            if flow == CycleFlow::Stop {
                controller.request_stop();
                break;
            }
        }
        controller.into_result()
    }
}
