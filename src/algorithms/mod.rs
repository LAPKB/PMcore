use crate::estimation::nonparametric::{NPCycle, Psi, Theta, Weights};
use crate::estimation::{EstimationProblem, Framework};
use crate::results::FitResult;

use anyhow::Result;

use pharmsol::prelude::data::{AssayErrorModels, Data};
use pharmsol::prelude::simulator::Equation;

use serde::{Deserialize, Serialize};

pub mod diagnostics;
pub mod nonparametric;
pub mod parametric;
mod stop;

/// Defines an algorithm that can fit an [`EstimationProblem`] to produce a result.
///
/// Implementors are the lightweight, user-facing configuration structs (e.g.
/// [`NpagConfig`](nonparametric::NpagConfig)). The heavy, mutable execution
/// state used while fitting is an internal implementation detail.
pub trait Algorithm<P> {
    /// The specific result struct (e.g. `NonParametricResult<E>`).
    type Output: FitResult;

    /// Consumes the configuration and the problem, runs the optimization to
    /// completion, and returns the strictly-typed result.
    fn fit(self, problem: P) -> Result<Self::Output>;
}

impl<E: Equation, F: Framework> EstimationProblem<E, F> {
    /// Consumes the problem and an algorithm configuration, runs the fit to
    /// completion, and returns the result.
    pub fn fit_with<A>(self, algorithm: A) -> Result<A::Output>
    where
        A: Algorithm<Self>,
    {
        algorithm.fit(self)
    }
}

/// Read-only view of the state of a running fit.
///
/// A runner exposes this state between cycles. Log-likelihoods are reported on
/// their natural scale: [`log_likelihood`](Self::log_likelihood) returns the
/// log-likelihood of the data under the current model (higher is better), and
/// [`n2ll`](Self::n2ll) the objective function actually minimized,
/// `-2 × log_likelihood` (lower is better). Runners that converge on a
/// different statistic (NCNPAG's marginal log-likelihood) report that statistic
/// instead, so `n2ll` is always the quantity being optimized.
pub trait FitState<E: Equation> {
    /// Get the equation used in the algorithm
    fn equation(&self) -> &E;
    /// Get the data used in the algorithm
    fn data(&self) -> &Data;
    /// Get the error models used in the algorithm
    fn error_models(&self) -> &AssayErrorModels;
    /// Get the current [Theta]
    fn theta(&self) -> &Theta;
    /// Get the current [Psi]
    fn psi(&self) -> &Psi;
    /// Get the current support point weights
    fn weights(&self) -> &Weights;
    /// Get the current cycle number
    fn cycle(&self) -> usize;
    /// Get the current [Status] of the algorithm
    fn status(&self) -> &Status;
    /// Get the current log-likelihood (higher is better)
    fn log_likelihood(&self) -> f64;
    /// Get the current objective function, `-2 × log_likelihood` (lower is better)
    fn n2ll(&self) -> f64 {
        -2.0 * self.log_likelihood()
    }
}

/// The mutable state of a non-parametric fit, advanced one cycle at a time.
///
/// Most algorithms run the same cycle: estimate the likelihood of every support
/// point, condense the support points, optimize the error models, then evaluate
/// the convergence criteria. Stages that don't apply default to no-ops, so an
/// algorithm only implements what it needs. Single-pass algorithms (NPMAP,
/// NCNPAG) implement `estimation` and `evaluation`, and report
/// [`StopReason::Converged`] once their single pass is done.
///
/// Only [`next_cycle`](Self::next_cycle) writes the [`Status`], so the status an
/// observer sees always matches the state of the fit.
pub trait NonParametricRunner<E: Equation + Send + 'static>: FitState<E> + Send + 'static {
    /// The result produced when the fit finishes.
    type Output: FitResult;

    /// Set the current [Status] of the algorithm
    fn set_status(&mut self, status: Status);
    /// Increment the cycle counter and return the new value
    fn increment_cycle(&mut self) -> usize;
    /// Evaluate the convergence criteria and return the resulting [Status]
    ///
    /// Implementations must not store the status they return:
    /// [`next_cycle`](Self::next_cycle) is the single writer.
    fn evaluation(&mut self) -> Result<Status>;
    /// Estimate the likelihood of every support point for every subject
    fn estimation(&mut self) -> Result<()>;
    /// Consume the runner and build the final result
    #[allow(clippy::wrong_self_convention)]
    fn into_result(self: Box<Self>) -> Result<Self::Output>;
    /// Append a cycle to the cycle log
    fn push_cycle(&mut self, cycle: NPCycle);

    /// Performs condensation of [Theta] and updates [Psi]
    ///
    /// This step reduces the number of support points in [Theta] based on the current weights,
    /// and updates the [Psi] matrix accordingly to reflect the new set of support points.
    /// It is typically performed after the estimation step in each cycle of the algorithm.
    fn condensation(&mut self) -> Result<()> {
        Ok(())
    }
    /// Performs optimizations on the current `AssayErrorModels` and updates [Psi] accordingly
    ///
    /// This step refines the error model parameters to better fit the data,
    /// and subsequently updates the [Psi] matrix to reflect these changes.
    fn optimizations(&mut self) -> Result<()> {
        Ok(())
    }
    /// Performs expansion of [Theta]
    ///
    /// This step increases the number of support points in [Theta] based on the current distribution,
    /// allowing for exploration of the parameter space.
    fn expansion(&mut self) -> Result<()> {
        Ok(())
    }
    /// Change in the objective function since the last logged cycle
    ///
    /// Called once per logged cycle, before the cycle is pushed to the log.
    /// Implementations may use this to update their own bookkeeping.
    fn objective_delta(&mut self) -> f64 {
        0.0
    }

    /// Initialize the algorithm, setting up initial [Theta] and [Status]
    fn initialize(&mut self) -> Result<()> {
        // If a stop file exists in the current directory, remove it
        crate::algorithms::stop::remove_stop_file()?;
        self.set_status(Status::Continue);

        Ok(())
    }

    /// Create and log a cycle state with the current algorithm state
    fn log_cycle_state(&mut self) {
        let delta = self.objective_delta();
        let state = NPCycle::new(
            self.cycle(),
            self.n2ll(),
            self.error_models().clone(),
            self.theta().clone(),
            self.weights().clone(),
            delta,
            self.status().clone(),
        );
        self.push_cycle(state);
    }

    /// Proceed to the next cycle of the algorithm
    ///
    /// This method increments the cycle counter, performs expansion if necessary,
    /// and then runs the estimation, condensation, optimization, logging, and evaluation steps
    /// in sequence. It returns the [Status] of the algorithm after completing these steps.
    ///
    /// A stop is final: once the runner has stopped, further cycles return the
    /// same [Status] without touching the fit state.
    fn next_cycle(&mut self) -> Result<Status> {
        if self.status().is_stop() {
            return Ok(self.status().clone());
        }

        let cycle = self.increment_cycle();

        if cycle > 1 {
            self.expansion()?;
        }

        let span = tracing::info_span!("", "{}", format!("Cycle {}", self.cycle()));
        let _enter = span.enter();
        self.estimation()?;
        self.condensation()?;
        self.optimizations()?;
        let status = self.evaluation()?;
        self.set_status(status.clone());
        self.log_cycle_state();
        Ok(status)
    }

    /// Fit the model until convergence or stopping criteria are met
    ///
    /// This method runs the full fitting process, starting with initialization,
    /// followed by iterative cycles of estimation, condensation, optimization, and evaluation
    /// until the algorithm converges or meets a stopping criteria.
    fn fit(self: Box<Self>) -> Result<Self::Output> {
        let mut runner = self;
        runner.initialize()?;
        while runner.next_cycle()?.is_continue() {}
        runner.into_result()
    }
}

/// Where a fit stands: still running, or stopped (and why).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    Continue,
    Stop(StopReason),
}

impl Status {
    /// Whether the fit is still running.
    pub fn is_continue(&self) -> bool {
        matches!(self, Status::Continue)
    }

    /// Whether the fit has stopped.
    pub fn is_stop(&self) -> bool {
        matches!(self, Status::Stop(_))
    }

    /// Why the fit stopped, or `None` if it's still running.
    pub fn stop_reason(&self) -> Option<&StopReason> {
        match self {
            Status::Stop(reason) => Some(reason),
            Status::Continue => None,
        }
    }

    /// Whether the fit stopped because it converged, rather than being cut short.
    pub fn converged(&self) -> bool {
        matches!(self, Status::Stop(StopReason::Converged))
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Continue => write!(f, "Continue"),
            Status::Stop(reason) => write!(f, "Stopped ({reason})"),
        }
    }
}

/// Why a fit stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// The convergence criteria were met.
    Converged,
    /// Hit the cycle limit before converging.
    MaxCycles,
    /// A `stop` file was found on disk.
    StopFile,
    /// Stopped from code — [`request_stop`](crate::algorithms::nonparametric::FitController::request_stop)
    /// or an observer returning [`CycleFlow::Stop`](crate::algorithms::nonparametric::CycleFlow::Stop).
    Aborted,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match self {
            StopReason::Converged => "converged",
            StopReason::MaxCycles => "maximum cycles reached",
            StopReason::StopFile => "stop file detected",
            StopReason::Aborted => "aborted",
        };
        f.write_str(reason)
    }
}
