use crate::estimation::{EstimationProblem, Framework};
use crate::results::FitResult;

use anyhow::Result;
use pharmsol::prelude::simulator::Equation;
use serde::{Deserialize, Serialize};

/// Defines an algorithm that can fit an [`EstimationProblem`] to produce a result.
///
/// Implementors are the lightweight, user-facing configuration structs (e.g.
/// `NpagConfig`). The heavy, mutable execution state used while fitting is an
/// internal implementation detail.
pub trait Algorithm<E: Equation, F: Framework> {
    /// The specific result struct (e.g. `NonParametricResult<E>`).
    type Output: FitResult;

    /// Consumes the configuration and the problem, runs the optimization to
    /// completion, and returns the strictly-typed result.
    fn fit(self, problem: EstimationProblem<E, F>) -> Result<Self::Output>;
}

// Module organization for algorithm types
pub mod nonparametric;
pub mod parametric;

impl<E: Equation, F: Framework> EstimationProblem<E, F> {
    /// Consumes the problem and an algorithm configuration, runs the fit to
    /// completion, and returns the result.
    pub fn fit_with<A>(self, algorithm: A) -> Result<A::Output>
    where
        A: Algorithm<E, F>,
    {
        algorithm.fit(self)
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
    /// Stopped from code, with
    /// [`FitController::request_stop`](crate::algorithms::nonparametric::FitController::request_stop)
    /// or [`FitController::run_until`](crate::algorithms::nonparametric::FitController::run_until).
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
