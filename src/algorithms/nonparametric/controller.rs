//! Step through a fit cycle by cycle, rather than running it all at once.
//!
//! [`fit_with`](crate::estimation::EstimationProblem::fit_with) runs to the end and gives back
//! only the final result. For debugging or monitoring, start a [`FitController`] with
//! [`fit_controller`](crate::estimation::EstimationProblem::fit_controller) instead, and either
//! advance it one [`step`](FitController::step) at a time or let it
//! [`run_until`](FitController::run_until) a condition of yours. Every route runs the same fit
//! and ends in the same result.

use anyhow::Result;
use pharmsol::prelude::{data::AssayErrorModels, simulator::Equation};

use crate::algorithms::Status;
use crate::estimation::nonparametric::{NonParametricResult, Psi, Theta, Weights};
use crate::estimation::{EstimationProblem, NonParametric};

use super::fit::Fit;
use super::NonParametricAlgorithm;

/// A running fit that you advance one cycle at a time.
///
/// It's ready to go as soon as it's built, so the first [`step`](Self::step) runs cycle 1.
/// The accessors read the current state between steps. Once the fit has stopped, whether on
/// its own or through [`request_stop`](Self::request_stop), it stays stopped: further steps do
/// nothing. [`finish`](Self::finish), [`run_until`](Self::run_until) and
/// [`into_result`](Self::into_result) turn it into a result.
pub struct FitController<E: Equation> {
    fit: Fit<E>,
}

impl<E: Equation> FitController<E> {
    /// Build a controller and initialize it, ready for the first [`step`](Self::step).
    pub(crate) fn new(
        algorithm: NonParametricAlgorithm,
        problem: EstimationProblem<E, NonParametric>,
    ) -> Result<Self> {
        Ok(Self {
            fit: algorithm.start(problem)?,
        })
    }

    /// Run one cycle and return the new [`Status`].
    ///
    /// [`Continue`](Status::Continue) means there's more to do. [`Stop`](Status::Stop) means the
    /// fit is finished: from then on, `step` does nothing and returns the same status.
    pub fn step(&mut self) -> Result<Status> {
        self.fit.step()
    }

    /// Stop the fit now, as [`Aborted`](crate::algorithms::StopReason::Aborted).
    ///
    /// Further steps do nothing. A fit that has already stopped keeps its original reason.
    pub fn request_stop(&mut self) {
        self.fit.request_stop();
    }

    /// The current cycle number (0 before the first [`step`](Self::step)).
    pub fn cycle(&self) -> usize {
        self.fit.state().cycle()
    }

    /// The current [`Status`] of the fit.
    pub fn status(&self) -> &Status {
        self.fit.state().status()
    }

    /// The current support points.
    pub fn theta(&self) -> &Theta {
        self.fit.state().solution().theta()
    }

    /// The current weight of each support point in [`theta`](Self::theta).
    pub fn weights(&self) -> &Weights {
        self.fit.state().solution().weights()
    }

    /// The current likelihood matrix.
    pub fn psi(&self) -> &Psi {
        self.fit.state().solution().psi()
    }

    /// The current log-likelihood.
    pub fn log_likelihood(&self) -> f64 {
        self.fit.state().solution().log_likelihood()
    }

    /// The current objective function, -2 × log-likelihood, as reported by
    /// [`NonParametricResult::objf`].
    pub fn objf(&self) -> f64 {
        self.fit.state().objf()
    }

    /// The current error models (updated when the algorithm optimizes them).
    pub fn error_models(&self) -> &AssayErrorModels {
        self.fit.state().error_models()
    }

    /// Run the fit to completion and return the result.
    ///
    /// Safe to call anytime: if the fit has already stopped, it just returns the result.
    pub fn finish(mut self) -> Result<NonParametricResult<E>> {
        self.fit.run()?;
        self.fit.into_result()
    }

    /// Run until the fit stops on its own or `stop` returns `true`, then return the result.
    ///
    /// `stop` is called after every cycle, the last one included, so it can also report
    /// progress. If it returns `true` while the fit is still running, the fit stops as
    /// [`Aborted`](crate::algorithms::StopReason::Aborted).
    ///
    /// ```no_run
    /// use pmcore::prelude::*;
    /// # fn run(problem: EstimationProblem<pmcore::prelude::ODE, NonParametric>) -> Result<()> {
    /// let result = problem
    ///     .fit_controller(NpagConfig::new())?
    ///     .run_until(|fit| {
    ///         println!("cycle {} | -2LL {:.4}", fit.cycle(), fit.objf());
    ///         fit.cycle() >= 50
    ///     })?;
    /// # let _ = result; Ok(())
    /// # }
    /// ```
    pub fn run_until(
        mut self,
        mut stop: impl FnMut(&Self) -> bool,
    ) -> Result<NonParametricResult<E>> {
        while self.status().is_continue() {
            self.step()?;
            if stop(&self) {
                self.request_stop();
            }
        }
        self.into_result()
    }

    /// Turn the current state into a result as-is, without running more cycles.
    pub fn into_result(self) -> Result<NonParametricResult<E>> {
        self.fit.into_result()
    }
}

impl<E: Equation> EstimationProblem<E, NonParametric> {
    /// Start a fit you drive yourself, one [`step`](FitController::step) at a time.
    ///
    /// Takes any non-parametric config (`NpagConfig`, `NpodConfig`, `NpmapConfig`, `NcnpagConfig`) or a
    /// [`NonParametricAlgorithm`].
    ///
    /// ```no_run
    /// use pmcore::prelude::*;
    /// # fn run(problem: EstimationProblem<pmcore::prelude::ODE, NonParametric>) -> Result<()> {
    /// let mut controller = problem.fit_controller(NpagConfig::new())?;
    /// while controller.step()?.is_continue() {
    ///     println!("cycle {} | -2LL {:.4}", controller.cycle(), controller.objf());
    /// }
    /// let result = controller.into_result()?;
    /// # let _ = result; Ok(())
    /// # }
    /// ```
    pub fn fit_controller(
        self,
        algorithm: impl Into<NonParametricAlgorithm>,
    ) -> Result<FitController<E>> {
        FitController::new(algorithm.into(), self)
    }
}
