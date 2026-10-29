//! Step through a fit cycle by cycle, rather than running it all at once.
//!
//! [`fit_with`](crate::estimation::EstimationProblem::fit_with) runs to the end and gives back
//! only the final result. For debugging or third-party monitoring, you can instead drive the fit yourself and inspect the state between cycles.
//!
//! - [`FitController`] — hold the fit and advance it one [`step`](FitController::step) at a
//!   time, inspecting state in between.
//! - [`fit_with_observer`](crate::estimation::EstimationProblem::fit_with_observer) — let it
//!   run, but get a callback after every cycle.
//!
//! Both drive the same underlying fit and finish at the same result.

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
/// The accessors read the current state between steps, and [`finish`](Self::finish) or
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
    /// [`Continue`](Status::Continue) means there's more to do; [`Stop`](Status::Stop) means the
    /// algorithm is finished and further steps won't change the outcome.
    pub fn step(&mut self) -> Result<Status> {
        self.fit.step()
    }

    /// Flag the fit as [`Aborted`](crate::algorithms::StopReason::Aborted) so the result shows an early stop.
    ///
    /// This only sets the status; it won't halt a later [`step`](Self::step), which runs a full
    /// cycle and overwrites it. Call it just before [`into_result`](Self::into_result).
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

    /// The current log-likelihood.
    #[deprecated(note = "renamed to `log_likelihood`")]
    pub fn likelihood(&self) -> f64 {
        self.log_likelihood()
    }

    /// The current objective function, -2 × log-likelihood.
    #[deprecated(note = "renamed to `objf`, matching `NonParametricResult::objf`")]
    pub fn n2ll(&self) -> f64 {
        self.objf()
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

    /// Turn the current state into a result as-is, without running more cycles.
    pub fn into_result(self) -> Result<NonParametricResult<E>> {
        self.fit.into_result()
    }
}

/// An observer's verdict after a cycle: keep going, or stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleFlow {
    /// Keep going — the algorithm may still stop on its own.
    Continue,
    /// Stop now.
    Stop,
}

/// A callback run after every cycle of a fit.
///
/// It receives the live [`FitController`] to read the current state, and returns a
/// [`CycleFlow`] to keep going or bail out. Any `FnMut(&FitController<E>) -> CycleFlow` works,
/// so you can hand a plain closure to
/// [`fit_with_observer`](crate::estimation::EstimationProblem::fit_with_observer).
pub trait FitObserver<E: Equation> {
    fn on_cycle(&mut self, controller: &FitController<E>) -> CycleFlow;
}

impl<E, F> FitObserver<E> for F
where
    E: Equation,
    F: FnMut(&FitController<E>) -> CycleFlow,
{
    fn on_cycle(&mut self, controller: &FitController<E>) -> CycleFlow {
        self(controller)
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

    /// Run a fit to completion, calling `observer` after every cycle.
    ///
    /// The observer sees every cycle, the last one included, and can return
    /// [`CycleFlow::Stop`] to bail out early. Returns the same result as
    /// [`fit_with`](Self::fit_with).
    ///
    /// ```no_run
    /// use pmcore::prelude::*;
    /// # fn run(problem: EstimationProblem<pmcore::prelude::ODE, NonParametric>) -> Result<()> {
    /// let result = problem.fit_with_observer(NpagConfig::new(), |c: &FitController<_>| {
    ///     println!("cycle {} | -2LL {:.4}", c.cycle(), c.objf());
    ///     CycleFlow::Continue
    /// })?;
    /// # let _ = result; Ok(())
    /// # }
    /// ```
    pub fn fit_with_observer<O: FitObserver<E>>(
        self,
        algorithm: impl Into<NonParametricAlgorithm>,
        mut observer: O,
    ) -> Result<NonParametricResult<E>> {
        let mut controller = self.fit_controller(algorithm)?;
        loop {
            let status = controller.step()?;
            let flow = observer.on_cycle(&controller);
            // Stop on the algorithm's own criteria, or when the observer asks us to.
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
