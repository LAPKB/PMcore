use std::fs;
use std::path::Path;

use crate::estimation::nonparametric::{NonParametricResult, Psi, Theta};
use crate::estimation::{EstimationProblem, Framework};
use crate::results::FitResult;

use anyhow::Context;
use anyhow::Result;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};

use pharmsol::prelude::{data::Data, simulator::Equation};

use pharmsol::{Predictions, Subject};
use serde::{Deserialize, Serialize};

/// Defines an algorithm that can fit an [`EstimationProblem`] to produce a result.
///
/// Implementors are the lightweight, user-facing configuration structs (e.g.
/// `NpagConfig`). The heavy, mutable execution state used while fitting is an
/// internal implementation detail.
pub trait Algorithm<E: Equation, F: crate::estimation::Framework> {
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

pub trait NonParametricRunner<E: Equation + Send + 'static>: Sync + Send + 'static {
    /// Identify subjects whose total probability given the model is zero or
    /// non-finite.
    ///
    /// Each row of [`Psi`] holds the likelihood of a subject across every
    /// support point, so a subject's probability is the sum across its row. A
    /// subject is flagged when that sum is zero or not finite, meaning the model
    /// cannot explain the subject's data. When any subject is flagged, detailed
    /// per-subject diagnostics are logged and an error is returned.
    fn check_zero_probability_subjects(&self) -> Result<()> {
        let psi = self.psi().matrix();

        // Report non-finite entries; these propagate into the row sums below.
        let nonfinite = psi
            .row_iter()
            .flat_map(|row| row.iter().copied())
            .filter(|v| !v.is_finite())
            .count();
        if nonfinite > 0 {
            tracing::warn!(
                "Psi matrix contains {} non-finite value(s) of {} total",
                nonfinite,
                psi.nrows() * psi.ncols()
            );
        }

        // A subject's probability is the sum across its row.
        let subjects = self.data().subjects();
        let flagged: Vec<usize> = (0..psi.nrows())
            .filter(|&i| {
                let probability: f64 = (0..psi.ncols()).map(|j| psi[(i, j)]).sum();
                !probability.is_finite() || probability == 0.0
            })
            .collect();

        if flagged.is_empty() {
            return Ok(());
        }

        tracing::error!(
            "{}/{} subjects have zero probability given the model",
            flagged.len(),
            psi.nrows()
        );

        for &i in &flagged {
            self.log_zero_probability_subject(subjects[i]);
        }

        let ids: Vec<&String> = flagged.iter().map(|&i| subjects[i].id()).collect();
        Err(anyhow::anyhow!(
            "The probability of {}/{} subjects is zero given the model. Affected subjects: {:?}",
            flagged.len(),
            psi.nrows(),
            ids
        ))
    }

    /// Log detailed likelihood diagnostics for a single subject whose
    /// probability given the model is zero or non-finite.
    fn log_zero_probability_subject(&self, subject: &Subject) {
        tracing::debug!("Subject with zero probability: {}", subject.id());

        let error_model = self.error_models().clone();

        // Simulate every support point for this subject in parallel.
        let mut results: Vec<_> = self
            .theta()
            .matrix()
            .row_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(i, spp)| {
                let support_point: Vec<f64> = spp.iter().copied().collect();
                let (pred, ll) = self
                    .equation()
                    .simulate_subject_dense(subject, &support_point, Some(&error_model))
                    .unwrap(); //TODO: Handle error
                (i, support_point, pred.get_predictions(), ll)
            })
            .collect();

        // Summarise the distribution of likelihood values.
        let mut nan = 0;
        let mut pos_inf = 0;
        let mut neg_inf = 0;
        let mut zero = 0;
        let mut valid = 0;
        for (_, _, _, ll) in &results {
            match ll {
                Some(v) if v.is_nan() => nan += 1,
                Some(v) if v.is_infinite() && v.is_sign_positive() => pos_inf += 1,
                Some(v) if v.is_infinite() => neg_inf += 1,
                Some(v) if *v == 0.0 => zero += 1,
                Some(_) => valid += 1,
                None => nan += 1,
            }
        }

        let total = results.len();
        let pct = |n: usize| 100.0 * n as f64 / total as f64;
        tracing::debug!(
            "\tLikelihood analysis for subject {} ({} support points):",
            subject.id(),
            total
        );
        tracing::debug!("\tNaN likelihoods: {} ({:.1}%)", nan, pct(nan));
        tracing::debug!("\t+Inf likelihoods: {} ({:.1}%)", pos_inf, pct(pos_inf));
        tracing::debug!("\t-Inf likelihoods: {} ({:.1}%)", neg_inf, pct(neg_inf));
        tracing::debug!("\tZero likelihoods: {} ({:.1}%)", zero, pct(zero));
        tracing::debug!("\tValid likelihoods: {} ({:.1}%)", valid, pct(valid));

        // Show the most likely support points to aid debugging.
        results.sort_by(|a, b| {
            b.3.unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(&a.3.unwrap_or(f64::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        const TAKE: usize = 3;
        tracing::debug!("Top {} most likely support points:", TAKE);
        for (i, support_point, preds, ll) in results.iter().take(TAKE) {
            tracing::debug!("\tSupport point #{}: {:?}", i, support_point);
            tracing::debug!("\t\tLog-likelihood: {:?}", ll);
            tracing::debug!(
                "\t\tTimes: {:?}",
                preds.iter().map(|x| x.time()).collect::<Vec<f64>>()
            );
            tracing::debug!(
                "\t\tObservations: {:?}",
                preds
                    .iter()
                    .map(|x| x.observation())
                    .collect::<Vec<Option<f64>>>()
            );
            tracing::debug!(
                "\t\tPredictions: {:?}",
                preds.iter().map(|x| x.prediction()).collect::<Vec<f64>>()
            );
            tracing::debug!(
                "\t\tOuteqs: {:?}",
                preds.iter().map(|x| x.outeq()).collect::<Vec<usize>>()
            );
            tracing::debug!(
                "\t\tStates: {:?}",
                preds
                    .iter()
                    .map(|x| x.state().to_vec())
                    .collect::<Vec<Vec<f64>>>()
            );
        }
        tracing::debug!("=====================");
    }

    fn error_models(&self) -> &pharmsol::prelude::data::AssayErrorModels;
    /// Get the equation used in the algorithm
    fn equation(&self) -> &E;
    /// Get the data used in the algorithm
    fn data(&self) -> &Data;

    /// Increment the cycle counter and return the new value
    fn increment_cycle(&mut self) -> usize;
    /// Get the current cycle number
    fn cycle(&self) -> usize;
    /// Set the current [Theta]
    fn set_theta(&mut self, theta: Theta);
    /// Get the current [Theta]
    fn theta(&self) -> &Theta;
    /// Get the current [Psi]
    fn psi(&self) -> &Psi;
    /// Get the current likelihood
    fn likelihood(&self) -> f64;
    /// Get the current negative two log-likelihood
    fn n2ll(&self) -> f64 {
        -2.0 * self.likelihood()
    }
    /// Get the current [Status] of the algorithm
    fn status(&self) -> &Status;
    /// Set the current [Status] of the algorithm
    fn set_status(&mut self, status: Status);
    /// Evaluate convergence criteria and update status
    fn evaluation(&mut self) -> Result<Status>;

    /// Create and log a cycle state with the current algorithm state
    fn log_cycle_state(&mut self);

    /// Initialize the algorithm, setting up initial [Theta] and [Status]
    fn initialize(&mut self) -> Result<()> {
        // If a stop file exists in the current directory, remove it
        if Path::new("stop").exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file("stop").context("Unable to remove previous stop file")?;
        }
        self.set_status(Status::Continue);

        Ok(())
    }
    fn estimation(&mut self) -> Result<()>;
    /// Performs condensation of [Theta] and updates [Psi]
    ///
    /// This step reduces the number of support points in [Theta] based on the current weights,
    /// and updates the [Psi] matrix accordingly to reflect the new set of support points.
    /// It is typically performed after the estimation step in each cycle of the algorithm.
    fn condensation(&mut self) -> Result<()>;

    /// Performs optimizations on the current [AssayErrorModels] and updates [Psi] accordingly
    ///
    /// This step refines the error model parameters to better fit the data,
    /// and subsequently updates the [Psi] matrix to reflect these changes.
    fn optimizations(&mut self) -> Result<()>;

    /// Performs expansion of [Theta]
    ///
    /// This step increases the number of support points in [Theta] based on the current distribution,
    /// allowing for exploration of the parameter space.
    fn expansion(&mut self) -> Result<()>;

    /// Proceed to the next cycle of the algorithm
    ///
    /// This method increments the cycle counter, performs expansion if necessary,
    /// and then runs the estimation, condensation, optimization, logging, and evaluation steps
    /// in sequence. It returns the current [Status] of the algorithm after completing these steps.
    fn next_cycle(&mut self) -> Result<Status> {
        let cycle = self.increment_cycle();

        if cycle > 1 {
            self.expansion()?;
        }

        let span = tracing::info_span!("", "{}", format!("Cycle {}", self.cycle()));
        let _enter = span.enter();
        self.estimation()?;
        self.condensation()?;
        self.optimizations()?;
        self.evaluation()
    }

    /// Fit the model until convergence or stopping criteria are met
    ///
    /// This method runs the full fitting process, starting with initialization,
    /// followed by iterative cycles of estimation, condensation, optimization, and evaluation
    /// until the algorithm converges or meets a stopping criteria.
    fn fit(&mut self) -> Result<NonParametricResult<E>> {
        self.initialize()?;
        while let Status::Continue = self.next_cycle()? {}
        self.into_result()
    }

    #[allow(clippy::wrong_self_convention)]
    fn into_result(&self) -> Result<NonParametricResult<E>>;
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
