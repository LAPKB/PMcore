//! The machinery every non-parametric algorithm runs on.
//!
//! A fit is split in two:
//!
//! - [`FitState`] describes the fit at a point in time: the problem (equation, data, error
//!   models, prior), the current support points with their likelihoods and weights, and the
//!   cycle bookkeeping. It is the same for every algorithm.
//! - A [`Runner`] holds only what is specific to one algorithm (its configuration and any
//!   adaptive state, such as NPAG's grid resolution) and knows how to run a single cycle.
//!
//! [`Fit`] pairs the two and owns the cycle loop. Both
//! [`fit_with`](crate::estimation::EstimationProblem::fit_with) and the stepping
//! [`FitController`](super::FitController) are thin wrappers around it.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use pharmsol::prelude::{
    data::{AssayErrorModel, AssayErrorModels, Data},
    simulator::Equation,
};
use pharmsol::{Predictions, Subject};
use rayon::prelude::*;

use crate::algorithms::{Status, StopReason};
use crate::estimation::nonparametric::{
    calculate_psi, ipm::burke, qr, CycleLog, NPCycle, NonParametricResult, Psi, Theta, Weights,
};
use crate::estimation::{EstimationProblem, NonParametric};

/// A file with this name in the working directory stops a running fit after the current cycle.
const STOP_FILE: &str = "stop";

/// One non-parametric algorithm: what a single cycle does to a [`FitState`].
pub(crate) trait Runner<E: Equation>: Send + Sync {
    /// Run cycle `state.cycle`, updating `state` in place.
    ///
    /// Returns [`Status::Continue`] to ask for another cycle, or [`Status::Stop`] once the
    /// algorithm is finished. Recording the cycle in the log is left to [`Fit`].
    fn step(&mut self, state: &mut FitState<E>) -> Result<Status>;
}

/// The state of a non-parametric fit, shared by every algorithm.
#[derive(Debug)]
pub(crate) struct FitState<E: Equation> {
    pub(crate) equation: E,
    pub(crate) data: Data,
    pub(crate) error_models: AssayErrorModels,
    /// The support points the fit started from.
    pub(crate) prior: Theta,
    /// The current support points.
    pub(crate) theta: Theta,
    /// Likelihood of each subject (row) at each support point (column) of `theta`.
    pub(crate) psi: Psi,
    /// Probability of each support point in `theta`.
    pub(crate) weights: Weights,
    /// Log-likelihood of the data given `theta` and `weights`.
    pub(crate) log_likelihood: f64,
    /// `log_likelihood` at the end of the previous cycle, or `None` during the first one.
    pub(crate) previous_log_likelihood: Option<f64>,
    /// The current cycle, starting at 1 (0 before the first cycle runs).
    pub(crate) cycle: usize,
    pub(crate) status: Status,
    cycle_log: CycleLog,
}

impl<E: Equation> FitState<E> {
    pub(crate) fn new(problem: EstimationProblem<E, NonParametric>) -> Self {
        Self::from_parts(
            problem.model.equation,
            problem.data,
            problem.error_models,
            problem.prior,
        )
    }

    pub(crate) fn from_parts(
        equation: E,
        data: Data,
        error_models: AssayErrorModels,
        prior: Theta,
    ) -> Self {
        Self {
            equation,
            data,
            error_models,
            theta: prior.clone(),
            prior,
            psi: Psi::new(),
            weights: Weights::default(),
            log_likelihood: f64::NEG_INFINITY,
            previous_log_likelihood: None,
            cycle: 0,
            status: Status::Continue,
            cycle_log: CycleLog::new(),
        }
    }

    /// The objective function, -2 × log-likelihood.
    pub(crate) fn n2ll(&self) -> f64 {
        -2.0 * self.log_likelihood
    }

    /// Absolute change in log-likelihood since the previous cycle (`None` in the first cycle).
    pub(crate) fn objective_change(&self) -> Option<f64> {
        self.previous_log_likelihood
            .map(|previous| (previous - self.log_likelihood).abs())
    }

    /// Compute [`Psi`] for the current support points and solve for their optimal weights.
    ///
    /// Only `psi` is stored; the weights are returned so the caller can
    /// [`condense`](Self::condense) on them.
    pub(crate) fn estimate(&mut self, progress: bool) -> Result<Weights> {
        self.psi = calculate_psi(
            &self.equation,
            &self.data,
            &self.theta,
            &self.error_models,
            progress,
        )?;
        self.check_zero_probability_subjects()?;
        let (lambda, _) = burke(&self.psi).context("IPM failed during estimation")?;
        Ok(lambda)
    }

    /// Drop support points that carry negligible weight or are linearly dependent, then
    /// re-solve the weights (and log-likelihood) over the survivors.
    ///
    /// A point is dropped when its weight in `lambda` is at most `prune_threshold` × the largest
    /// weight, or when its column of [`Psi`] is (numerically) a combination of the others, as
    /// judged by a rank-revealing QR decomposition with tolerance `qr_tolerance`.
    pub(crate) fn condense(
        &mut self,
        lambda: &Weights,
        prune_threshold: f64,
        qr_tolerance: f64,
    ) -> Result<()> {
        let max_lambda = lambda.iter().fold(f64::NEG_INFINITY, f64::max);
        let keep: Vec<usize> = lambda
            .iter()
            .enumerate()
            .filter(|&(_, lam)| lam > max_lambda * prune_threshold)
            .map(|(i, _)| i)
            .collect();
        self.retain_support_points(&keep, "Lambda (max/1000)");

        let (r, perm) = qr::qrd(&self.psi)?;
        let rank_bound = self.psi.matrix().ncols().min(self.psi.matrix().nrows());
        let keep: Vec<usize> = (0..rank_bound)
            .filter(|&i| (r.get(i, i) / r.col(i).norm_l2()).abs() >= qr_tolerance)
            .map(|i| perm[i])
            .collect();
        self.retain_support_points(&keep, "QR decomposition");

        self.check_zero_probability_subjects()?;
        (self.weights, self.log_likelihood) =
            burke(&self.psi).context("IPM failed during condensation")?;
        Ok(())
    }

    /// Keep only the support points at `keep`, in `theta` and `psi` alike.
    fn retain_support_points(&mut self, keep: &[usize], step: &str) {
        let dropped = self.theta.nspp() - keep.len();
        if dropped > 0 {
            tracing::debug!("{} dropped {} support point(s)", step, dropped);
        }
        self.theta.filter_indices(keep);
        self.psi.filter_column_indices(keep);
    }

    /// Log the objective function and error models at the end of a cycle, and warn if the
    /// objective got worse.
    pub(crate) fn log_cycle_summary(&self) {
        tracing::info!("Objective function = {:.4}", self.n2ll());
        tracing::debug!("Support points: {}", self.theta.nspp());

        for (outeq, em) in self.error_models.iter() {
            if *em != AssayErrorModel::None {
                tracing::debug!(
                    "Error model for outeq {}: {:.4}",
                    outeq,
                    em.factor().unwrap_or_default()
                );
            }
        }

        // A falling log-likelihood signals instability or model misspecification.
        if let Some(previous) = self.previous_log_likelihood {
            if previous > self.log_likelihood + 1e-4 {
                tracing::warn!(
                    "Objective function decreased from {:.4} to {:.4} (delta = {})",
                    -2.0 * previous,
                    self.n2ll(),
                    -2.0 * previous - self.n2ll()
                );
            }
        }
    }

    /// Why the fit must stop regardless of convergence: the cycle cap was reached, or a stop
    /// file was found.
    pub(crate) fn limit_reached(&self, max_cycles: usize) -> Option<StopReason> {
        if self.cycle >= max_cycles {
            tracing::warn!("Maximum number of cycles reached");
            return Some(StopReason::MaxCycles);
        }
        if Path::new(STOP_FILE).exists() {
            tracing::warn!("Stopfile detected - breaking");
            return Some(StopReason::StopFile);
        }
        None
    }

    /// Append the current cycle to the cycle log.
    fn record_cycle(&mut self) {
        let delta = self.objective_change().unwrap_or(f64::INFINITY);
        self.cycle_log.push(NPCycle::new(
            self.cycle,
            self.n2ll(),
            self.error_models.clone(),
            self.theta.clone(),
            self.weights.clone(),
            self.theta.nspp(),
            delta,
            self.status.clone(),
        ));
        self.previous_log_likelihood = Some(self.log_likelihood);
    }

    pub(crate) fn into_result(self) -> Result<NonParametricResult<E>> {
        NonParametricResult::new(
            self.equation,
            self.data,
            self.error_models,
            self.prior,
            self.theta,
            self.psi,
            self.weights,
            -2.0 * self.log_likelihood,
            self.cycle,
            self.status,
            self.cycle_log,
        )
    }

    /// Fail if any subject has zero (or non-finite) probability given the model.
    ///
    /// Each row of [`Psi`] holds a subject's likelihood across every support point, so a
    /// subject's probability is the sum of its row. When any subject is flagged, per-subject
    /// diagnostics are logged and an error naming the subjects is returned.
    fn check_zero_probability_subjects(&self) -> Result<()> {
        let psi = self.psi.matrix();

        // Non-finite entries propagate into the row sums below.
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

        let subjects = self.data.subjects();
        let flagged: Vec<usize> = (0..psi.nrows())
            .filter(|&i| {
                let probability: f64 = psi.row(i).iter().sum();
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
        anyhow::bail!(
            "The probability of {}/{} subjects is zero given the model. Affected subjects: {:?}",
            flagged.len(),
            psi.nrows(),
            ids
        )
    }

    /// Log likelihood diagnostics for a subject with zero probability given the model.
    fn log_zero_probability_subject(&self, subject: &Subject) {
        tracing::debug!("Subject with zero probability: {}", subject.id());

        // Simulate every support point for this subject. Points that fail to simulate count
        // as having no likelihood.
        let mut results: Vec<_> = (0..self.theta.nspp())
            .into_par_iter()
            .map(|i| {
                let support_point: Vec<f64> = self.theta.matrix().row(i).iter().copied().collect();
                match self
                    .equation
                    .simulate(subject, &support_point, Some(&self.error_models))
                {
                    Ok((pred, ll)) => (i, support_point, pred.get_predictions(), ll),
                    Err(_) => (i, support_point, Vec::new(), None),
                }
            })
            .collect();

        let (mut nan, mut pos_inf, mut neg_inf, mut zero, mut valid) = (0, 0, 0, 0, 0);
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
                preds.iter().map(|x| x.outeq()).collect::<Vec<_>>()
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
}

/// A non-parametric fit in progress: the shared [`FitState`] plus the [`Runner`] that advances it.
pub(crate) struct Fit<E: Equation> {
    state: FitState<E>,
    runner: Box<dyn Runner<E>>,
}

impl<E: Equation> Fit<E> {
    /// Prepare a fit to run its first cycle. Removes any stale stop file left by an earlier run.
    pub(crate) fn new(state: FitState<E>, runner: Box<dyn Runner<E>>) -> Result<Self> {
        if Path::new(STOP_FILE).exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file(STOP_FILE).context("Unable to remove previous stop file")?;
        }
        Ok(Self { state, runner })
    }

    pub(crate) fn state(&self) -> &FitState<E> {
        &self.state
    }

    /// Run the next cycle, record it in the cycle log, and return the new [`Status`].
    pub(crate) fn step(&mut self) -> Result<Status> {
        self.state.cycle += 1;
        let span = tracing::info_span!("", "{}", format!("Cycle {}", self.state.cycle));
        let _enter = span.enter();

        self.state.status = self.runner.step(&mut self.state)?;
        self.state.record_cycle();
        Ok(self.state.status.clone())
    }

    /// Run cycles until the fit stops. Does nothing if it already has.
    pub(crate) fn run(&mut self) -> Result<()> {
        while self.state.status.is_continue() {
            self.step()?;
        }
        Ok(())
    }

    /// Mark the fit as [`Aborted`](StopReason::Aborted).
    pub(crate) fn request_stop(&mut self) {
        self.state.status = Status::Stop(StopReason::Aborted);
    }

    pub(crate) fn into_result(self) -> Result<NonParametricResult<E>> {
        self.state.into_result()
    }
}
