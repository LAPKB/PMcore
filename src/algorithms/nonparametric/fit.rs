//! The machinery every non-parametric algorithm runs on.
//!
//! A fit is split in two:
//!
//! - [`FitState`] describes the fit at a point in time: the problem (equation, data, error
//!   models, prior), the current [`Solution`], and the cycle bookkeeping. It is the same for
//!   every algorithm.
//! - A [`Runner`] holds only what is specific to one algorithm (its configuration and any
//!   adaptive state, such as NPAG's grid resolution) and knows how to run a single cycle.
//!
//! [`Fit`] pairs the two and owns everything around the cycle: the counter, the cycle cap, the
//! stop file, logging, and the cycle log. Both
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

use super::solution::Solution;
use crate::algorithms::{Status, StopReason};
use crate::estimation::nonparametric::{
    calculate_psi, qr, CycleLog, NPCycle, NonParametricResult, Psi, Theta,
};
use crate::estimation::{EstimationProblem, NonParametric};

/// A file with this name in the working directory stops a watched fit after the current cycle.
const STOP_FILE: &str = "stop";

/// What a runner concluded at the end of a cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CycleOutcome {
    /// The algorithm wants another cycle.
    Continue,
    /// The algorithm's own convergence criterion is met.
    Converged,
}

/// One non-parametric algorithm: what a single cycle does to a [`FitState`].
pub(crate) trait Runner<E: Equation>: Send + Sync {
    /// Run cycle [`state.cycle()`](FitState::cycle), replacing the solution (and, if it
    /// optimizes them, the error models) as it goes.
    ///
    /// Everything else (the cycle cap, the stop file, logging, and the cycle log) is handled
    /// by [`Fit`].
    fn step(&mut self, state: &mut FitState<E>) -> Result<CycleOutcome>;
}

/// The state of a non-parametric fit, shared by every algorithm.
///
/// Runners can read everything but only change the solution and the error models, and only
/// through [`set_solution`](Self::set_solution) and
/// [`adopt_error_models`](Self::adopt_error_models). The cycle bookkeeping belongs to [`Fit`].
#[derive(Debug)]
pub(crate) struct FitState<E: Equation> {
    equation: E,
    data: Data,
    error_models: AssayErrorModels,
    /// The support points the fit started from.
    prior: Theta,
    solution: Solution,
    /// Log-likelihood at the end of the previous cycle, or `None` during the first one.
    previous_log_likelihood: Option<f64>,
    /// The current cycle, starting at 1 (0 before the first cycle runs).
    cycle: usize,
    status: Status,
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
            solution: Solution::unsolved(prior.clone()),
            prior,
            previous_log_likelihood: None,
            cycle: 0,
            status: Status::Continue,
            cycle_log: CycleLog::new(),
        }
    }

    pub(crate) fn equation(&self) -> &E {
        &self.equation
    }

    pub(crate) fn data(&self) -> &Data {
        &self.data
    }

    pub(crate) fn error_models(&self) -> &AssayErrorModels {
        &self.error_models
    }

    pub(crate) fn solution(&self) -> &Solution {
        &self.solution
    }

    pub(crate) fn set_solution(&mut self, solution: Solution) {
        self.solution = solution;
    }

    /// Switch to new error models, together with the solution computed under them.
    pub(crate) fn adopt_error_models(
        &mut self,
        error_models: AssayErrorModels,
        solution: Solution,
    ) {
        self.error_models = error_models;
        self.solution = solution;
    }

    pub(crate) fn cycle(&self) -> usize {
        self.cycle
    }

    pub(crate) fn is_first_cycle(&self) -> bool {
        self.cycle == 1
    }

    pub(crate) fn status(&self) -> &Status {
        &self.status
    }

    /// The objective function, -2 × log-likelihood.
    pub(crate) fn objf(&self) -> f64 {
        -2.0 * self.solution.log_likelihood()
    }

    /// Absolute change in log-likelihood since the previous cycle (`None` in the first cycle).
    pub(crate) fn objective_change(&self) -> Option<f64> {
        self.previous_log_likelihood
            .map(|previous| (previous - self.solution.log_likelihood()).abs())
    }

    /// Compute [`Psi`] for `theta` under the current error models.
    pub(crate) fn compute_psi(&self, theta: &Theta, progress: bool) -> Result<Psi> {
        calculate_psi(
            &self.equation,
            &self.data,
            theta,
            &self.error_models,
            progress,
        )
    }

    /// Evaluate `theta` and solve for its optimal weights.
    pub(crate) fn estimate(&self, theta: Theta, progress: bool) -> Result<Solution> {
        let psi = self.compute_psi(&theta, progress)?;
        self.check_zero_probability_subjects(&theta, &psi)?;
        Solution::optimal(theta, psi).context("IPM failed during estimation")
    }

    /// Drop support points that carry negligible weight or are linearly dependent, then solve
    /// for the weights of the survivors.
    ///
    /// A point is dropped when its weight in `estimate` is at most `prune_threshold` × the
    /// largest weight, or when its column of [`Psi`] is (numerically) a combination of the
    /// others, as judged by a rank-revealing QR decomposition with tolerance `qr_tolerance`.
    pub(crate) fn condense(
        &self,
        estimate: Solution,
        prune_threshold: f64,
        qr_tolerance: f64,
    ) -> Result<Solution> {
        let (mut theta, mut psi, lambda, _) = estimate.into_parts();

        let max_lambda = lambda.iter().fold(f64::NEG_INFINITY, f64::max);
        let keep: Vec<usize> = lambda
            .iter()
            .enumerate()
            .filter(|&(_, lam)| lam > max_lambda * prune_threshold)
            .map(|(i, _)| i)
            .collect();
        retain_support_points(&mut theta, &mut psi, &keep, "Lambda (max/1000)");

        let (r, perm) = qr::qrd(&psi)?;
        let rank_bound = psi.matrix().ncols().min(psi.matrix().nrows());
        let keep: Vec<usize> = (0..rank_bound)
            .filter(|&i| (r.get(i, i) / r.col(i).norm_l2()).abs() >= qr_tolerance)
            .map(|i| perm[i])
            .collect();
        retain_support_points(&mut theta, &mut psi, &keep, "QR decomposition");

        self.check_zero_probability_subjects(&theta, &psi)?;
        Solution::optimal(theta, psi).context("IPM failed during condensation")
    }

    /// Log the objective function and error models at the end of a cycle, and warn if the
    /// objective got worse.
    fn log_cycle_summary(&self) {
        tracing::info!("Objective function = {:.4}", self.objf());
        tracing::debug!("Support points: {}", self.solution.theta().nspp());

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
            if previous > self.solution.log_likelihood() + 1e-4 {
                tracing::warn!(
                    "Objective function decreased from {:.4} to {:.4} (delta = {})",
                    -2.0 * previous,
                    self.objf(),
                    -2.0 * previous - self.objf()
                );
            }
        }
    }

    /// Append the current cycle to the cycle log.
    fn record_cycle(&mut self) {
        let delta = self.objective_change().unwrap_or(f64::INFINITY);
        let theta = self.solution.theta();
        self.cycle_log.push(NPCycle::new(
            self.cycle,
            self.objf(),
            self.error_models.clone(),
            theta.clone(),
            self.solution.weights().clone(),
            theta.nspp(),
            delta,
            self.status.clone(),
        ));
        self.previous_log_likelihood = Some(self.solution.log_likelihood());
    }

    fn into_result(self) -> Result<NonParametricResult<E>> {
        let objf = self.objf();
        let (theta, psi, weights, _) = self.solution.into_parts();
        NonParametricResult::new(
            self.equation,
            self.data,
            self.error_models,
            self.prior,
            theta,
            psi,
            weights,
            objf,
            self.cycle,
            self.status,
            self.cycle_log,
        )
    }

    /// Fail if any subject has zero (or non-finite) probability at the support points `theta`.
    ///
    /// Each row of `psi` holds a subject's likelihood across every support point, so a
    /// subject's probability is the sum of its row. When any subject is flagged, per-subject
    /// diagnostics are logged and an error naming the subjects is returned.
    fn check_zero_probability_subjects(&self, theta: &Theta, psi: &Psi) -> Result<()> {
        let psi = psi.matrix();

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
            self.log_zero_probability_subject(subjects[i], theta);
        }

        let ids: Vec<&String> = flagged.iter().map(|&i| subjects[i].id()).collect();
        anyhow::bail!(
            "The probability of {}/{} subjects is zero given the model. Affected subjects: {:?}",
            flagged.len(),
            psi.nrows(),
            ids
        )
    }

    /// Log likelihood diagnostics for a subject with zero probability at the support points
    /// `theta`.
    fn log_zero_probability_subject(&self, subject: &Subject, theta: &Theta) {
        tracing::debug!("Subject with zero probability: {}", subject.id());

        // Simulate every support point for this subject. Points that fail to simulate count
        // as having no likelihood.
        let mut results: Vec<_> = (0..theta.nspp())
            .into_par_iter()
            .map(|i| {
                let support_point: Vec<f64> = theta.matrix().row(i).iter().copied().collect();
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

/// Keep only the support points at `keep`, in `theta` and `psi` alike.
fn retain_support_points(theta: &mut Theta, psi: &mut Psi, keep: &[usize], step: &str) {
    let dropped = theta.nspp() - keep.len();
    if dropped > 0 {
        tracing::debug!("{} dropped {} support point(s)", step, dropped);
    }
    theta.filter_indices(keep);
    psi.filter_column_indices(keep);
}

/// A non-parametric fit in progress: the shared [`FitState`] plus the [`Runner`] that advances it.
pub(crate) struct Fit<E: Equation> {
    state: FitState<E>,
    runner: Box<dyn Runner<E>>,
    max_cycles: usize,
    watch_stop_file: bool,
}

impl<E: Equation> Fit<E> {
    /// Prepare a fit that stops on convergence or after `max_cycles` cycles.
    pub(crate) fn new(state: FitState<E>, runner: Box<dyn Runner<E>>, max_cycles: usize) -> Self {
        Self {
            state,
            runner,
            max_cycles,
            watch_stop_file: false,
        }
    }

    /// Also stop after any cycle at which a [`STOP_FILE`] exists in the working directory.
    ///
    /// A stop file left over from an earlier run is removed now, so it can't end this one
    /// before it starts.
    pub(crate) fn watch_stop_file(mut self) -> Result<Self> {
        if Path::new(STOP_FILE).exists() {
            tracing::info!("Removing existing stop file prior to run");
            fs::remove_file(STOP_FILE).context("Unable to remove previous stop file")?;
        }
        self.watch_stop_file = true;
        Ok(self)
    }

    pub(crate) fn state(&self) -> &FitState<E> {
        &self.state
    }

    /// Run the next cycle, record it in the cycle log, and return the new [`Status`].
    pub(crate) fn step(&mut self) -> Result<Status> {
        self.state.cycle += 1;
        let span = tracing::info_span!("", "{}", format!("Cycle {}", self.state.cycle));
        let _enter = span.enter();

        let outcome = self.runner.step(&mut self.state)?;
        self.state.log_cycle_summary();

        self.state.status = match outcome {
            CycleOutcome::Converged => {
                tracing::info!("The model converged after {} cycles", self.state.cycle);
                Status::Stop(StopReason::Converged)
            }
            CycleOutcome::Continue => self.limit_reached().map_or(Status::Continue, Status::Stop),
        };
        self.state.record_cycle();
        Ok(self.state.status.clone())
    }

    /// Why the fit must stop although it has not converged, if it must.
    fn limit_reached(&self) -> Option<StopReason> {
        if self.state.cycle >= self.max_cycles {
            tracing::warn!("Maximum number of cycles reached");
            return Some(StopReason::MaxCycles);
        }
        if self.watch_stop_file && Path::new(STOP_FILE).exists() {
            tracing::warn!("Stopfile detected - breaking");
            return Some(StopReason::StopFile);
        }
        None
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
