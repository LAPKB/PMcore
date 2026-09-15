//! Cycle stages shared by the non-parametric algorithms.
//!
//! NPAG and NPOD run the same pipeline: estimate the likelihood of every
//! support point for every subject, condense the support points, then solve for
//! the weights. Only the tolerances and a few extra checks differ, so the
//! implementation lives here and each runner passes its configuration in.

use anyhow::Result;

use pharmsol::prelude::data::{AssayErrorModels, Data};
use pharmsol::prelude::simulator::Equation;
use pharmsol::prelude::AssayErrorModel;

use crate::algorithms::diagnostics;
use crate::estimation::nonparametric::{calculate_psi, ipm::burke, qr, Psi, Theta, Weights};

/// Estimate the likelihood of every support point for every subject, then solve
/// for the weights with the interior point method.
///
/// Returns the likelihood matrix, the support point weights, and the
/// log-likelihood of the data under the weighted support points (higher is
/// better) — the quantity that [`FitState::n2ll`](crate::algorithms::FitState::n2ll)
/// negates and doubles.
pub(crate) fn estimate<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &AssayErrorModels,
    theta: &Theta,
    show_progress: bool,
) -> Result<(Psi, Weights, f64)> {
    let psi = calculate_psi(equation, data, theta, error_models, show_progress)?;
    diagnostics::check_zero_probability_subjects(equation, data, error_models, theta, &psi)?;
    let (lambda, log_likelihood) =
        burke(&psi).map_err(|err| anyhow::anyhow!("Error in IPM during estimation: {err:?}"))?;

    Ok((psi, lambda, log_likelihood))
}

/// How aggressively to drop support points during condensation.
pub(crate) struct CondensationOptions {
    /// Support points whose weight is below `max_weight × prune_threshold` are dropped.
    pub prune_threshold: f64,
    /// Minimum `|r_ii| / ‖r_i‖` ratio for a support point to survive the QR decomposition.
    pub qr_tolerance: f64,
    /// Whether to re-check for zero-probability subjects after condensing.
    pub check_zero_probability: bool,
}

/// Drop support points that carry negligible weight or that are numerically
/// redundant, then solve for the weights of the survivors.
///
/// `theta` and `psi` are filtered in place and stay aligned; the returned
/// weights and log-likelihood describe the surviving support points.
pub(crate) fn condense<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &AssayErrorModels,
    theta: &mut Theta,
    psi: &mut Psi,
    lambda: &Weights,
    options: &CondensationOptions,
) -> Result<(Weights, f64)> {
    // Drop the support points with lambda < max(lambda) * prune_threshold
    let max_lambda = lambda.iter().fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

    let mut keep = Vec::<usize>::new();
    for (index, lam) in lambda.iter().enumerate() {
        if lam > max_lambda * options.prune_threshold {
            keep.push(index);
        }
    }
    if psi.matrix().ncols() != keep.len() {
        tracing::debug!(
            "Weight pruning (max × {}) dropped {} support point(s)",
            options.prune_threshold,
            psi.matrix().ncols() - keep.len(),
        );
    }

    theta.filter_indices(keep.as_slice());
    psi.filter_column_indices(keep.as_slice());

    // Rank-revealing factorization
    let (r, perm) = qr::qrd(psi)?;

    let mut keep = Vec::<usize>::new();

    // The minimum between the number of subjects and the actual number of support points
    let keep_n = psi.matrix().ncols().min(psi.matrix().nrows());
    for i in 0..keep_n {
        let test = r.col(i).norm_l2();
        let ratio = r.get(i, i) / test;
        if ratio.abs() >= options.qr_tolerance {
            keep.push(*perm.get(i).unwrap());
        }
    }

    // If a support point is dropped, log it as a debug message
    if psi.matrix().ncols() != keep.len() {
        tracing::debug!(
            "QR decomposition dropped {} support point(s)",
            psi.matrix().ncols() - keep.len(),
        );
    }

    theta.filter_indices(keep.as_slice());
    psi.filter_column_indices(keep.as_slice());

    if options.check_zero_probability {
        diagnostics::check_zero_probability_subjects(equation, data, error_models, theta, psi)?;
    }

    let (lambda, log_likelihood) =
        burke(psi).map_err(|err| anyhow::anyhow!("Error in IPM during condensation: {err:?}"))?;

    Ok((lambda, log_likelihood))
}

/// Log the objective function, the number of support points and the current
/// error models, warning when the log-likelihood decreased since the last cycle.
pub(crate) fn log_objective(
    log_likelihood: f64,
    last_log_likelihood: f64,
    theta: &Theta,
    error_models: &AssayErrorModels,
) {
    tracing::info!("Objective function = {:.4}", -2.0 * log_likelihood);
    tracing::debug!("Support points: {}", theta.nspp());
    error_models.iter().for_each(|(outeq, em)| {
        if AssayErrorModel::None == *em {
            return;
        }
        tracing::debug!(
            "Error model for outeq {}: {:.4}",
            outeq,
            em.factor().unwrap_or_default()
        );
    });

    // A decreasing log-likelihood signals instability or model misspecification.
    if last_log_likelihood > log_likelihood + 1e-4 {
        tracing::warn!(
            "Log-likelihood decreased from {:.4} to {:.4} (objective function increased from {:.4} to {:.4})",
            last_log_likelihood,
            log_likelihood,
            -2.0 * last_log_likelihood,
            -2.0 * log_likelihood
        );
    }
}
