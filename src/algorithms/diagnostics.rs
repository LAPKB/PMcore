//! Diagnostics shared by the non-parametric algorithms.
//!
//! These are free functions rather than methods on the algorithm state, so they
//! can be applied to any combination of equation, data and support points, and
//! so that logging helpers stay out of the algorithm trait.

use anyhow::Result;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};

use pharmsol::prelude::data::{AssayErrorModels, Data};
use pharmsol::prelude::simulator::Equation;
use pharmsol::{Predictions, Subject};

use crate::estimation::nonparametric::{Psi, Theta};

/// Identify subjects whose total probability given the model is zero or
/// non-finite.
///
/// Each row of [`Psi`] holds the likelihood of a subject across every
/// support point, so a subject's probability is the sum across its row. A
/// subject is flagged when that sum is zero or not finite, meaning the model
/// cannot explain the subject's data. When any subject is flagged, detailed
/// per-subject diagnostics are logged and an error is returned.
pub fn check_zero_probability_subjects<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &AssayErrorModels,
    theta: &Theta,
    psi: &Psi,
) -> Result<()> {
    let matrix = psi.matrix();

    // Report non-finite entries; these propagate into the row sums below.
    let nonfinite = matrix
        .row_iter()
        .flat_map(|row| row.iter().copied())
        .filter(|v| !v.is_finite())
        .count();
    if nonfinite > 0 {
        tracing::warn!(
            "Psi matrix contains {} non-finite value(s) of {} total",
            nonfinite,
            matrix.nrows() * matrix.ncols()
        );
    }

    // A subject's probability is the sum across its row.
    let subjects = data.subjects();
    let flagged: Vec<usize> = (0..matrix.nrows())
        .filter(|&i| {
            let probability: f64 = (0..matrix.ncols()).map(|j| matrix[(i, j)]).sum();
            !probability.is_finite() || probability == 0.0
        })
        .collect();

    if flagged.is_empty() {
        return Ok(());
    }

    tracing::error!(
        "{}/{} subjects have zero probability given the model",
        flagged.len(),
        matrix.nrows()
    );

    for &i in &flagged {
        log_zero_probability_subject(equation, error_models, theta, subjects[i]);
    }

    let ids: Vec<&String> = flagged.iter().map(|&i| subjects[i].id()).collect();
    Err(anyhow::anyhow!(
        "The probability of {}/{} subjects is zero given the model. Affected subjects: {:?}",
        flagged.len(),
        matrix.nrows(),
        ids
    ))
}

/// Log detailed likelihood diagnostics for a single subject whose
/// probability given the model is zero or non-finite.
///
/// This is best-effort: support points that fail to simulate are reported at
/// debug level and skipped, since the caller is already handling an error.
fn log_zero_probability_subject<E: Equation>(
    equation: &E,
    error_models: &AssayErrorModels,
    theta: &Theta,
    subject: &Subject,
) {
    tracing::debug!("Subject with zero probability: {}", subject.id());

    // Simulate every support point for this subject in parallel.
    let mut results: Vec<_> = theta
        .matrix()
        .row_iter()
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .filter_map(|(i, spp)| {
            let support_point: Vec<f64> = spp.iter().copied().collect();
            match equation.simulate_subject_dense(subject, &support_point, Some(error_models)) {
                Ok((pred, ll)) => Some((i, support_point, pred.get_predictions(), ll)),
                Err(err) => {
                    tracing::debug!("Support point #{} could not be simulated: {}", i, err);
                    None
                }
            }
        })
        .collect();

    if results.is_empty() {
        tracing::debug!("\tNo support point could be simulated for this subject");
        return;
    }

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
