//! Stage 1: Posterior Density Calculation
//!
//! Two-step Bayesian posterior refinement process that transforms a population prior
//! into a patient-specific posterior distribution.
//!
//! # Overview
//!
//! The posterior calculation uses a two-step approach:
//!
//! ## Step 1: NPAGFULL11 - Bayesian Filtering
//!
//! Filters the population prior to identify parameter regions compatible with patient data:
//!
//! 1. Calculate likelihood P(data|θᵢ) for each prior support point
//! 2. Apply Bayes' rule: P(θᵢ|data) ∝ P(data|θᵢ) × P(θᵢ)
//! 3. Filter: Keep points where P(θᵢ|data) > 1e-100 × max(P(θᵢ|data))
//! 4. Renormalize weights
//!
//! **Output**: Filtered posterior with typically 5-50 support points
//!
//! ## Step 2: NPAGFULL - Local Refinement
//!
//! Refines each filtered point through full NPAG optimization:
//!
//! 1. For each filtered support point: Run NPAG optimization starting from that point
//! 2. Find refined "daughter" point in local parameter space
//! 3. Preserve NPAGFULL11 weights (no recalculation)
//!
//! **Output**: Refined posterior with improved parameter estimates
//!
//! # Key Differences from Standard NPAG
//!
//! - **NPAGFULL11**: Uses only lambda filtering (no QR decomposition)
//! - **NPAGFULL**: Refines individual points (not population estimation)
//! - **Weight preservation**: NPAGFULL11 probabilities are kept, not recalculated
//!
//! # Configuration
//!
//! The `KEEP_UNREFINED_POINTS` constant controls behavior when refinement fails:
//! - `true`: Keep original filtered point (maintains point count)
//! - `false`: Skip point entirely (may reduce posterior size)
//!
//! # Functions
//!
//! - [`npagfull11_filter`]: Step 1 - Bayesian filtering
//! - [`npagfull_refinement`]: Step 2 - Local optimization
//! - [`calculate_two_step_posterior`]: Complete two-step process
//!
//! # See Also
//!
//! - [`crate::algorithms::npag`]: Standard NPAG algorithm for comparison

use anyhow::Result;
use faer::Mat;

use crate::algorithms::npag::burke;
use crate::algorithms::npag::NPAG;
use crate::algorithms::Algorithms;
use crate::prelude::*;
use crate::structs::psi::calculate_psi;
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;
use pharmsol::prelude::*;

// =============================================================================
// CONFIGURATION: Control refinement behavior
// =============================================================================
/// Control whether to keep or skip points when refinement fails
///
/// **Keep unrefined points (true)**: All filtered points are kept in posterior.
///   - If refinement succeeds → use refined point
///   - If refinement fails → use original filtered point
///   - Result: Same number of points as NPAGFULL11 filtering produced
///
/// **Skip failed refinements (false)**: Points are skipped when refinement fails.
///   - If refinement succeeds → use refined point
///   - If refinement fails → skip point entirely
///   - Result: Fewer points than NPAGFULL11 filtering
const KEEP_UNREFINED_POINTS: bool = true;

/// Step 1.1: NPAGFULL11 - Bayesian filtering to get compatible points
///
/// Implements Bayesian filtering by:
/// 1. Calculate P(data|θᵢ) for each prior support point
/// 2. Apply Bayes' rule to get P(θᵢ|data)
/// 3. Filter: Keep points where P(θᵢ|data) > 1e-100 × max_weight
///
/// Note: This uses only lambda filtering, NO QR decomposition or second burke call.
///
/// Returns: (filtered_theta, filtered_posterior_weights, filtered_population_weights)
pub fn npagfull11_filter(
    population_theta: &Theta,
    population_weights: &Weights,
    past_data: &Data,
    eq: &ODE,
    error_models: &ErrorModels,
) -> Result<(Theta, Weights, Weights)> {
    tracing::info!("Stage 1.1: NPAGFULL11 Bayesian filtering");

    // Calculate psi matrix P(data|theta_i) for all support points
    let psi = calculate_psi(eq, past_data, population_theta, error_models, false, true)?;

    // First burke call to get initial posterior probabilities
    let (initial_weights, _) = burke(&psi)?;

    // NPAGFULL11 filtering: Keep all points within 1e-100 of the maximum weight
    // This is different from NPAG's condensation - NO QR decomposition here!
    let max_weight = initial_weights
        .iter()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let threshold = 1e-100; // NPAGFULL11-specific threshold

    let keep_lambda: Vec<usize> = initial_weights
        .iter()
        .enumerate()
        .filter(|(_, lam)| *lam > threshold * max_weight)
        .map(|(i, _)| i)
        .collect();

    // Filter theta to keep only points above threshold
    let mut filtered_theta = population_theta.clone();
    filtered_theta.filter_indices(&keep_lambda);

    // Filter and renormalize posterior weights
    let filtered_weights: Vec<f64> = keep_lambda.iter().map(|&i| initial_weights[i]).collect();
    let sum: f64 = filtered_weights.iter().sum();
    let final_posterior_weights =
        Weights::from_vec(filtered_weights.iter().map(|w| w / sum).collect());

    // Also filter the prior weights to match the filtered theta
    let filtered_population_weights: Vec<f64> =
        keep_lambda.iter().map(|&i| population_weights[i]).collect();
    let prior_sum: f64 = filtered_population_weights.iter().sum();
    let final_population_weights = Weights::from_vec(
        filtered_population_weights
            .iter()
            .map(|w| w / prior_sum)
            .collect(),
    );

    tracing::info!(
        "  {} → {} support points (lambda filter, threshold={:.0e})",
        population_theta.matrix().nrows(),
        filtered_theta.matrix().nrows(),
        threshold * max_weight
    );

    Ok((
        filtered_theta,
        final_posterior_weights,
        final_population_weights,
    ))
}

/// Step 1.2: NPAGFULL - Refine each filtered point with full NPAG optimization
///
/// For each filtered support point from NPAGFULL11, run a full NPAG optimization
/// starting from that point to get a refined "daughter" point.
///
/// Behavior controlled by KEEP_UNREFINED_POINTS configuration:
/// - If refinement succeeds → use refined point
/// - If refinement fails → keep original filtered point (when enabled)
///
/// The NPAGFULL11 probabilities are preserved (not recalculated from NPAG).
///
/// Parameters:
/// - max_cycles: Maximum NPAG cycles for refinement (0=skip refinement)
pub fn npagfull_refinement(
    filtered_theta: &Theta,
    filtered_weights: &Weights,
    past_data: &Data,
    eq: &ODE,
    settings: &Settings,
    max_cycles: usize,
) -> Result<(Theta, Weights)> {
    if max_cycles == 0 {
        tracing::info!("Stage 1.2: NPAGFULL refinement skipped (max_cycles=0)");
        return Ok((filtered_theta.clone(), filtered_weights.clone()));
    }

    tracing::info!("Stage 1.2: NPAGFULL refinement (max_cycles={})", max_cycles);

    let mut refined_points = Vec::new();
    let mut kept_weights: Vec<f64> = Vec::new();
    let num_points = filtered_theta.matrix().nrows();

    for i in 0..num_points {
        tracing::debug!("  Refining point {}/{}", i + 1, num_points);

        // Get the current filtered point as starting point
        let point: Vec<f64> = filtered_theta.matrix().row(i).iter().copied().collect();

        // Create a single-point theta for NPAG initialization
        let n_params = point.len();
        let single_point_matrix = Mat::from_fn(1, n_params, |_r, c| point[c]);
        let single_point_theta =
            Theta::from_parts(single_point_matrix, settings.parameters().clone()).unwrap();

        // Configure NPAG for refinement
        let mut npag_settings = settings.clone();
        npag_settings.disable_output(); // Don't write files for each refinement
        npag_settings.set_prior(crate::routines::initialization::Prior::Theta(
            single_point_theta.clone(),
        ));

        // Create and run NPAG
        let mut npag = NPAG::new(npag_settings, eq.clone(), past_data.clone())?;
        npag.set_theta(single_point_theta);

        // Run NPAG optimization
        let refinement_result = npag.initialize().and_then(|_| {
            while !npag.next_cycle()? {}
            Ok(())
        });

        // Handle refinement failure based on configuration
        if let Err(e) = refinement_result {
            if KEEP_UNREFINED_POINTS {
                // Keep the original filtered point
                tracing::warn!(
                    "  Failed to refine point {}/{}: {} - using original point",
                    i + 1,
                    num_points,
                    e
                );
                refined_points.push(point);
                kept_weights.push(filtered_weights[i]);
            } else {
                // Skip this point entirely
                tracing::warn!(
                    "  Failed to refine point {}/{}: {} - skipping",
                    i + 1,
                    num_points,
                    e
                );
            }
            continue;
        }

        // Extract refined point (use first if multiple)
        let refined_theta = npag.theta();

        // Check if refinement produced any points
        if refined_theta.matrix().nrows() == 0 {
            if KEEP_UNREFINED_POINTS {
                // Keep the original filtered point
                tracing::warn!(
                    "  NPAG refinement produced no points for point {}/{} - using original point",
                    i + 1,
                    num_points
                );
                refined_points.push(point);
                kept_weights.push(filtered_weights[i]);
            } else {
                // Skip this point entirely
                tracing::warn!(
                    "  NPAG refinement produced no points for point {}/{} - skipping",
                    i + 1,
                    num_points
                );
            }
            continue;
        }

        // Refinement succeeded - use the refined point
        let refined_point: Vec<f64> = refined_theta.matrix().row(0).iter().copied().collect();

        refined_points.push(refined_point);
        kept_weights.push(filtered_weights[i]);
    }

    // Build refined theta matrix
    let n_params = settings.parameters().len();
    let n_points = refined_points.len();
    let refined_matrix = Mat::from_fn(n_points, n_params, |r, c| refined_points[r][c]);
    let refined_theta = Theta::from_parts(refined_matrix, settings.parameters().clone()).unwrap();

    // Renormalize weights
    let weight_sum: f64 = kept_weights.iter().sum();
    let normalized_weights = if weight_sum > 0.0 {
        Weights::from_vec(kept_weights.iter().map(|w| w / weight_sum).collect())
    } else {
        Weights::uniform(n_points)
    };

    tracing::info!(
        "  {} → {} refined points",
        filtered_theta.matrix().nrows(),
        refined_theta.matrix().nrows()
    );

    Ok((refined_theta, normalized_weights))
}

/// Calculate two-step posterior (NPAGFULL11 + NPAGFULL)
///
/// This is the complete Stage 1 of the BestDose algorithm.
///
/// Returns (posterior_theta, posterior_weights, filtered_population_weights) suitable for dose optimization.
/// The filtered_population_weights are the original prior weights filtered to match the posterior support points.
pub fn calculate_two_step_posterior(
    population_theta: &Theta,
    population_weights: &Weights,
    past_data: &Data,
    eq: &ODE,
    error_models: &ErrorModels,
    settings: &Settings,
    max_cycles: usize,
) -> Result<(Theta, Weights, Weights)> {
    tracing::info!("=== STAGE 1: Posterior Density Calculation ===");

    // Step 1.1: NPAGFULL11 filtering (returns filtered posterior AND filtered prior)
    let (filtered_theta, filtered_posterior_weights, filtered_population_weights) =
        npagfull11_filter(
            population_theta,
            population_weights,
            past_data,
            eq,
            error_models,
        )?;

    // Step 1.2: NPAGFULL refinement
    let (refined_theta, refined_weights) = npagfull_refinement(
        &filtered_theta,
        &filtered_posterior_weights,
        past_data,
        eq,
        settings,
        max_cycles,
    )?;

    tracing::info!(
        "  Final posterior: {} points",
        refined_theta.matrix().nrows()
    );

    Ok((refined_theta, refined_weights, filtered_population_weights))
}
