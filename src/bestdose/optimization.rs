//! Stage 2: Dose Optimization
//!
//! Implements the dual optimization strategy that compares patient-specific and
//! population-based approaches to find the best dosing regimen.
//!
//! # Dual Optimization Strategy
//!
//! The algorithm runs two independent optimizations:
//!
//! ## Optimization 1: Posterior Weights (Patient-Specific)
//!
//! - Uses refined posterior weights from NPAGFULL11 + NPAGFULL
//! - Emphasizes parameter values compatible with patient history
//! - Best when patient has substantial historical data
//! - Variance term dominates cost function
//!
//! ## Optimization 2: Uniform Weights (Population-Based)
//!
//! - Treats all posterior support points equally (weight = 1/M)
//! - Emphasizes population-typical behavior
//! - More robust when patient history is limited
//! - Population mean (from prior) influences cost
//!
//! ## Selection
//!
//! The algorithm compares both results and selects the one with lower cost.
//! This automatic selection provides robustness across diverse patient scenarios.
//!
//! # Optimization Method
//!
//! Uses the Nelder-Mead simplex algorithm (derivative-free):
//! - **Initial simplex**: -20% perturbation from starting doses
//! - **Max iterations**: 1000
//! - **Convergence tolerance**: 1e-10 (standard deviation of simplex)
//!
//! # See Also
//!
//! - [`dual_optimization`]: Main entry point for Stage 2
//! - [`create_initial_simplex`]: Simplex construction
//! - [`crate::bestdose::cost::calculate_cost`]: Cost function implementation

use anyhow::Result;
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;

use crate::bestdose::cost::calculate_cost;
use crate::bestdose::predictions::calculate_final_predictions;
use crate::bestdose::types::{BestDoseProblem, BestDoseResult, BestDoseStatus, OptimalMethod};
use crate::structs::weights::Weights;
use pharmsol::prelude::*;

/// Create initial simplex for Nelder-Mead optimization
///
/// Constructs a simplex with n+1 vertices in n-dimensional space,
/// where n is the number of doses to optimize.
fn create_initial_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let n = initial_point.len();
    let perturbation_percentage = -0.2; // -20% perturbation
    let mut simplex = Vec::with_capacity(n + 1);

    // First vertex is the initial point
    simplex.push(initial_point.to_vec());

    // Create n additional vertices by perturbing each dimension
    for i in 0..n {
        let mut vertex = initial_point.to_vec();
        let perturbation = if initial_point[i] == 0.0 {
            0.00025 // Special case for zero values
        } else {
            perturbation_percentage * initial_point[i]
        };
        vertex[i] += perturbation;
        simplex.push(vertex);
    }

    simplex
}

/// Implement CostFunction trait for BestDoseProblem
///
/// This allows the Nelder-Mead optimizer to evaluate candidate doses.
impl CostFunction for BestDoseProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        calculate_cost(self, param)
    }
}

/// Run single optimization with specified weights
///
/// This is a helper for the dual optimization approach.
///
/// When `problem.time_offset` is set (past/future separation mode):
/// - Only optimizes doses where `dose_optimization_mask[i] == true`
/// - Creates a reduced-dimension simplex for future doses only
/// - Maps optimized doses back to full vector (past doses unchanged)
///
/// Returns: (optimal_doses, final_cost)
fn run_single_optimization(
    problem: &BestDoseProblem,
    weights: &Weights,
    method_name: &str,
) -> Result<(Vec<f64>, f64)> {
    let min_dose = problem.doserange.min;
    let max_dose = problem.doserange.max;
    let target_subject = &problem.target;

    // Get all doses from target subject
    let all_doses: Vec<f64> = target_subject
        .iter()
        .flat_map(|occ| {
            occ.iter().filter_map(|event| match event {
                Event::Bolus(bolus) => Some(bolus.amount()),
                Event::Infusion(infusion) => Some(infusion.amount()),
                Event::Observation(_) => None,
            })
        })
        .collect();

    // Count optimizable doses (amount == 0)
    let num_optimizable = all_doses.iter().filter(|&&d| d == 0.0).count();
    let num_fixed = all_doses.len() - num_optimizable;
    let num_support_points = problem.theta.matrix().nrows();

    tracing::info!(
        "  │  {} doses: {} optimizable, {} fixed | {} support points",
        method_name,
        num_optimizable,
        num_fixed,
        num_support_points
    );

    // If no doses to optimize, return current doses with zero cost
    if num_optimizable == 0 {
        tracing::warn!("  │  ⚠ No doses to optimize (all fixed)");
        return Ok((all_doses, 0.0));
    }

    // Create initial simplex for optimizable doses only
    let initial_guess = (min_dose + max_dose) / 2.0;
    let initial_point = vec![initial_guess; num_optimizable];
    let initial_simplex = create_initial_simplex(&initial_point);

    // Create modified problem with the specified weights
    let mut problem_with_weights = problem.clone();
    problem_with_weights.posterior = weights.clone();

    // Run Nelder-Mead optimization
    let solver: NelderMead<Vec<f64>, f64> =
        NelderMead::new(initial_simplex).with_sd_tolerance(1e-10)?;

    let opt = Executor::new(problem_with_weights, solver)
        .configure(|state| state.max_iters(1000))
        .run()?;

    let result = opt.state();
    let optimized_doses = result.best_param.clone().unwrap();
    let final_cost = result.best_cost;

    tracing::info!("  │  → Cost: {:.6}", final_cost);

    // Map optimized doses back to full vector
    // For past/future mode: combine fixed past doses + optimized future doses
    let mut full_doses = Vec::with_capacity(all_doses.len());
    let mut opt_idx = 0;

    for &original_dose in all_doses.iter() {
        if original_dose == 0.0 {
            // This was a placeholder dose - use optimized value
            full_doses.push(optimized_doses[opt_idx]);
            opt_idx += 1;
        } else {
            // This was a fixed dose - keep original value
            full_doses.push(original_dose);
        }
    }

    Ok((full_doses, final_cost))
}

/// Stage 2 & 3: Dual optimization + Final predictions
///
/// # Algorithm Flow (Matches Diagram)
///
/// ```text
/// ┌─────────────────────────────────────────────────┐
/// │ STAGE 2: Dual Optimization                      │
/// │                                                 │
/// │  OPTIMIZATION 1: Posterior Weights              │
/// │    Use NPAGFULL11 posterior probabilities       │
/// │    → (doses₁, cost₁)                            │
/// │                                                 │
/// │  OPTIMIZATION 2: Uniform Weights                │
/// │    Use equal weights (1/M) for all points       │
/// │    → (doses₂, cost₂)                            │
/// │                                                 │
/// │  SELECTION: Choose min(cost₁, cost₂)            │
/// │    → (optimal_doses, optimal_cost, method)      │
/// └────────────┬────────────────────────────────────┘
///              ↓
/// ┌─────────────────────────────────────────────────┐
/// │ STAGE 3: Final Predictions                      │
/// │                                                 │
/// │  Calculate predictions with:                    │
/// │    - Optimal doses from winning optimization    │
/// │    - Winning weights (posterior or uniform)     │
/// │                                                 │
/// │  Return: BestDoseResult                         │
/// └─────────────────────────────────────────────────┘
/// ```
///
/// This dual optimization ensures robust performance:
/// - Posterior weights: Best for atypical patients with good data
/// - Uniform weights: Best for typical patients or limited data
/// - Automatic selection gives optimal result in both cases
pub fn dual_optimization(problem: &BestDoseProblem) -> Result<BestDoseResult> {
    let n_points = problem.theta.matrix().nrows();

    // ═════════════════════════════════════════════════════════════
    // STAGE 2: Dual Optimization
    // ═════════════════════════════════════════════════════════════
    tracing::info!("─────────────────────────────────────────────────────────────");
    tracing::info!("STAGE 2: Dual Optimization");
    tracing::info!("─────────────────────────────────────────────────────────────");

    // OPTIMIZATION 1: Posterior weights (patient-specific adaptation)
    tracing::info!("│");
    tracing::info!("├─ Optimization 1: Posterior Weights (Patient-Specific)");
    let (doses1, cost1) = run_single_optimization(problem, &problem.posterior, "Posterior")?;

    // OPTIMIZATION 2: Uniform weights (population robustness)
    tracing::info!("│");
    tracing::info!("├─ Optimization 2: Uniform Weights (Population-Based)");
    let uniform_weights = Weights::uniform(n_points);
    let (doses2, cost2) = run_single_optimization(problem, &uniform_weights, "Uniform")?;

    // SELECTION: Compare and choose the better result
    tracing::info!("│");
    tracing::info!("└─ Selection: Compare Results");
    tracing::info!("     Posterior cost: {:.6}", cost1);
    tracing::info!("     Uniform cost:   {:.6}", cost2);

    let (final_doses, final_cost, method, final_weights) = if cost1 <= cost2 {
        tracing::info!("     → Winner: Posterior (lower cost) ✓");
        (
            doses1,
            cost1,
            OptimalMethod::Posterior,
            problem.posterior.clone(),
        )
    } else {
        tracing::info!("     → Winner: Uniform (lower cost) ✓");
        (doses2, cost2, OptimalMethod::Uniform, uniform_weights)
    };

    // ═════════════════════════════════════════════════════════════
    // STAGE 3: Final Predictions
    // ═════════════════════════════════════════════════════════════
    tracing::info!("─────────────────────────────────────────────────────────────");
    tracing::info!("STAGE 3: Final Predictions");
    tracing::info!("─────────────────────────────────────────────────────────────");
    tracing::info!(
        "  Calculating predictions with optimal doses and {} weights",
        method
    );

    // Generate target subject with optimal doses
    let mut optimal_subject = problem.target.clone();
    let mut dose_number = 0;

    for occasion in optimal_subject.iter_mut() {
        for event in occasion.iter_mut() {
            match event {
                Event::Bolus(bolus) => {
                    bolus.set_amount(final_doses[dose_number]);
                    dose_number += 1;
                }
                Event::Infusion(infusion) => {
                    infusion.set_amount(final_doses[dose_number]);
                    dose_number += 1;
                }
                Event::Observation(_) => {}
            }
        }
    }

    let (preds, auc_predictions) =
        calculate_final_predictions(problem, &final_doses, &final_weights)?;

    tracing::info!("  ✓ Predictions complete");
    tracing::info!("─────────────────────────────────────────────────────────────");

    Ok(BestDoseResult {
        optimal_subject,
        objf: final_cost,
        status: BestDoseStatus::Converged,
        preds,
        auc_predictions,
        optimization_method: method,
    })
}
