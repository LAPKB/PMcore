//! Stage 2: Dose Optimization
//!
//! Dual optimization approach matching Fortran BESTDOS113+:
//! 1. Optimize using posterior weights (patient-specific)
//! 2. Optimize using uniform weights (population-based)
//! 3. Compare costs and select the better result

use anyhow::Result;
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;

use crate::bestdose::cost::calculate_cost;
use crate::bestdose::predictions::calculate_final_predictions;
use crate::bestdose::types::{BestDoseProblem, BestDoseResult};
use crate::structs::weights::Weights;
use pharmsol::prelude::*;

/// Create initial simplex for Nelder-Mead optimization
///
/// Constructs a simplex with n+1 vertices in n-dimensional space,
/// where n is the number of doses to optimize.
fn create_initial_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let n = initial_point.len();
    let perturbation_percentage = 0.008; // 0.8% perturbation (matches Fortran/mod_old.rs)
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
/// Returns: (optimal_doses, final_cost)
fn run_single_optimization(
    problem: &BestDoseProblem,
    weights: &Weights,
    method_name: &str,
) -> Result<(Vec<f64>, f64)> {
    let min_dose = problem.doserange.min;
    let max_dose = problem.doserange.max;
    let target_subject = &problem.target;

    // Get number of doses to optimize
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

    tracing::info!(
        "  {} optimization: {} doses, {} support points",
        method_name,
        all_doses.len(),
        problem.theta.matrix().nrows()
    );

    // Create initial simplex
    let initial_guess = (min_dose + max_dose) / 2.0;
    let initial_point = vec![initial_guess; all_doses.len()];
    let initial_simplex = create_initial_simplex(&initial_point);

    // Create modified problem with the specified weights
    let mut problem_with_weights = problem.clone();
    problem_with_weights.posterior = weights.clone();

    // Run Nelder-Mead optimization
    let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(initial_simplex);
    let opt = Executor::new(problem_with_weights, solver)
        .configure(|state| state.max_iters(50))
        .run()?;

    let result = opt.state();
    let optimal_doses = result.best_param.clone().unwrap();
    let final_cost = result.best_cost;

    tracing::info!("    → Cost: {:.6}", final_cost);

    Ok((optimal_doses, final_cost))
}

/// Stage 2: Dual optimization (posterior vs uniform weights)
///
/// Matches Fortran BESTDOS113+ approach:
/// 1. Optimize with posterior weights from NPAGFULL11
/// 2. Optimize with uniform weights (1/N for all points)
/// 3. Compare costs and select the better result
///
/// This ensures we get good results whether the patient is "typical"
/// (uniform weights work well) or "atypical" (posterior weights work better).
pub fn dual_optimization(problem: &BestDoseProblem) -> Result<BestDoseResult> {
    let n_points = problem.theta.matrix().nrows();

    tracing::info!("=== STAGE 2: Dual Optimization ===");

    // OPTIMIZATION 1: Posterior weights (patient-specific)
    tracing::info!("Optimization 1: Posterior weights");
    let (doses1, cost1) = run_single_optimization(problem, &problem.posterior, "Posterior")?;

    // OPTIMIZATION 2: Uniform weights (population-based)
    tracing::info!("Optimization 2: Uniform weights");
    let uniform_weights = Weights::uniform(n_points);
    let (doses2, cost2) = run_single_optimization(problem, &uniform_weights, "Uniform")?;

    // Compare and select the better result
    tracing::info!("Comparison:");
    tracing::info!("  Posterior: cost = {:.6}", cost1);
    tracing::info!("  Uniform:   cost = {:.6}", cost2);

    let (final_doses, final_cost, method, final_weights) = if cost1 <= cost2 {
        tracing::info!("  → Selected: Posterior (lower cost)");
        (doses1, cost1, "posterior", problem.posterior.clone())
    } else {
        tracing::info!("  → Selected: Uniform (lower cost)");
        (doses2, cost2, "uniform", uniform_weights)
    };

    // Calculate final predictions with the winning weights
    tracing::info!("=== STAGE 3: Final Predictions ===");
    let (preds, auc_predictions) =
        calculate_final_predictions(problem, &final_doses, &final_weights)?;

    Ok(BestDoseResult {
        dose: final_doses,
        objf: final_cost,
        status: "Converged".to_string(),
        preds,
        auc_predictions,
        optimization_method: method.to_string(),
    })
}
