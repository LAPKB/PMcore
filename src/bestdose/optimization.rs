//! Dose optimization
//!
//! Minimizes the hybrid cost function over the optimizable doses of a target
//! subject using the Nelder-Mead simplex algorithm, then returns the optimal
//! dosing subject and its cost.

use anyhow::Result;
use argmin::core::{CostFunction, Executor};
use argmin::solver::neldermead::NelderMead;

use crate::bestdose::cost::{calculate_cost, evaluate};
use crate::bestdose::types::{BestDoseObjective, BestDoseResult};
use pharmsol::prelude::*;

/// Create initial simplex for Nelder-Mead optimization.
///
/// Constructs a simplex with `n + 1` vertices in `n`-dimensional space, where
/// `n` is the number of doses to optimize.
fn create_initial_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let n = initial_point.len();
    let perturbation_percentage = -0.2; // -20% perturbation
    let mut simplex = Vec::with_capacity(n + 1);

    simplex.push(initial_point.to_vec());

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

/// Let the Nelder-Mead optimizer evaluate candidate doses via the cost function.
impl<E: Equation> CostFunction for BestDoseObjective<E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        calculate_cost(self, param)
    }
}

/// Solve for the optimal doses of `objective.target` and return the optimal
/// dosing subject together with the final cost.
pub(crate) fn optimize<E: Equation>(objective: &BestDoseObjective<E>) -> Result<BestDoseResult> {
    let min_dose = objective.doserange.min;
    let max_dose = objective.doserange.max;

    // All dose amounts in the target subject, in order.
    let all_doses: Vec<f64> = objective
        .target
        .iter()
        .flat_map(|occ| {
            occ.iter().filter_map(|event| match event {
                Event::Bolus(bolus) => Some(bolus.amount()),
                Event::Infusion(infusion) => Some(infusion.amount()),
                Event::Observation(_) => None,
            })
        })
        .collect();

    // Optimizable doses are those with amount == 0.0.
    let num_optimizable = all_doses.iter().filter(|&&d| d == 0.0).count();

    tracing::debug!(
        "BestDose optimization: {} optimizable doses, {} fixed, {} support points",
        num_optimizable,
        all_doses.len() - num_optimizable,
        objective.theta.matrix().nrows()
    );

    // Solve for the optimizable doses (those with amount == 0.0).
    let optimizable_doses: Vec<f64> = if num_optimizable == 0 {
        tracing::warn!("No doses to optimize (all fixed)");
        Vec::new()
    } else {
        let initial_guess = (min_dose + max_dose) / 2.0;
        let initial_point = vec![initial_guess; num_optimizable];
        let initial_simplex = create_initial_simplex(&initial_point);

        let solver: NelderMead<Vec<f64>, f64> =
            NelderMead::new(initial_simplex).with_sd_tolerance(1e-10)?;

        let opt = Executor::new(objective.clone(), solver)
            .configure(|state| state.max_iters(1000))
            .run()?;

        opt.state().best_param.clone().unwrap()
    };

    // Evaluate once at the optimum to recover the cost and target achievements.
    let evaluation = evaluate(objective, &optimizable_doses)?;
    tracing::debug!("BestDose optimization cost: {:.6}", evaluation.cost);

    // Map optimizable doses back into the full dose vector (fixed doses kept).
    let mut optimized_doses = Vec::with_capacity(all_doses.len());
    let mut opt_idx = 0;
    for &original_dose in all_doses.iter() {
        if original_dose == 0.0 {
            optimized_doses.push(optimizable_doses[opt_idx]);
            opt_idx += 1;
        } else {
            optimized_doses.push(original_dose);
        }
    }

    // Build the optimal subject with the solved dose amounts.
    let mut subject = objective.target.clone();
    let mut dose_number = 0;
    for occasion in subject.iter_mut() {
        for event in occasion.iter_mut() {
            match event {
                Event::Bolus(bolus) => {
                    bolus.set_amount(optimized_doses[dose_number]);
                    dose_number += 1;
                }
                Event::Infusion(infusion) => {
                    infusion.set_amount(optimized_doses[dose_number]);
                    dose_number += 1;
                }
                Event::Observation(_) => {}
            }
        }
    }

    Ok(BestDoseResult {
        subject,
        cost: evaluation.cost,
        achievements: evaluation.achievements,
    })
}
