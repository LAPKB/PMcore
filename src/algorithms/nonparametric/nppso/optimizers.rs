//! Optimizers for NPPSO algorithm
//!
//! Contains subject MAP optimization using COBYLA and D-optimal refinement.

use anyhow::Result;
use cobyla::{minimize, RhoBeg};
use ndarray::Axis;
use pharmsol::prelude::{
    data::{Data, ErrorModels},
    simulator::Equation,
};
use pharmsol::Subject;

// ============================================================================
// SUBJECT MAP OPTIMIZER (using COBYLA)
// ============================================================================

/// Find the MAP (Maximum A Posteriori) estimate for a single subject
/// This identifies the parameter values that best explain that subject's data
pub fn optimize_subject_map<E: Equation>(
    equation: &E,
    subject: &Subject,
    error_models: &ErrorModels,
    ranges: &[(f64, f64)],
    start: &[f64],
    max_evals: usize,
) -> Result<Vec<f64>> {
    // Create single-subject data
    let single_data = Data::new(vec![subject.clone()]);

    // Closure that computes -log P(y|θ) for this subject
    // We minimize this (= maximize P(y|θ))
    let objective = |params: &[f64], _: &mut (&E, &Data, &ErrorModels)| -> f64 {
        // Clamp to bounds
        let clamped: Vec<f64> = params
            .iter()
            .zip(ranges.iter())
            .map(|(v, (lo, hi))| v.clamp(*lo, *hi))
            .collect();

        let theta = ndarray::Array1::from(clamped).insert_axis(Axis(0));

        match pharmsol::prelude::simulator::psi(
            equation,
            &single_data,
            &theta,
            error_models,
            false,
            false,
        ) {
            Ok(psi) => {
                let p = psi.iter().next().unwrap_or(&1e-300);
                if *p > 0.0 {
                    -p.ln() // Minimize -log P = maximize P
                } else {
                    700.0 // Very bad
                }
            }
            Err(_) => 700.0,
        }
    };

    // Convert ranges to cobyla format
    let bounds: Vec<(f64, f64)> = ranges.to_vec();
    let cons: Vec<fn(&[f64], &mut (&E, &Data, &ErrorModels)) -> f64> = vec![];

    // User data for the closure (not actually used in our objective)
    let user_data = (equation, &single_data, error_models);

    let result = minimize(
        objective,
        start,
        &bounds,
        &cons,
        user_data,
        max_evals,
        RhoBeg::All(0.1),
        None,
    );

    match result {
        Ok((_, x, _)) => {
            // Clamp result to bounds with margin
            let clamped: Vec<f64> = x
                .iter()
                .zip(ranges.iter())
                .map(|(v, (lo, hi))| {
                    let margin = (hi - lo) * 0.01;
                    v.clamp(lo + margin, hi - margin)
                })
                .collect();
            Ok(clamped)
        }
        Err((_, x, _)) => {
            // Even on "failure", use the best point found
            let clamped: Vec<f64> = x
                .iter()
                .zip(ranges.iter())
                .map(|(v, (lo, hi))| {
                    let margin = (hi - lo) * 0.01;
                    v.clamp(lo + margin, hi - margin)
                })
                .collect();
            Ok(clamped)
        }
    }
}

// ============================================================================
// D-OPTIMAL REFINEMENT (using COBYLA)
// ============================================================================

/// Refine a support point position to maximize D-criterion
pub fn refine_d_optimal<E: Equation>(
    equation: &E,
    data: &Data,
    error_models: &ErrorModels,
    pyl: &ndarray::Array1<f64>,
    ranges: &[(f64, f64)],
    start: &[f64],
    max_evals: usize,
) -> Result<Vec<f64>> {
    // Closure that computes -D(θ) (we minimize, so negate)
    let objective = |params: &[f64], _: &mut ()| -> f64 {
        // Clamp to bounds
        let clamped: Vec<f64> = params
            .iter()
            .zip(ranges.iter())
            .map(|(v, (lo, hi))| v.clamp(*lo, *hi))
            .collect();

        let theta = ndarray::Array1::from(clamped).insert_axis(Axis(0));

        match pharmsol::prelude::simulator::psi(
            equation,
            data,
            &theta,
            error_models,
            false,
            false,
        ) {
            Ok(psi) => {
                let nsub = psi.nrows() as f64;
                let mut d_sum = -nsub;

                for (p_i, pyl_i) in psi.iter().zip(pyl.iter()) {
                    if *pyl_i > 1e-300 {
                        d_sum += p_i / pyl_i;
                    }
                }

                -d_sum // Minimize -D = Maximize D
            }
            Err(_) => 1e10, // Very bad
        }
    };

    let bounds: Vec<(f64, f64)> = ranges.to_vec();
    let cons: Vec<fn(&[f64], &mut ()) -> f64> = vec![];

    let result = minimize(
        objective,
        start,
        &bounds,
        &cons,
        (),
        max_evals,
        RhoBeg::All(0.05),
        None,
    );

    match result {
        Ok((_, x, _)) | Err((_, x, _)) => {
            let clamped: Vec<f64> = x
                .iter()
                .zip(ranges.iter())
                .map(|(v, (lo, hi))| {
                    let margin = (hi - lo) * 0.01;
                    v.clamp(lo + margin, hi - margin)
                })
                .collect();
            Ok(clamped)
        }
    }
}

// ============================================================================
// ELITE POINT
// ============================================================================

/// An elite point preserved across cycles
#[derive(Debug, Clone)]
pub struct ElitePoint {
    pub params: Vec<f64>,
    #[allow(dead_code)]
    pub d_value: f64,
    pub cycle_added: usize,
}
