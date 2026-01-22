//! Optimizers for NPOPT algorithm

use super::constants::*;
use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, Axis};
use pharmsol::prelude::{
    data::{Data, AssayErrorModels},
    simulator::Equation,
};
use pharmsol::Subject;

// ============================================================================
// D-OPTIMAL OPTIMIZER
// ============================================================================

/// Optimizer for D-criterion maximization
pub struct DOptimalOptimizer<'a, E: Equation> {
    pub equation: &'a E,
    pub data: &'a Data,
    pub error_models: &'a AssayErrorModels,
    pub pyl: &'a Array1<f64>,
}

impl<E: Equation> CostFunction for DOptimalOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, spp: &Self::Param) -> Result<Self::Output, Error> {
        let theta = Array1::from(spp.clone()).insert_axis(Axis(0));

        let psi = pharmsol::prelude::simulator::psi(
            self.equation,
            self.data,
            &theta,
            self.error_models,
            false,
            false,
        )?;

        let nsub = psi.nrows() as f64;
        let mut d_sum = -nsub;
        for (p_i, pyl_i) in psi.iter().zip(self.pyl.iter()) {
            if *pyl_i > 0.0 {
                d_sum += p_i / pyl_i;
            }
        }

        Ok(-d_sum) // Minimize -D = Maximize D
    }
}

impl<'a, E: Equation> DOptimalOptimizer<'a, E> {
    /// Optimize a point using Nelder-Mead
    pub fn optimize(self, start: Vec<f64>, max_iters: u64) -> Result<Vec<f64>, Error> {
        let simplex = create_initial_simplex(&start, 0.05);
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(simplex).with_sd_tolerance(1e-3)?;

        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(max_iters))
            .run()?;

        Ok(res.state.best_param.unwrap())
    }
}

// ============================================================================
// SUBJECT MAP OPTIMIZER
// ============================================================================

/// Optimizer for finding MAP estimate for a single subject
pub struct SubjectMapOptimizer<'a, E: Equation> {
    pub equation: &'a E,
    pub subject: &'a Subject,
    pub error_models: &'a AssayErrorModels,
    pub ranges: &'a [(f64, f64)],
}

impl<E: Equation> CostFunction for SubjectMapOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        // Clamp to bounds
        let clamped: Vec<f64> = params
            .iter()
            .zip(self.ranges.iter())
            .map(|(v, (lo, hi))| v.clamp(*lo, *hi))
            .collect();

        // Create single-subject data
        let single_data = Data::new(vec![self.subject.clone()]);
        let theta = ndarray::Array1::from(clamped).insert_axis(Axis(0));

        let psi = pharmsol::prelude::simulator::psi(
            self.equation,
            &single_data,
            &theta,
            self.error_models,
            false,
            false,
        )?;

        // Minimize -log P(y|θ) = Maximize P(y|θ)
        let p = psi.iter().next().unwrap_or(&1e-300);
        let log_p = if *p > 0.0 { p.ln() } else { -700.0 };

        Ok(-log_p)
    }
}

impl<'a, E: Equation> SubjectMapOptimizer<'a, E> {
    /// Optimize to find MAP estimate
    pub fn optimize(self, start: Vec<f64>, max_iters: u64) -> Result<Vec<f64>, Error> {
        let ranges = self.ranges;
        let simplex = create_initial_simplex_bounded(&start, ranges, 0.05);
        let solver: NelderMead<Vec<f64>, f64> = NelderMead::new(simplex).with_sd_tolerance(1e-3)?;

        let res = Executor::new(self, solver)
            .configure(|state| state.max_iters(max_iters))
            .run()?;

        // Clamp result to bounds
        let result = res.state.best_param.unwrap();
        let clamped: Vec<f64> = result
            .iter()
            .zip(ranges.iter())
            .map(|(v, (lo, hi))| {
                let margin = (hi - lo) * BOUNDARY_MARGIN;
                v.clamp(lo + margin, hi - margin)
            })
            .collect();

        Ok(clamped)
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Create initial simplex for Nelder-Mead
fn create_initial_simplex(initial_point: &[f64], perturbation_frac: f64) -> Vec<Vec<f64>> {
    let num_dims = initial_point.len();

    let mut vertices = Vec::with_capacity(num_dims + 1);
    vertices.push(initial_point.to_vec());

    for i in 0..num_dims {
        let perturbation = if initial_point[i] == 0.0 {
            0.001
        } else {
            perturbation_frac * initial_point[i].abs()
        };

        let mut perturbed = initial_point.to_vec();
        perturbed[i] += perturbation;
        vertices.push(perturbed);
    }

    vertices
}

/// Create initial simplex with bounds awareness
fn create_initial_simplex_bounded(
    initial_point: &[f64],
    ranges: &[(f64, f64)],
    perturbation_frac: f64,
) -> Vec<Vec<f64>> {
    let num_dims = initial_point.len();

    let mut vertices = Vec::with_capacity(num_dims + 1);
    vertices.push(initial_point.to_vec());

    for i in 0..num_dims {
        let (lo, hi) = ranges[i];
        let range = hi - lo;
        let perturbation = perturbation_frac * range;

        let mut perturbed = initial_point.to_vec();
        let new_val = initial_point[i] + perturbation;

        if new_val <= hi {
            perturbed[i] = new_val;
        } else {
            perturbed[i] = initial_point[i] - perturbation;
        }

        vertices.push(perturbed);
    }

    vertices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_creation() {
        let point = vec![1.0, 2.0, 3.0];
        let simplex = create_initial_simplex(&point, 0.05);

        assert_eq!(simplex.len(), 4); // n+1 vertices
        assert_eq!(simplex[0], point);
    }

    #[test]
    fn test_simplex_bounded() {
        let point = vec![0.5, 0.95];
        let ranges = vec![(0.0, 1.0), (0.0, 1.0)];
        let simplex = create_initial_simplex_bounded(&point, &ranges, 0.05);

        assert_eq!(simplex.len(), 3);
        for vertex in &simplex {
            for (i, val) in vertex.iter().enumerate() {
                assert!(*val >= ranges[i].0 && *val <= ranges[i].1);
            }
        }
    }
}
