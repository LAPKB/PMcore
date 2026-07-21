//! Support-point refinement for NPOD.
//!
//! The optimizer sits beside the non-parametric objective because its cost
//! function assembles subject likelihoods over generated predictions.

use argmin::{
    core::{CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};
use ndarray::{Array1, Axis};
use pharmsol::{Data, Equation};

use crate::estimation::likelihood::matrix::nonparametric_log_likelihood_matrix;
use crate::AssayErrorModels;

pub(crate) struct ParameterOptimizer<'a, E: Equation> {
    equation: &'a E,
    data: &'a Data,
    error_models: &'a AssayErrorModels,
    reference_likelihoods: &'a Array1<f64>,
}

impl<E: Equation> CostFunction for ParameterOptimizer<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, parameters: &Self::Param) -> Result<Self::Output, Error> {
        let support_point = Array1::from(parameters.clone()).insert_axis(Axis(0));
        let log_likelihoods = nonparametric_log_likelihood_matrix(
            self.equation,
            self.data,
            &support_point,
            self.error_models,
            false,
        )?;

        if log_likelihoods.ncols() != 1 {
            return Err(Error::msg(format!(
                "support-point optimizer expected one likelihood column, found {}",
                log_likelihoods.ncols()
            )));
        }
        if log_likelihoods.nrows() != self.reference_likelihoods.len() {
            return Err(Error::msg(format!(
                "support-point optimizer has {} subjects but {} reference likelihoods",
                log_likelihoods.nrows(),
                self.reference_likelihoods.len()
            )));
        }

        let n_subjects = log_likelihoods.nrows() as f64;
        let objective = log_likelihoods
            .column(0)
            .iter()
            .zip(self.reference_likelihoods.iter())
            .fold(-n_subjects, |sum, (log_likelihood, reference)| {
                sum + log_likelihood.exp() / reference
            });
        Ok(-objective)
    }
}

impl<'a, E: Equation> ParameterOptimizer<'a, E> {
    pub(crate) fn new(
        equation: &'a E,
        data: &'a Data,
        error_models: &'a AssayErrorModels,
        reference_likelihoods: &'a Array1<f64>,
    ) -> Self {
        Self {
            equation,
            data,
            error_models,
            reference_likelihoods,
        }
    }

    pub(crate) fn optimize_point(self, parameters: Array1<f64>) -> Result<Array1<f64>, Error> {
        let solver =
            NelderMead::new(initial_simplex(&parameters.to_vec())).with_sd_tolerance(1e-2)?;
        let result = Executor::new(self, solver)
            .configure(|state| state.max_iters(5))
            .run()?;
        let best = result
            .state
            .best_param
            .ok_or_else(|| Error::msg("support-point optimizer produced no best parameter"))?;
        Ok(Array1::from(best))
    }
}

fn initial_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let mut simplex = Vec::with_capacity(initial_point.len() + 1);
    simplex.push(initial_point.to_vec());
    for dimension in 0..initial_point.len() {
        let mut point = initial_point.to_vec();
        point[dimension] += if point[dimension] == 0.0 {
            0.00025
        } else {
            0.008 * point[dimension]
        };
        simplex.push(point);
    }
    simplex
}

#[cfg(test)]
mod tests {
    use super::initial_simplex;

    #[test]
    fn initial_simplex_perturbs_each_dimension() {
        let simplex = initial_simplex(&[1.0, 0.0]);
        assert_eq!(
            simplex,
            vec![vec![1.0, 0.0], vec![1.008, 0.0], vec![1.0, 0.00025]]
        );
    }
}
