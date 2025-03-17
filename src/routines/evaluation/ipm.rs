use crate::structs::psi::Psi;
use anyhow::Result;

use good_lp::*;
use ndarray::{ArrayBase, OwnedRepr};
type OneDimArray = ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;
/// Applies the Burke's Interior Point Method (IPM) to solve a specific optimization problem.
///
/// The Burke's IPM is an iterative optimization technique used for solving convex optimization
/// problems. It is applied to a matrix `psi`, iteratively updating variables and calculating
/// an objective function until convergence.
///
/// The objective function to maximize is:
/// f(x) = Σ(log(Σ(ψ_ij * x_j))) for i = 1 to n_sub
///
/// Subject to the constraints:
/// 1. x_j >= 0 for all j = 1 to n_point
/// 2. Σ(x_j) = 1 for j = 1 to n_point
///
/// Where:
/// - ψ is an n_sub x n_point matrix with non-negative entries.
/// - x is a probability vector of length n_point.
///
/// # Arguments
///
/// * `psi` - A reference to a 2D Array representing the input matrix for optimization.
///
/// # Returns
///
/// A `Result` containing a tuple with two elements:
///
/// * `lam` - An `Array1<f64>` representing the solution of the optimization problem.
/// * `obj` - A f64 value representing the objective function value at the solution.
///
/// # Errors
///
/// This function returns an error if any of the optimization steps encounter issues. The error
/// type is a boxed dynamic error (`Box<dyn error::Error>`).
///
/// # Example
///
/// Note: This function applies the Interior Point Method (IPM) to iteratively update variables
/// until convergence, solving the convex optimization problem.
///
pub fn burke(psi: &Psi) -> Result<(OneDimArray, f64)> {
    let psi = psi.matrix();

    let mut vars = ProblemVariables::new();
    let w = vars.add_vector(VariableDefinition::new().min(0.0), psi.ncols());

    // Objective should be sum(log(psi * w))
    let mut objective = Expression::from(0.0);
    for i in 0..psi.nrows() {
        let row = psi.row(i);
        let mut dot_product = Expression::from(0.0);

        for j in 0..psi.ncols() {
            dot_product += row[j] * w[j];
        }

        // Can't use logarithm with linear programming expressions
        // Using the dot product directly as a linear approximation
        objective += dot_product;
    }

    let sum_expr: Expression = vars.iter_variables_with_def().map(|(var, _)| var).sum();

    let solution = vars
        .maximise(&objective)
        .using(default_solver)
        .with(constraint!(sum_expr <= 1.0))
        .solve()
        .unwrap();

    let mut w_vec = w
        .iter()
        .map(|var| solution.value(*var))
        .collect::<Vec<f64>>();

    w_vec.iter_mut().for_each(|x| *x = x.max(0.0));

    let objf = objective.eval_with(&solution);

    println!("Optimal w: {:?}", &w_vec);
    println!("Objective function value: {}", &objf);

    //let w_row = Row::from_fn(w_vec.len(), |i| w_vec[i]);

    Ok((w_vec.into(), objf))
}
