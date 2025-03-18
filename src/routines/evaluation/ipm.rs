use anyhow::Result;
use clarabel::algebra::*;
use clarabel::solver::*;
use ndarray::{Array1, ArrayBase, OwnedRepr};

use crate::structs::psi::Psi;

/// Alias for a one-dimensional array.
type OneDimArray = ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;

/// Solves the problem
///
///   maximize    ∑₍ᵢ₌₁₎ⁿ log(∑₍ⱼ₌₁₎ᵐ ψ[i,j] * x[j])
///   subject to  x ≥ 0  and  ∑₍ⱼ₌₁₎ᵐ x[j] = 1,
///
/// by reformulating it into conic form via an epigraph (exponential cone) approach.
pub fn burke(psi: &Psi) -> Result<(OneDimArray, f64)> {
    // Retrieve the psi matrix and its dimensions.
    let psi_matrix = psi.matrix(); // shape (n, m)
    let num_rows = psi_matrix.nrows();
    let num_cols = psi_matrix.ncols();
    let num_vars = num_cols + 2 * num_rows;

    // --- Objective ---
    // We want to minimize sum(t) so that at optimum t_i = -log(z_i).
    // Set q = [0 (for x); 1 (for t); 0 (for z)].
    let mut q_vector = vec![0.0; num_vars];
    for i in 0..num_rows {
        q_vector[num_cols + i] = 1.0;
    }
    // P is zero (linear objective).
    let p_matrix = CscMatrix::zeros((num_vars, num_vars));

    // --- Constraint Group 1: Simplex constraint on x ---
    let mut simplex_triplets = Vec::new();
    for j in 0..num_cols {
        simplex_triplets.push((0, j, 1.0));
    }
    let (i1, j1, v1) = convert_triplets(simplex_triplets);
    let a1_matrix = CscMatrix::new_from_triplets(1, num_vars, i1, j1, v1);
    let b1_vector = vec![1.0];

    // --- Constraint Group 2: Linking x and z ---
    let mut linking_triplets = Vec::new();
    for i in 0..num_rows {
        for j in 0..num_cols {
            let value = -psi_matrix[(i, j)];
            if value != 0.0 {
                linking_triplets.push((i, j, value));
            }
        }
        linking_triplets.push((i, num_cols + num_rows + i, 1.0));
    }
    let (i2, j2, v2) = convert_triplets(linking_triplets);
    let a2_matrix = CscMatrix::new_from_triplets(num_rows, num_vars, i2, j2, v2);
    let b2_vector = vec![0.0; num_rows];

    // --- Constraint Group 3: Exponential cone constraints ---
    let group3_rows = 3 * num_rows;
    let mut exp_cone_triplets = Vec::new();
    for i in 0..num_rows {
        let base_row = 3 * i;
        // Row for t_i.
        exp_cone_triplets.push((base_row, num_cols + i, 1.0));
        // Row for z_i.
        exp_cone_triplets.push((base_row + 2, num_cols + num_rows + i, -1.0));
    }
    let (i3, j3, v3) = convert_triplets(exp_cone_triplets);
    let a3_matrix = CscMatrix::new_from_triplets(group3_rows, num_vars, i3, j3, v3);
    let mut b3_vector = Vec::with_capacity(group3_rows);
    for _ in 0..num_rows {
        b3_vector.push(0.0);
        b3_vector.push(1.0);
        b3_vector.push(0.0);
    }

    // --- Constraint Group 4: Nonnegativity of x ---
    let mut nonneg_triplets = Vec::new();
    for j in 0..num_cols {
        nonneg_triplets.push((j, j, -1.0));
    }
    let (i4, j4, v4) = convert_triplets(nonneg_triplets);
    let a4_matrix = CscMatrix::new_from_triplets(num_cols, num_vars, i4, j4, v4);
    let b4_vector = vec![0.0; num_cols];

    // --- Assemble the full constraint matrix and vector ---
    let a_temp = CscMatrix::vcat(&a1_matrix, &a2_matrix)?;
    let a_temp2 = CscMatrix::vcat(&a_temp, &a3_matrix)?;
    let a_matrix = CscMatrix::vcat(&a_temp2, &a4_matrix)?;

    let mut b_vector = Vec::new();
    b_vector.extend(b1_vector);
    b_vector.extend(b2_vector);
    b_vector.extend(b3_vector);
    b_vector.extend(b4_vector);

    // --- Define cone specifications ---
    let mut cones = Vec::new();
    cones.push(ZeroConeT(1)); // Group 1
    cones.push(ZeroConeT(num_rows)); // Group 2
    cones.extend(std::iter::repeat_with(|| ExponentialConeT()).take(num_rows)); // Group 3
    cones.push(NonnegativeConeT(num_cols)); // Group 4

    // --- Solver Settings and solve ---
    let settings = DefaultSettings::default();
    let mut solver =
        DefaultSolver::new(&p_matrix, &q_vector, &a_matrix, &b_vector, &cones, settings);
    solver.solve();

    if solver.solution.status != SolverStatus::Solved {
        return Err(anyhow::anyhow!("Solver did not converge"));
    }

    // The solution vector is ordered as [x; t; z].
    let solution = solver.solution.x;
    let x_optimal = solution[0..num_cols].to_vec();

    // Our conic objective is ∑ t_i; since at optimality t_i = -log(z_i),
    // the original objective (sum of logarithms) is -∑ t_i.
    let t_sum: f64 = solution[num_cols..num_cols + num_rows].iter().sum();
    let original_obj = -t_sum;

    Ok((Array1::from(x_optimal).into(), original_obj))
}

/// Helper function to convert triplets to components
fn convert_triplets(triplets: Vec<(usize, usize, f64)>) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    for (i, j, v) in triplets {
        rows.push(i);
        cols.push(j);
        values.push(v);
    }
    (rows, cols, values)
}
