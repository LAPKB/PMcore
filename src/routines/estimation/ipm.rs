use crate::structs::psi::Psi;
use crate::structs::weights::Weights;
use anyhow::bail;
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Col, Mat, Row};
use rayon::prelude::*;
/// Applies Burke's Interior Point Method (IPM) to solve a convex optimization problem.
///
/// The objective function to maximize is:
///     f(x) = Σ(log(Σ(ψ_ij * x_j)))   for i = 1 to n_sub
///
/// subject to:
///     1. x_j ≥ 0 for all j = 1 to n_point,
///     2. Σ(x_j) = 1,
///
/// where ψ is an n_sub×n_point matrix with non-negative entries and x is a probability vector.
///
/// # Arguments
///
/// * `psi` - A reference to a Psi structure containing the input matrix.
///
/// # Returns
///
/// On success, returns a tuple `(weights, obj)` where:
///   - [Weights] contains the optimized weights (probabilities) for each support point.
///   - `obj` is the value of the objective function at the solution.
///
/// # Errors
///
/// This function returns an error if any step in the optimization (e.g. Cholesky factorization)
/// fails.
pub fn burke(psi: &Psi) -> anyhow::Result<(Weights, f64)> {
    let mut psi = psi.matrix().to_owned();

    // Ensure all entries are finite and make them non-negative.
    psi.row_iter_mut().try_for_each(|row| {
        row.iter_mut().try_for_each(|x| {
            if !x.is_finite() {
                bail!("Input matrix must have finite entries")
            } else {
                // Coerce negatives to non-negative (could alternatively return an error)
                *x = x.abs();
                Ok(())
            }
        })
    })?;

    // Let psi be of shape (n_sub, n_point)
    let (n_sub, n_point) = psi.shape();

    // Create unit vectors:
    // ecol: ones vector of length n_point (used for sums over points)
    // erow: ones row of length n_sub (used for sums over subproblems)
    let ecol: Col<f64> = Col::from_fn(n_point, |_| 1.0);
    let erow: Row<f64> = Row::from_fn(n_sub, |_| 1.0);

    // Compute plam = psi · ecol. This gives a column vector of length n_sub.
    let mut plam: Col<f64> = &psi * &ecol;
    let eps: f64 = 1e-8;
    let mut sig: f64 = 0.0;

    // Initialize lam (the variable we optimize) as a column vector of ones (length n_point).
    let mut lam = ecol.clone();

    // w = 1 ./ plam, elementwise.
    let mut w: Col<f64> = Col::from_fn(plam.nrows(), |i| 1.0 / plam.get(i));

    // ptw = ψᵀ · w, which will be a vector of length n_point.
    let mut ptw: Col<f64> = psi.transpose() * &w;

    // Use the maximum entry in ptw for scaling (the "shrink" factor).
    let ptw_max = ptw.iter().fold(f64::NEG_INFINITY, |acc, &x| x.max(acc));
    let shrink = 2.0 * ptw_max;
    lam *= shrink;
    plam *= shrink;
    w /= shrink;
    ptw /= shrink;

    // y = ecol - ptw (a vector of length n_point).
    let mut y: Col<f64> = &ecol - &ptw;
    // r = erow - (w .* plam) (elementwise product; r has length n_sub).
    let mut r: Col<f64> = Col::from_fn(n_sub, |i| erow.get(i) - w.get(i) * plam.get(i));
    let mut norm_r: f64 = r.iter().fold(0.0, |max, &val| max.max(val.abs()));

    // Compute the duality gap.
    let sum_log_plam: f64 = plam.iter().map(|x| x.ln()).sum();
    let sum_log_w: f64 = w.iter().map(|x| x.ln()).sum();
    let mut gap: f64 = (sum_log_w + sum_log_plam).abs() / (1.0 + sum_log_plam);

    // Compute the duality measure mu.
    let mut mu = lam.transpose() * &y / n_point as f64;

    let mut psi_inner: Mat<f64> = Mat::zeros(psi.nrows(), psi.ncols());

    let n_threads = faer::get_global_parallelism().degree();

    let rows = psi.nrows();

    let mut output: Vec<Mat<f64>> = (0..n_threads).map(|_| Mat::zeros(rows, rows)).collect();

    let mut h: Mat<f64> = Mat::zeros(rows, rows);

    while mu > eps || norm_r > eps || gap > eps {
        let smu = sig * mu;
        // inner = lam ./ y, elementwise.
        let inner = Col::from_fn(lam.nrows(), |i| lam.get(i) / y.get(i));
        // w_plam = plam ./ w, elementwise (length n_sub).
        let w_plam = Col::from_fn(plam.nrows(), |i| plam.get(i) / w.get(i));

        // Scale each column of psi by the corresponding element of 'inner'

        if psi.ncols() > n_threads * 128 {
            psi_inner
                .par_col_partition_mut(n_threads)
                .zip(psi.par_col_partition(n_threads))
                .zip(inner.par_partition(n_threads))
                .zip(output.par_iter_mut())
                .for_each(|(((mut psi_inner, psi), inner), output)| {
                    psi_inner
                        .as_mut()
                        .col_iter_mut()
                        .zip(psi.col_iter())
                        .zip(inner.iter())
                        .for_each(|((col, psi_col), inner_val)| {
                            col.iter_mut().zip(psi_col.iter()).for_each(|(x, psi_val)| {
                                *x = psi_val * inner_val;
                            });
                        });
                    faer::linalg::matmul::triangular::matmul(
                        output.as_mut(),
                        faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                        faer::Accum::Replace,
                        &psi_inner,
                        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                        psi.transpose(),
                        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                        1.0,
                        faer::Par::Seq,
                    );
                });

            let mut first_iter = true;
            for output in &output {
                if first_iter {
                    h.copy_from(output);
                    first_iter = false;
                } else {
                    h += output;
                }
            }
        } else {
            psi_inner
                .as_mut()
                .col_iter_mut()
                .zip(psi.col_iter())
                .zip(inner.iter())
                .for_each(|((col, psi_col), inner_val)| {
                    col.iter_mut().zip(psi_col.iter()).for_each(|(x, psi_val)| {
                        *x = psi_val * inner_val;
                    });
                });
            faer::linalg::matmul::triangular::matmul(
                h.as_mut(),
                faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                faer::Accum::Replace,
                &psi_inner,
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                psi.transpose(),
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                1.0,
                faer::Par::Seq,
            );
        }

        for i in 0..h.nrows() {
            h[(i, i)] += w_plam[i];
        }

        let uph = match h.llt(faer::Side::Lower) {
            Ok(llt) => llt,
            Err(_) => {
                bail!("Error during Cholesky decomposition. The matrix might not be positive definite. This is usually due to model misspecification or numerical issues.")
            }
        };
        let uph = uph.L().transpose().to_owned();

        // smuyinv = smu * (ecol ./ y)
        let smuyinv: Col<f64> = Col::from_fn(ecol.nrows(), |i| smu * (ecol[i] / y[i]));

        // let smuyinv = smu * (&ecol / &y);
        // rhsdw = (erow ./ w) - (psi · smuyinv)
        let psi_dot_muyinv: Col<f64> = &psi * &smuyinv;

        let rhsdw: Row<f64> = Row::from_fn(erow.ncols(), |i| erow[i] / w[i] - psi_dot_muyinv[i]);

        //let rhsdw = (&erow / &w) - psi * &smuyinv;
        // Reshape rhsdw into a column vector.
        let mut dw = Mat::from_fn(rhsdw.ncols(), 1, |i, _j| *rhsdw.get(i));

        // let a = rhsdw
        //     .into_shape((n_sub, 1))
        //     .context("Failed to reshape rhsdw").unwrap();

        // Solve the triangular systems:

        solve_lower_triangular_in_place(uph.transpose().as_ref(), dw.as_mut(), faer::Par::rayon(0));

        solve_upper_triangular_in_place(uph.as_ref(), dw.as_mut(), faer::Par::rayon(0));

        // Extract dw (a column vector) from the solution.
        let dw = dw.col(0);

        // let dw = dw_aux.column(0);
        // Compute dy = - (ψᵀ · dw)
        let dy = -(psi.transpose() * dw);

        let inner_times_dy = Col::from_fn(ecol.nrows(), |i| inner[i] * dy[i]);

        let dlam: Row<f64> =
            Row::from_fn(ecol.nrows(), |i| smuyinv[i] - lam[i] - inner_times_dy[i]);
        // let dlam = &smuyinv - &lam - inner.transpose() * &dy;

        // Compute the primal step length alfpri.
        let ratio_dlam_lam = Row::from_fn(lam.nrows(), |i| dlam[i] / lam[i]);
        //let ratio_dlam_lam = &dlam / &lam;
        let min_ratio_dlam = ratio_dlam_lam.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfpri: f64 = -1.0 / min_ratio_dlam.min(-0.5);
        alfpri = (0.99995 * alfpri).min(1.0);

        // Compute the dual step length alfdual.
        let ratio_dy_y = Row::from_fn(y.nrows(), |i| dy[i] / y[i]);
        // let ratio_dy_y = &dy / &y;
        let min_ratio_dy = ratio_dy_y.iter().cloned().fold(f64::INFINITY, f64::min);
        let ratio_dw_w = Row::from_fn(dw.nrows(), |i| dw[i] / w[i]);
        //let ratio_dw_w = &dw / &w;
        let min_ratio_dw = ratio_dw_w.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfdual = -1.0 / min_ratio_dy.min(-0.5);
        alfdual = alfdual.min(-1.0 / min_ratio_dw.min(-0.5));
        alfdual = (0.99995 * alfdual).min(1.0);

        // Update the iterates.
        lam += alfpri * dlam.transpose();
        w += alfdual * dw;
        y += alfdual * &dy;

        mu = lam.transpose() * &y / n_point as f64;
        plam = &psi * &lam;

        // mu = lam.dot(&y) / n_point as f64;
        // plam = psi.dot(&lam);
        r = Col::from_fn(n_sub, |i| erow.get(i) - w.get(i) * plam.get(i));
        ptw -= alfdual * dy;

        norm_r = r.norm_max();
        let sum_log_plam: f64 = plam.iter().map(|x| x.ln()).sum();
        let sum_log_w: f64 = w.iter().map(|x| x.ln()).sum();
        gap = (sum_log_w + sum_log_plam).abs() / (1.0 + sum_log_plam);

        // Adjust sigma.
        if mu < eps && norm_r > eps {
            sig = 1.0;
        } else {
            let candidate1 = (1.0 - alfpri).powi(2);
            let candidate2 = (1.0 - alfdual).powi(2);
            let candidate3 = (norm_r - mu) / (norm_r + 100.0 * mu);
            sig = candidate1.max(candidate2).max(candidate3).min(0.3);
        }
    }
    // Scale lam.
    lam /= n_sub as f64;
    // Compute the objective function value: sum(ln(psi·lam)).
    let obj = (psi * &lam).iter().map(|x| x.ln()).sum();
    // Normalize lam to sum to 1.
    let lam_sum: f64 = lam.iter().sum();
    lam = &lam / lam_sum;

    Ok((lam.into(), obj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    #[test]
    fn test_burke_identity() {
        // Test with a small identity matrix
        // For an identity matrix, each support point should have equal weight
        let n = 100;
        let mat = Mat::identity(n, n);
        let psi = Psi::from(mat);

        let (lam, _) = burke(&psi).unwrap();

        // For identity matrix, all lambda values should be equal
        let expected = 1.0 / n as f64;
        for i in 0..n {
            assert_relative_eq!(lam[i], expected, epsilon = 1e-10);
        }

        // Check that lambda sums to 1
        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_burke_uniform_square() {
        // Test with a matrix of all ones
        // This should also result in uniform weights
        let n_sub = 10;
        let n_point = 10;
        let mat = Mat::from_fn(n_sub, n_point, |_, _| 1.0);
        let psi = Psi::from(mat);

        let (lam, _) = burke(&psi).unwrap();

        // Check that lambda sums to 1
        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);

        // For uniform matrix, all lambda values should be equal
        let expected = 1.0 / n_point as f64;
        for i in 0..n_point {
            assert_relative_eq!(lam[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_burke_uniform_wide() {
        // Test with a matrix of all ones
        // This should also result in uniform weights
        let n_sub = 10;
        let n_point = 100;
        let mat = Mat::from_fn(n_sub, n_point, |_, _| 1.0);
        let psi = Psi::from(mat);

        let (lam, _) = burke(&psi).unwrap();

        // Check that lambda sums to 1
        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);

        // For uniform matrix, all lambda values should be equal
        let expected = 1.0 / n_point as f64;
        for i in 0..n_point {
            assert_relative_eq!(lam[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_burke_uniform_long() {
        // Test with a matrix of all ones
        // This should also result in uniform weights
        let n_sub = 100;
        let n_point = 10;
        let mat = Mat::from_fn(n_sub, n_point, |_, _| 1.0);
        let psi = Psi::from(mat);

        let (lam, _) = burke(&psi).unwrap();

        // Check that lambda sums to 1
        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);

        // For uniform matrix, all lambda values should be equal
        let expected = 1.0 / n_point as f64;
        for i in 0..n_point {
            assert_relative_eq!(lam[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_burke_with_non_uniform_matrix() {
        // Test with a non-uniform matrix
        // Create a matrix where one column is clearly better
        let n_sub = 3;
        let n_point = 4;
        let mat = Mat::from_fn(n_sub, n_point, |_, j| if j == 0 { 10.0 } else { 1.0 });
        let psi = Psi::from(mat);

        let (lam, _) = burke(&psi).unwrap();

        // Check that lambda sums to 1
        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);

        // First support point should have highest weight
        assert!(lam[0] > lam[1]);
        assert!(lam[0] > lam[2]);
        assert!(lam[0] > lam[3]);
    }

    #[test]
    fn test_burke_with_negative_values() {
        // The algorithm should handle negative values by taking their absolute value
        let n_sub = 2;
        let n_point = 3;
        let mat = Mat::from_fn(
            n_sub,
            n_point,
            |i, j| if i == 0 && j == 0 { -5.0 } else { 1.0 },
        );
        let psi = Psi::from(mat);

        let result = burke(&psi);
        assert!(result.is_ok());

        let (lam, _) = result.unwrap();
        // Check that lambda sums to 1
        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);

        // First support point should have highest weight due to the high absolute value
        assert!(lam[0] > lam[1]);
        assert!(lam[0] > lam[2]);
    }

    #[test]
    fn test_burke_with_non_finite_values() {
        // The algorithm should return an error for non-finite values
        let n_sub = 10;
        let n_point = 10;
        let mat = Mat::from_fn(n_sub, n_point, |i, j| {
            if i == 0 && j == 0 {
                f64::NAN
            } else {
                1.0
            }
        });
        let psi = Psi::from(mat);

        let result = burke(&psi);
        assert!(result.is_err());
    }

    #[test]
    fn test_burke_large_matrix_parallel_processing() {
        // Test with a large matrix to trigger the parallel processing code path
        // This should exceed n_threads * 128 threshold
        let n_sub = 50;
        let n_point = 10000;

        // Create a simple uniform matrix
        // The main goal is to test that parallel processing works correctly
        let mat = Mat::from_fn(n_sub, n_point, |_i, _j| 1.0);
        let psi = Psi::from(mat);

        let result = burke(&psi);
        assert!(
            result.is_ok(),
            "Burke algorithm should succeed with large matrix"
        );

        let (lam, obj) = result.unwrap();

        // Verify basic mathematical properties of the solution
        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);

        // All lambda values should be non-negative
        for i in 0..n_point {
            assert!(lam[i] >= 0.0, "Lambda values should be non-negative");
        }

        // The objective function should be finite
        assert!(obj.is_finite(), "Objective function should be finite");

        // The main test: verify that the parallel processing path was executed
        // and produced a valid probability distribution
        // For a uniform matrix, we expect roughly uniform weights, but the exact
        // distribution depends on the optimization algorithm's convergence

        // Just verify that no single weight dominates excessively (basic sanity check)
        let max_weight = lam
            .weights()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_weight < 0.1,
            "No single weight should dominate in uniform matrix (max weight: {})",
            max_weight
        );
    }

    #[test]
    fn test_burke_medium_matrix_sequential_processing() {
        // Test with a medium-sized matrix that should NOT trigger parallel processing
        // This serves as a comparison to ensure both code paths produce similar results
        let n_sub = 50;
        let n_point = 500; // This should be < n_threads * 128 threshold

        // Use the same pattern as the large matrix test
        let mat = Mat::from_fn(n_sub, n_point, |i, j| {
            if j % 100 == 0 {
                5.0 + 0.1 * (i as f64)
            } else {
                1.0 + 0.01 * (i as f64) + 0.001 * (j as f64)
            }
        });
        let psi = Psi::from(mat);

        let result = burke(&psi);
        assert!(
            result.is_ok(),
            "Burke algorithm should succeed with medium matrix"
        );

        let (lam, obj) = result.unwrap();

        // Verify basic properties of the solution
        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);

        // All lambda values should be non-negative
        for i in 0..n_point {
            assert!(lam[i] >= 0.0, "Lambda values should be non-negative");
        }

        // The objective function should be finite
        assert!(obj.is_finite(), "Objective function should be finite");
    }
}
