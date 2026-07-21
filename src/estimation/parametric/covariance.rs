use anyhow::Result;
use ndarray::Array2;

/// Dense identity matrix used for initial Ω placeholders and tests.
pub(crate) fn identity_matrix(size: usize) -> Array2<f64> {
    Array2::from_shape_fn(
        (size, size),
        |(row, col)| if row == col { 1.0 } else { 0.0 },
    )
}

/// Lower Cholesky factor of a symmetric positive-definite covariance matrix.
///
/// Kept small and ndarray-native for now. This is the shared PMcore path for
/// η/Ω prior scoring until a crate-wide linear algebra backend is selected.
pub(crate) fn cholesky_lower(matrix: &Array2<f64>) -> Result<Vec<Vec<f64>>> {
    if matrix.nrows() != matrix.ncols() {
        anyhow::bail!("omega must be square");
    }

    let n = matrix.nrows();
    let mut unit_lower = vec![vec![0.0; n]; n];
    let mut diagonal = vec![0.0; n];
    for row in 0..n {
        for col in 0..row {
            if matrix[[row, col]] != matrix[[col, row]] {
                anyhow::bail!("omega must be symmetric");
            }
            let sum = (0..col)
                .map(|k| unit_lower[row][k] * unit_lower[col][k] * diagonal[k])
                .sum::<f64>();
            unit_lower[row][col] = (matrix[[row, col]] - sum) / diagonal[col];
            if !unit_lower[row][col].is_finite() {
                anyhow::bail!("omega must be positive definite");
            }
        }
        unit_lower[row][row] = 1.0;
        let sum = (0..row)
            .map(|k| unit_lower[row][k].powi(2) * diagonal[k])
            .sum::<f64>();
        diagonal[row] = matrix[[row, row]] - sum;
        if diagonal[row] <= 0.0 || !diagonal[row].is_finite() {
            anyhow::bail!("omega must be positive definite");
        }
    }

    Ok((0..n)
        .map(|row| {
            (0..n)
                .map(|col| {
                    if col <= row {
                        unit_lower[row][col] * diagonal[col].sqrt()
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect())
}

/// Invert an SPD matrix from its unmodified lower Cholesky factor.
///
/// The inverse is formed as `L^-T L^-1`; each lower-triangle value is computed
/// once and copied to its transpose, so the result is exactly symmetric without
/// regularization or numerical repair.
pub(crate) fn inverse_spd_from_cholesky(lower: &[Vec<f64>]) -> Result<Array2<f64>> {
    let n = lower.len();
    if n == 0 || lower.iter().any(|row| row.len() != n) {
        anyhow::bail!("Cholesky factor must be nonempty and square");
    }
    let mut inverse_lower = Array2::<f64>::zeros((n, n));
    for column in 0..n {
        for row in column..n {
            let rhs = if row == column { 1.0 } else { 0.0 };
            let sum = (column..row)
                .map(|k| lower[row][k] * inverse_lower[[k, column]])
                .sum::<f64>();
            let diagonal = lower[row][row];
            if !diagonal.is_finite() || diagonal <= 0.0 {
                anyhow::bail!("Cholesky factor diagonal must be finite and positive");
            }
            inverse_lower[[row, column]] = (rhs - sum) / diagonal;
        }
    }
    if inverse_lower.iter().any(|value| !value.is_finite()) {
        anyhow::bail!("Cholesky inversion produced a non-finite triangular inverse");
    }

    let mut inverse = Array2::<f64>::zeros((n, n));
    for row in 0..n {
        for column in 0..=row {
            let value = (row.max(column)..n)
                .map(|k| inverse_lower[[k, row]] * inverse_lower[[k, column]])
                .sum::<f64>();
            if !value.is_finite() {
                anyhow::bail!("Cholesky inversion produced a non-finite covariance");
            }
            inverse[[row, column]] = value;
            inverse[[column, row]] = value;
        }
    }
    if (0..n).any(|index| inverse[[index, index]] <= 0.0) {
        anyhow::bail!("Cholesky inversion produced a non-positive variance");
    }
    Ok(inverse)
}

/// Compute sqrt(λ_max(L^T V L)) where I = L L^T is the Cholesky decomposition
/// of the observed-information matrix.
///
/// This is the worst-contrast metric: the largest standard-deviation scale of V
/// expressed in I-normalized coordinates. It is invariant under nonsingular
/// linear reparameterization.
///
/// No clamp, jitter, projection, or repair is applied. Fails with a typed
/// error when inputs are non-finite, non-symmetric, dimensionally mismatched,
/// or when the information matrix is not strictly positive definite.
pub(crate) fn worst_contrast(information: &Array2<f64>, variance: &Array2<f64>) -> Result<f64> {
    if information.nrows() != information.ncols() || variance.nrows() != variance.ncols() {
        anyhow::bail!("worst-contrast inputs must be square");
    }
    if information.nrows() != variance.nrows() {
        anyhow::bail!("worst-contrast information and variance dimensions must match");
    }

    let n = information.nrows();

    // Both matrices must be finite and symmetric.
    for row in 0..n {
        for col in 0..n {
            if !information[[row, col]].is_finite() || !variance[[row, col]].is_finite() {
                anyhow::bail!("worst-contrast inputs must be finite");
            }
        }
        for col in 0..row {
            if information[[row, col]] != information[[col, row]] {
                anyhow::bail!("worst-contrast information must be symmetric");
            }
            if variance[[row, col]] != variance[[col, row]] {
                anyhow::bail!("worst-contrast variance must be symmetric");
            }
        }
    }

    let (variance_min, _) = eigenvalue_extrema_symmetric(variance)?;
    if variance_min < 0.0 {
        anyhow::bail!("worst-contrast variance must be positive semidefinite");
    }

    // Cholesky requires strictly positive-definite information.
    let lower = cholesky_lower(information)?;

    // Compute M = L^T V L as a full symmetric matrix.
    let mut m = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in (i.max(j))..n {
                // Accumulate L[k][i] * (V L)[k][j].
                let mut vl = 0.0;
                for p in j..n {
                    vl += variance[[k, p]] * lower[p][j];
                }
                sum += lower[k][i] * vl;
            }
            m[[i, j]] = sum;
            m[[j, i]] = sum;
        }
    }

    let (_, max_eig) = eigenvalue_extrema_symmetric(&m)?;
    if max_eig < 0.0 {
        anyhow::bail!("worst-contrast eigenvalue must be nonnegative, got {max_eig}");
    }
    Ok(max_eig.sqrt())
}

/// Largest eigenvalue of a symmetric matrix via Jacobi rotation.
///
/// Operates on a mutable copy; the original matrix is never modified.
pub(crate) fn eigenvalue_extrema_symmetric(matrix: &Array2<f64>) -> Result<(f64, f64)> {
    let n = matrix.nrows();
    if n == 0 {
        anyhow::bail!("cannot compute eigenvalue of 0x0 matrix");
    }

    // Scalar: identity
    if n == 1 {
        return Ok((matrix[[0, 0]], matrix[[0, 0]]));
    }

    // 2×2: closed form
    if n == 2 {
        let trace = matrix[[0, 0]] + matrix[[1, 1]];
        let det = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
        let discriminant = trace * trace - 4.0 * det;
        if discriminant < 0.0 {
            anyhow::bail!("2x2 worst-contrast discriminant must be nonnegative");
        }
        let root = discriminant.sqrt();
        return Ok((0.5 * (trace - root), 0.5 * (trace + root)));
    }

    // General n×n: symmetric Jacobi iteration.
    let mut work = matrix.clone();
    let max_rotations = 50 * n * n;
    let mut converged = false;
    for _ in 0..max_rotations {
        // Find the largest off-diagonal element.
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.0;
        for row in 0..n {
            for col in 0..row {
                let abs_val = work[[row, col]].abs();
                if abs_val > largest {
                    largest = abs_val;
                    p = row;
                    q = col;
                }
            }
        }
        let scale = work.iter().fold(1.0_f64, |s, x| s.max(x.abs()));
        if largest <= 64.0 * f64::EPSILON * scale {
            converged = true;
            break;
        }
        let angle = 0.5 * (2.0 * work[[p, q]]).atan2(work[[q, q]] - work[[p, p]]);
        let (sin, cos) = angle.sin_cos();
        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let kp = work[[k, p]];
            let kq = work[[k, q]];
            work[[k, p]] = cos * kp - sin * kq;
            work[[p, k]] = work[[k, p]];
            work[[k, q]] = sin * kp + cos * kq;
            work[[q, k]] = work[[k, q]];
        }
        let pp = work[[p, p]];
        let qq = work[[q, q]];
        let pq = work[[p, q]];
        work[[p, p]] = cos * cos * pp - 2.0 * sin * cos * pq + sin * sin * qq;
        work[[q, q]] = sin * sin * pp + 2.0 * sin * cos * pq + cos * cos * qq;
        work[[p, q]] = 0.0;
        work[[q, p]] = 0.0;
    }

    if !converged {
        anyhow::bail!("worst-contrast Jacobi eigensolver did not converge");
    }
    let minimum = (0..n)
        .map(|index| work[[index, index]])
        .fold(f64::INFINITY, f64::min);
    let maximum = (0..n)
        .map(|index| work[[index, index]])
        .fold(f64::NEG_INFINITY, f64::max);
    Ok((minimum, maximum))
}

/// Smallest generalized eigenvalue of `current` relative to `reference`.
///
/// This is the dimensionless SPD-boundary margin
/// `min(x' current x / x' reference x)`. It is positive exactly when both
/// covariance matrices are strictly positive definite and approaches zero as
/// `current` approaches the SPD boundary relative to the declared reference.
pub(crate) fn relative_spd_margin(current: &Array2<f64>, reference: &Array2<f64>) -> Result<f64> {
    if current.dim() != reference.dim() || current.nrows() != current.ncols() {
        anyhow::bail!("relative SPD margin covariance dimensions must match and be square");
    }
    let n = current.nrows();
    if n == 0 {
        anyhow::bail!("relative SPD margin requires a non-empty covariance");
    }
    let reference_lower = cholesky_lower(reference)?;
    cholesky_lower(current)?;

    let mut inverse_lower = Array2::zeros((n, n));
    for col in 0..n {
        let mut unit = vec![0.0; n];
        unit[col] = 1.0;
        let solution = solve_lower(&reference_lower, &unit)?;
        for row in 0..n {
            inverse_lower[[row, col]] = solution[row];
        }
    }

    let mut whitened = Array2::zeros((n, n));
    for row in 0..n {
        for col in 0..=row {
            let mut value = 0.0;
            for left in 0..n {
                for right in 0..n {
                    value += inverse_lower[[row, left]]
                        * current[[left, right]]
                        * inverse_lower[[col, right]];
                }
            }
            whitened[[row, col]] = value;
            whitened[[col, row]] = value;
        }
    }
    let (minimum, _) = eigenvalue_extrema_symmetric(&whitened)?;
    if !minimum.is_finite() || minimum <= 0.0 {
        anyhow::bail!("relative SPD margin must be finite and positive");
    }
    Ok(minimum)
}

/// Solve `lower * x = rhs` for x.
pub(crate) fn solve_lower(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
    if lower.len() != rhs.len() {
        anyhow::bail!("eta length does not match omega dimension");
    }

    let mut solution = vec![0.0; rhs.len()];
    for row in 0..rhs.len() {
        let sum = (0..row)
            .map(|col| lower[row][col] * solution[col])
            .sum::<f64>();
        solution[row] = (rhs[row] - sum) / lower[row][row];
    }
    Ok(solution)
}

pub(crate) fn cholesky_log_determinant(lower: &[Vec<f64>]) -> f64 {
    2.0 * lower
        .iter()
        .enumerate()
        .map(|(index, row)| row[index].ln())
        .sum::<f64>()
}

/// Floor and regularize a covariance matrix until it is positive definite.
///
/// Sparse early SAEM iterations can produce rank-deficient Ω estimates.
/// Numerical robustness work showed that allowing Ω to collapse starves the MCMC kernel,
/// so PMcore applies a diagonal floor and bounded jitter here.
#[cfg(test)]
pub(crate) fn ensure_positive_definite_covariance(
    matrix: &Array2<f64>,
    minimum_variance: f64,
) -> Array2<f64> {
    let n = matrix.nrows();
    let mut candidate = matrix.clone();
    for index in 0..n {
        candidate[[index, index]] = candidate[[index, index]].max(minimum_variance);
    }

    let mut jitter = minimum_variance.max(f64::EPSILON.sqrt());
    for _ in 0..8 {
        if cholesky_lower(&candidate).is_ok() {
            return candidate;
        }
        for index in 0..n {
            candidate[[index, index]] += jitter;
        }
        jitter *= 10.0;
    }

    Array2::from_shape_fn((n, n), |(row, col)| {
        if row == col {
            matrix[[row, row]].max(minimum_variance)
        } else {
            0.0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_log_determinant_handles_correlated_covariance() {
        let omega = ndarray::array![[4.0, 1.0], [1.0, 2.0]];
        let lower = cholesky_lower(&omega).unwrap();
        assert!((cholesky_log_determinant(&lower) - 7.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn cholesky_scaling_rejects_rank_one_and_accepts_high_scale_covariance() {
        assert!(cholesky_lower(&ndarray::array![[4.0, 4.0], [4.0, 4.0]]).is_err());
        assert!(cholesky_lower(&ndarray::array![[1e200, 5e199], [5e199, 1e200]]).is_ok());
    }

    #[test]
    fn relative_spd_margin_matches_known_generalized_eigenvalue() {
        let reference = ndarray::array![[4.0, 1.0], [1.0, 2.0]];
        let lower = cholesky_lower(&reference).unwrap();
        let whitened = ndarray::array![[0.95, -0.75], [-0.75, 0.95]];
        let mut current = Array2::zeros((2, 2));
        for row in 0..2 {
            for col in 0..2 {
                for left in 0..2 {
                    for right in 0..2 {
                        current[[row, col]] +=
                            lower[row][left] * whitened[[left, right]] * lower[col][right];
                    }
                }
            }
        }

        assert!((relative_spd_margin(&current, &reference).unwrap() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn relative_spd_margin_is_invariant_under_dense_congruence() {
        let reference = ndarray::array![[4.0, 1.0], [1.0, 2.0]];
        let current = ndarray::array![[2.0, 0.2], [0.2, 0.5]];
        let transform = ndarray::array![[1.2, -0.4], [0.7, 1.5]];
        let congruence = |matrix: &Array2<f64>| {
            let mut transformed = Array2::zeros((2, 2));
            for row in 0..2 {
                for col in 0..2 {
                    for left in 0..2 {
                        for right in 0..2 {
                            transformed[[row, col]] += transform[[row, left]]
                                * matrix[[left, right]]
                                * transform[[col, right]];
                        }
                    }
                }
            }
            transformed
        };

        let margin = relative_spd_margin(&current, &reference).unwrap();
        let transformed_margin =
            relative_spd_margin(&congruence(&current), &congruence(&reference)).unwrap();
        assert!((margin - transformed_margin).abs() < 1e-12);
    }

    #[test]
    fn relative_spd_margin_handles_scalar_and_rejects_invalid_inputs() {
        let reference = ndarray::array![[0.4]];
        let current = ndarray::array![[0.2]];
        assert!((relative_spd_margin(&current, &reference).unwrap() - 0.5).abs() < 1e-12);

        let singular = ndarray::array![[1.0, 1.0], [1.0, 1.0]];
        let identity = identity_matrix(2);
        assert!(relative_spd_margin(&singular, &identity).is_err());
        assert!(relative_spd_margin(&Array2::zeros((0, 0)), &Array2::zeros((0, 0))).is_err());
    }

    // ── worst-contrast scalar and 2D cases ──

    #[test]
    fn scalar_identity_is_one() {
        let info = ndarray::array![[4.0]];
        let var = ndarray::array![[4.0]];
        // L = [2], L^T V L = 2 * 4 * 2 = 16, sqrt(16) = 4
        assert!((worst_contrast(&info, &var).unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_information_scale_invariance() {
        let info = ndarray::array![[9.0]];
        let var = ndarray::array![[4.0]];
        // L = [3], M = 3 * 4 * 3 = 36, sqrt(36) = 6
        assert!((worst_contrast(&info, &var).unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn two_dimensional_diagonal_has_exact_hand_calculated_value() {
        let info = ndarray::array![[4.0, 0.0], [0.0, 1.0]];
        let var = ndarray::array![[8.0, 0.0], [0.0, 9.0]];
        // L = [[2, 0], [0, 1]], M = L^T V L = [[32, 0], [0, 9]]
        // λ_max = 32, worst = sqrt(32) ≈ 5.656854249492381
        let contrast = worst_contrast(&info, &var).unwrap();
        let expected = 32_f64.sqrt();
        assert!((contrast - expected).abs() < 1e-12);
    }

    #[test]
    fn two_dimensional_correlated_information_variance_has_closed_form() {
        // I = [[5, 2], [2, 2]] is SPD (det = 6 > 0).
        // Cholesky: L = [[√5, 0], [2/√5, √(6/5)]]
        // V = [[3, 1], [1, 2]] is PSD.
        // M = L^T V L = [[20.6, 9√6/5], [9√6/5, 2.4]]
        //   trace = 23, det = 30, λ_max = (23+√409)/2 ≈ 21.61187
        // worst = √λ_max ≈ 4.648857
        let info = ndarray::array![[5.0, 2.0], [2.0, 2.0]];
        let var = ndarray::array![[3.0, 1.0], [1.0, 2.0]];
        let contrast = worst_contrast(&info, &var).unwrap();
        assert!((contrast - 4.648857301324525).abs() < 1e-10);
    }

    // ── linear coordinate invariance ──

    #[test]
    fn nonsingular_linear_coordinate_change_preserves_worst_contrast() {
        let info = ndarray::array![[5.0, 2.0], [2.0, 2.0]];
        let var = ndarray::array![[3.0, 1.0], [1.0, 2.0]];
        let baseline = worst_contrast(&info, &var).unwrap();

        // Nonsingular transformation A; new coordinates x' = A x.
        // Then I' = A^{-T} I A^{-1}, V' = A V A^T.
        let a = ndarray::array![[2.0, 1.0], [-1.0, 3.0]];
        let a_inv = ndarray::array![[3.0 / 7.0, -1.0 / 7.0], [1.0 / 7.0, 2.0 / 7.0]];
        let mut info_transformed = a_inv.t().dot(&info).dot(&a_inv);
        let mut var_transformed = a.dot(&var).dot(&a.t());
        // Force exact symmetry after floating-point operations.
        symmetrize(&mut info_transformed);
        symmetrize(&mut var_transformed);
        let contrast_transformed = worst_contrast(&info_transformed, &var_transformed).unwrap();

        assert!((baseline - contrast_transformed).abs() < 1e-10);
    }

    /// Replace M with (M + M^T)/2.
    fn symmetrize(matrix: &mut Array2<f64>) {
        let n = matrix.nrows();
        for i in 0..n {
            for j in 0..i {
                let avg = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
                matrix[[i, j]] = avg;
                matrix[[j, i]] = avg;
            }
        }
    }

    #[test]
    fn diagonal_scaling_coordinate_change_is_invariant() {
        let info = ndarray::array![[4.0, 1.0], [1.0, 3.0]];
        let var = ndarray::array![[3.0, 0.0], [0.0, 2.0]];
        let baseline = worst_contrast(&info, &var).unwrap();

        // Diagonal scaling: D = diag(3, 7).
        // I' = D^{-1} I D^{-1}, V' = D V D.
        let d_inv = ndarray::array![[1.0 / 3.0, 0.0], [0.0, 1.0 / 7.0]];
        let d = ndarray::array![[3.0, 0.0], [0.0, 7.0]];
        let mut info_scaled = d_inv.dot(&info).dot(&d_inv);
        let mut var_scaled = d.dot(&var).dot(&d);
        symmetrize(&mut info_scaled);
        symmetrize(&mut var_scaled);
        let contrast_scaled = worst_contrast(&info_scaled, &var_scaled).unwrap();

        assert!((baseline - contrast_scaled).abs() < 1e-10);
    }

    #[test]
    fn three_dimensional_correlated_information_and_variance_invariance() {
        // Use a 3×3 known SPD I and PSD V, then apply an orthogonal
        // (rotation) coordinate change and confirm invariance.
        let info = ndarray::array![[5.0, 1.0, 2.0], [1.0, 4.0, 0.0], [2.0, 0.0, 3.0]];
        let var = ndarray::array![[2.0, 0.5, 0.0], [0.5, 3.0, 1.0], [0.0, 1.0, 4.0]];
        let baseline = worst_contrast(&info, &var).unwrap();

        // Rotation with det = 1: x' = A x.
        // I' = A^{-T} I A^{-1} = A I A^T (orthogonal), V' = A V A^T.
        let a = ndarray::array![[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]];
        let mut info_rotated = a.dot(&info).dot(&a.t());
        let mut var_rotated = a.dot(&var).dot(&a.t());
        symmetrize(&mut info_rotated);
        symmetrize(&mut var_rotated);
        let contrast_rotated = worst_contrast(&info_rotated, &var_rotated).unwrap();

        assert!((baseline - contrast_rotated).abs() < 1e-10);
    }

    // ── error branches ──

    #[test]
    fn worst_contrast_rejects_non_square_inputs() {
        let info = ndarray::Array2::<f64>::zeros((1, 2));
        let var = ndarray::Array2::<f64>::zeros((2, 1));
        assert!(worst_contrast(&info, &var).is_err());
    }

    #[test]
    fn worst_contrast_rejects_dimension_mismatch() {
        let info = ndarray::Array2::<f64>::zeros((2, 2));
        let var = ndarray::Array2::<f64>::zeros((3, 3));
        assert!(worst_contrast(&info, &var).is_err());
    }

    #[test]
    fn worst_contrast_rejects_non_finite() {
        let info = ndarray::array![[1.0]];
        let var = ndarray::array![[f64::NAN]];
        assert!(worst_contrast(&info, &var).is_err());
    }

    #[test]
    fn worst_contrast_rejects_non_symmetric_information() {
        let info = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let var = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        assert!(worst_contrast(&info, &var).is_err());
    }

    #[test]
    fn worst_contrast_rejects_non_symmetric_variance() {
        let info = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let var = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        assert!(worst_contrast(&info, &var).is_err());
    }

    #[test]
    fn worst_contrast_rejects_rank_deficient_information() {
        // I = [[1, 1], [1, 1]] has zero eigenvalue, not SPD.
        let info = ndarray::array![[1.0, 1.0], [1.0, 1.0]];
        let var = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        assert!(worst_contrast(&info, &var).is_err());
    }

    #[test]
    fn worst_contrast_accepts_psd_variance_that_is_not_spd() {
        // V is rank-1 PSD but not SPD; should still compute.
        let info = ndarray::array![[4.0, 0.0], [0.0, 1.0]];
        let var = ndarray::array![[1.0, 0.0], [0.0, 0.0]];
        // L = [[2, 0], [0, 1]], M = L^T V L = [[4, 0], [0, 0]]
        // λ_max = 4, worst = 2
        assert!((worst_contrast(&info, &var).unwrap() - 2.0).abs() < 1e-12);
    }
}
