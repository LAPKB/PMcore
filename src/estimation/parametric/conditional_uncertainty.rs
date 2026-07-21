//! Conditional-mode curvature via deterministic central finite-difference Hessian.
//!
//! Computes a strict SPD observed-Fisher-information matrix at a joint
//! latent-coordinate mode for one subject. No regularization, repair, ridge,
//! jitter, clipping, SVD, or pseudoinverse is applied. Any non-finite center,
//! perturbation, or non-SPD Hessian is classified as a typed unavailable
//! status.
//!
//! # Coordinate convention
//!
//! The flattened coordinate order is caller-provided `[eta, kappa...]`. Each
//! coordinate carries a prior standard deviation used for adaptive step sizing.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::covariance::{cholesky_lower, eigenvalue_extrema_symmetric, inverse_spd_from_cholesky};

// ── public serializable types ──────────────────────────────────────────────

/// Kind of one joint latent coordinate in the `[eta, kappa...]` ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JointLatentCoordinateKind {
    /// Inter-individual random effect coordinate.
    Eta {
        /// Index of the associated population parameter.
        parameter_index: usize,
    },
    /// Inter-occasion random effect coordinate.
    Kappa {
        /// Subject-level occasion index.
        occasion_index: usize,
        /// Index within the declared IOV random-effect vector.
        effect_index: usize,
        /// Index of the associated population parameter.
        parameter_index: usize,
    },
}

/// One deterministic latent coordinate with its prior standard deviation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointLatentCoordinate {
    /// Zero-based index in the `[eta, kappa...]` flattened vector.
    pub index: usize,
    /// Human-readable name.
    pub name: String,
    /// Kind and source indices.
    #[serde(flatten)]
    pub kind: JointLatentCoordinateKind,
    /// Prior standard deviation for adaptive step sizing.
    pub prior_sd: f64,
}

/// Explicit no-regularization marker for conditional-mode curvature.
///
/// Always [`ConditionalCurvatureRegularization::None`]. No repair, ridge,
/// jitter, clipping, SVD, or pseudoinverse is ever applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionalCurvatureRegularization {
    /// No regularization of any kind was applied.
    None,
}

/// Convergence metadata for the conditional-mode optimization.
///
/// Informational only; the curvature computation does not depend on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionalModeMetadata {
    /// Whether the optimizer reported convergence.
    pub converged: bool,
    /// Number of iterations taken.
    pub iterations: u64,
    /// Final objective value at the mode.
    pub objective_value: f64,
    /// Human-readable termination reason.
    pub termination_message: String,
}

/// Typed reason why conditional-mode curvature is unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", content = "detail", rename_all = "snake_case")]
pub enum ConditionalCurvatureUnavailableReason {
    /// The mode or a perturbation produced a non-finite objective.
    NonFiniteModeOrPerturbation,
    /// The computed Hessian is not strictly positive definite.
    NonSpdHessian,
    /// The coordinate vector is empty.
    ZeroSize,
    /// Mode, prior-SD, and coordinate dimensions do not match.
    DimensionMismatch {
        mode: usize,
        prior_sds: usize,
        coordinates: usize,
    },
    /// The mode or prior-SD metadata contains a non-finite value.
    NonFiniteInput,
    /// The finite-difference center objective is non-finite.
    NonFiniteCenterObjective,
    /// Cholesky inversion produced non-finite or non-positive diagonal entries.
    InversionFailed,
    /// The spectral condition number is non-finite.
    NonFiniteConditionNumber,
}

/// Availability status of conditional-mode curvature diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "reason", rename_all = "snake_case")]
pub enum ConditionalCurvatureStatus {
    /// Curvature is available with all fields populated.
    Available,
    /// Curvature could not be derived; see reason.
    Unavailable(ConditionalCurvatureUnavailableReason),
}

/// Compatibility name emphasizing that proposal selection consumes only availability.
pub type ConditionalCurvatureAvailability = ConditionalCurvatureStatus;

/// Strict conditional-mode curvature diagnostics for one subject.
///
/// All matrix fields are row-major `Vec<Vec<f64>>` in the same flattened
/// `[eta, kappa...]` order as the coordinates. The observed Fisher
/// information is the second derivative of the negative log-density at the
/// mode; the latent covariance is its strict Cholesky inverse. No
/// regularization, repair, or fallback is applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionalCurvatureDiagnostics {
    /// Flattened latent coordinate definitions.
    pub coordinates: Vec<JointLatentCoordinate>,
    /// Whether the curvature derivation succeeded.
    pub status: ConditionalCurvatureStatus,
    /// Regularization classification (always [`ConditionalCurvatureRegularization::None`]).
    pub regularization: ConditionalCurvatureRegularization,
    /// Observed-Fisher-information inverse (latent covariance).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent_covariance: Option<Vec<Vec<f64>>>,
    /// Diagonal square-roots of [`Self::latent_covariance`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent_standard_errors: Option<Vec<f64>>,
    /// Spectral condition number λ_max / λ_min of the observed-Fisher matrix.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spectral_condition_number: Option<f64>,
    /// Retained finite-difference steps, one per coordinate.
    pub finite_difference_steps: Vec<f64>,
    /// Raw Hessian of the exact subject negative-log-posterior at the mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian: Option<Vec<Vec<f64>>>,
    /// Convergence metadata passed through from the mode optimization.
    pub mode_metadata: ConditionalModeMetadata,
}

impl ConditionalCurvatureDiagnostics {
    /// Construct an unavailable diagnostic with the given reason.
    pub(crate) fn unavailable(
        coordinates: Vec<JointLatentCoordinate>,
        reason: ConditionalCurvatureUnavailableReason,
        mode_metadata: ConditionalModeMetadata,
    ) -> Self {
        Self {
            coordinates,
            status: ConditionalCurvatureStatus::Unavailable(reason),
            regularization: ConditionalCurvatureRegularization::None,
            latent_covariance: None,
            latent_standard_errors: None,
            spectral_condition_number: None,
            finite_difference_steps: Vec::new(),
            hessian: None,
            mode_metadata,
        }
    }
}

// ── pub(crate) central finite-difference Hessian ───────────────────────────

/// Compute strict conditional-mode curvature via central finite-difference
/// Hessian of the exact objective.
///
/// # Arguments
///
/// * `mode` - The joint latent-coordinate mode `[eta, kappa...]`.
/// * `prior_sds` - Prior standard deviations, same length as `mode`.
/// * `coordinates` - Flattened coordinate metadata in mode order.
/// * `mode_metadata` - Convergence metadata (informational only).
/// * `objective` - Exact negative log-density `FnMut(&[f64]) -> f64`.
///
/// # Step sizes
///
/// For each coordinate `i`:
///
/// ```text
/// h_i = ε^(1/4) × max(1, |mode_i|, prior_sd_i)
/// ```
///
/// where ε = [`f64::EPSILON`].
///
/// # Hessian formulas
///
/// *Diagonal* (central second derivative):
///
/// ```text
/// H_ii = [f(x + h_i e_i) - 2f(x) + f(x - h_i e_i)] / h_i²
/// ```
///
/// *Off-diagonal* (mixed central partial derivative):
///
/// ```text
/// H_ij = [f_++ - f_+- - f_-+ + f_--] / (4 h_i h_j)
/// ```
///
/// where `f_ab = f(x + a·h_i e_i + b·h_j e_j)`.
///
/// # Strictness
///
/// No regularization, repair, ridge, jitter, clipping, SVD, or pseudoinverse
/// is applied. Non-finite center/perturbation and non-SPD Hessians produce a
/// typed [`ConditionalCurvatureStatus::Unavailable`] result.
pub(crate) fn conditional_mode_curvature<F: FnMut(&[f64]) -> f64>(
    mode: &[f64],
    prior_sds: &[f64],
    coordinates: &[JointLatentCoordinate],
    mode_metadata: &ConditionalModeMetadata,
    mut objective: F,
) -> ConditionalCurvatureDiagnostics {
    let n = mode.len();

    // ── guard: zero-size ──────────────────────────────────────────────
    if n == 0 {
        return ConditionalCurvatureDiagnostics::unavailable(
            coordinates.to_vec(),
            ConditionalCurvatureUnavailableReason::ZeroSize,
            mode_metadata.clone(),
        );
    }
    if prior_sds.len() != n || coordinates.len() != n {
        return ConditionalCurvatureDiagnostics::unavailable(
            coordinates.to_vec(),
            ConditionalCurvatureUnavailableReason::DimensionMismatch {
                mode: n,
                prior_sds: prior_sds.len(),
                coordinates: coordinates.len(),
            },
            mode_metadata.clone(),
        );
    }
    if mode.iter().any(|value| !value.is_finite())
        || prior_sds
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return ConditionalCurvatureDiagnostics::unavailable(
            coordinates.to_vec(),
            ConditionalCurvatureUnavailableReason::NonFiniteInput,
            mode_metadata.clone(),
        );
    }

    // ── step sizes ────────────────────────────────────────────────────
    let eps_quarter = f64::EPSILON.powf(0.25);
    let steps: Vec<f64> = mode
        .iter()
        .zip(prior_sds)
        .map(|(value, sd)| eps_quarter * f64::max(f64::max(1.0, value.abs()), *sd))
        .collect();

    // ── center evaluation ─────────────────────────────────────────────
    let f_center = objective(mode);
    if !f_center.is_finite() {
        return ConditionalCurvatureDiagnostics::unavailable(
            coordinates.to_vec(),
            ConditionalCurvatureUnavailableReason::NonFiniteCenterObjective,
            mode_metadata.clone(),
        );
    }

    // ── forward / backward evaluations per coordinate ─────────────────
    let mut point_plus = mode.to_vec();
    let mut point_minus = mode.to_vec();
    let mut f_plus = vec![0.0; n];
    let mut f_minus = vec![0.0; n];

    for i in 0..n {
        point_plus[i] = mode[i] + steps[i];
        f_plus[i] = objective(&point_plus);
        point_plus[i] = mode[i]; // restore

        point_minus[i] = mode[i] - steps[i];
        f_minus[i] = objective(&point_minus);
        point_minus[i] = mode[i]; // restore
    }

    // ── check perturbation finiteness ─────────────────────────────────
    if !f_center.is_finite()
        || f_plus.iter().any(|v| !v.is_finite())
        || f_minus.iter().any(|v| !v.is_finite())
    {
        return ConditionalCurvatureDiagnostics::unavailable(
            coordinates.to_vec(),
            ConditionalCurvatureUnavailableReason::NonFiniteModeOrPerturbation,
            mode_metadata.clone(),
        );
    }

    // ── diagonal entries ──────────────────────────────────────────────
    let mut hessian = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let h_i = steps[i];
        hessian[[i, i]] = (f_plus[i] - 2.0 * f_center + f_minus[i]) / (h_i * h_i);
    }

    // ── off-diagonal entries via mixed central differences ────────────
    let mut work = mode.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let h_i = steps[i];
            let h_j = steps[j];

            // f_++: +h_i, +h_j
            work[i] = mode[i] + h_i;
            work[j] = mode[j] + h_j;
            let f_pp = objective(&work);

            // f_+-: +h_i, -h_j
            work[j] = mode[j] - h_j;
            let f_pm = objective(&work);

            // f_-+: -h_i, +h_j
            work[i] = mode[i] - h_i;
            work[j] = mode[j] + h_j;
            let f_mp = objective(&work);

            // f_--: -h_i, -h_j
            work[j] = mode[j] - h_j;
            let f_mm = objective(&work);

            // Restore
            work[i] = mode[i];
            work[j] = mode[j];

            if !f_pp.is_finite() || !f_pm.is_finite() || !f_mp.is_finite() || !f_mm.is_finite() {
                return ConditionalCurvatureDiagnostics::unavailable(
                    coordinates.to_vec(),
                    ConditionalCurvatureUnavailableReason::NonFiniteModeOrPerturbation,
                    mode_metadata.clone(),
                );
            }

            let mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_i * h_j);
            hessian[[i, j]] = mixed;
            hessian[[j, i]] = mixed;
        }
    }

    // ── check finite Hessian ──────────────────────────────────────────
    if !hessian.iter().all(|v| v.is_finite()) {
        return ConditionalCurvatureDiagnostics::unavailable(
            coordinates.to_vec(),
            ConditionalCurvatureUnavailableReason::NonFiniteModeOrPerturbation,
            mode_metadata.clone(),
        );
    }

    // ── strict SPD Cholesky ───────────────────────────────────────────
    // The observed Fisher information is the Hessian itself at the mode.
    let fisher = hessian.clone();

    let lower = match cholesky_lower(&fisher) {
        Ok(lower) => lower,
        Err(_) => {
            return ConditionalCurvatureDiagnostics::unavailable(
                coordinates.to_vec(),
                ConditionalCurvatureUnavailableReason::NonSpdHessian,
                mode_metadata.clone(),
            );
        }
    };

    // ── strict Cholesky inverse ───────────────────────────────────────
    let covariance = match inverse_spd_from_cholesky(&lower) {
        Ok(covariance) => covariance,
        Err(_) => {
            return ConditionalCurvatureDiagnostics::unavailable(
                coordinates.to_vec(),
                ConditionalCurvatureUnavailableReason::InversionFailed,
                mode_metadata.clone(),
            );
        }
    };

    // ── latent standard errors ────────────────────────────────────────
    let latent_se: Vec<f64> = (0..n).map(|idx| covariance[[idx, idx]].sqrt()).collect();
    if latent_se.iter().any(|v| !v.is_finite()) {
        return ConditionalCurvatureDiagnostics::unavailable(
            coordinates.to_vec(),
            ConditionalCurvatureUnavailableReason::InversionFailed,
            mode_metadata.clone(),
        );
    }

    // ── spectral condition number ─────────────────────────────────────
    let (lambda_min, lambda_max) = match eigenvalue_extrema_symmetric(&fisher) {
        Ok(extrema) => extrema,
        Err(_) => {
            return ConditionalCurvatureDiagnostics::unavailable(
                coordinates.to_vec(),
                ConditionalCurvatureUnavailableReason::InversionFailed,
                mode_metadata.clone(),
            );
        }
    };
    if !lambda_min.is_finite() || !lambda_max.is_finite() || lambda_min <= 0.0 {
        return ConditionalCurvatureDiagnostics::unavailable(
            coordinates.to_vec(),
            ConditionalCurvatureUnavailableReason::NonFiniteConditionNumber,
            mode_metadata.clone(),
        );
    }
    let cond = lambda_max / lambda_min;
    if !cond.is_finite() {
        return ConditionalCurvatureDiagnostics::unavailable(
            coordinates.to_vec(),
            ConditionalCurvatureUnavailableReason::NonFiniteConditionNumber,
            mode_metadata.clone(),
        );
    }

    // ── convert ndarray to Vec<Vec<f64>> for serialization ────────────
    let fisher_rows: Vec<Vec<f64>> = fisher.rows().into_iter().map(|row| row.to_vec()).collect();
    let cov_rows: Vec<Vec<f64>> = covariance
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    ConditionalCurvatureDiagnostics {
        coordinates: coordinates.to_vec(),
        status: ConditionalCurvatureStatus::Available,
        regularization: ConditionalCurvatureRegularization::None,
        latent_covariance: Some(cov_rows),
        latent_standard_errors: Some(latent_se),
        spectral_condition_number: Some(cond),
        finite_difference_steps: steps,
        hessian: Some(fisher_rows),
        mode_metadata: mode_metadata.clone(),
    }
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal 2D coordinate list: eta on param 0, eta on param 1.
    fn two_eta_coordinates(sd0: f64, sd1: f64) -> Vec<JointLatentCoordinate> {
        vec![
            JointLatentCoordinate {
                index: 0,
                name: "eta:CL".into(),
                kind: JointLatentCoordinateKind::Eta { parameter_index: 0 },
                prior_sd: sd0,
            },
            JointLatentCoordinate {
                index: 1,
                name: "eta:V".into(),
                kind: JointLatentCoordinateKind::Eta { parameter_index: 1 },
                prior_sd: sd1,
            },
        ]
    }

    fn single_eta_coordinate(sd: f64) -> Vec<JointLatentCoordinate> {
        vec![JointLatentCoordinate {
            index: 0,
            name: "eta:CL".into(),
            kind: JointLatentCoordinateKind::Eta { parameter_index: 0 },
            prior_sd: sd,
        }]
    }

    fn default_mode_metadata() -> ConditionalModeMetadata {
        ConditionalModeMetadata {
            converged: true,
            iterations: 42,
            objective_value: 0.0,
            termination_message: "test".into(),
        }
    }

    // ── exact quadratic 2D ────────────────────────────────────────────────

    /// Exact quadratic: f(eta) = 0.5 * eta^T H eta, where H = [[4, 1], [1, 2]].
    /// Mode at (0, 0) with f_center = 0.
    fn quadratic_2d(eta: &[f64]) -> f64 {
        let x = eta[0];
        let y = eta[1];
        0.5 * (4.0 * x * x + 2.0 * x * y + 2.0 * y * y)
    }

    #[test]
    fn exact_quadratic_2d_hessian_and_covariance_recovery() {
        let mode = vec![0.0, 0.0];
        let prior_sds = vec![1.0, 1.0];
        let coords = two_eta_coordinates(1.0, 1.0);
        let metadata = default_mode_metadata();

        let diagnostics =
            conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, quadratic_2d);

        assert_eq!(diagnostics.status, ConditionalCurvatureStatus::Available);
        assert_eq!(
            diagnostics.regularization,
            ConditionalCurvatureRegularization::None
        );
        assert_eq!(diagnostics.coordinates.len(), 2);

        // Hessian should be close to [[4, 1], [1, 2]].
        let fisher = diagnostics.hessian.unwrap();
        assert!(
            (fisher[0][0] - 4.0).abs() < 1e-8,
            "H[0,0] = {} expected 4.0",
            fisher[0][0]
        );
        assert!(
            (fisher[0][1] - 1.0).abs() < 1e-8,
            "H[0,1] = {} expected 1.0",
            fisher[0][1]
        );
        assert!(
            (fisher[1][1] - 2.0).abs() < 1e-8,
            "H[1,1] = {} expected 2.0",
            fisher[1][1]
        );

        // Covariance (= H^{-1}): det = 8-1 = 7.
        // H^{-1} = 1/7 * [[2, -1], [-1, 4]] ≈ [[0.285714, -0.142857], [-0.142857, 0.571429]]
        let cov = diagnostics.latent_covariance.unwrap();
        assert!(
            (cov[0][0] - 2.0 / 7.0).abs() < 1e-8,
            "cov[0,0] = {}",
            cov[0][0]
        );
        assert!(
            (cov[0][1] + 1.0 / 7.0).abs() < 1e-8,
            "cov[0,1] = {}",
            cov[0][1]
        );
        assert!(
            (cov[1][0] + 1.0 / 7.0).abs() < 1e-8,
            "cov[1,0] = {}",
            cov[1][0]
        );
        assert!(
            (cov[1][1] - 4.0 / 7.0).abs() < 1e-8,
            "cov[1,1] = {}",
            cov[1][1]
        );

        // Standard errors.
        let se = diagnostics.latent_standard_errors.unwrap();
        assert!((se[0] - (2.0_f64 / 7.0).sqrt()).abs() < 1e-8);
        assert!((se[1] - (4.0_f64 / 7.0).sqrt()).abs() < 1e-8);

        // Spectral condition number.
        // Eigenvalues of H = [[4,1],[1,2]]: λ = (6 ± √(36-28))/2 = (6 ± √8)/2 = 3 ± √2.
        // λ_max = 3 + √2, λ_min = 3 - √2.
        let expected_condition = (3.0 + 2.0_f64.sqrt()) / (3.0 - 2.0_f64.sqrt());
        let cond = diagnostics.spectral_condition_number.unwrap();
        assert!(
            (cond - expected_condition).abs() < 1e-8,
            "condition = {} expected {}",
            cond,
            expected_condition
        );
    }

    #[test]
    fn three_dimensional_covariance_is_exactly_symmetric_for_proposal_reuse() {
        let mode = vec![0.0; 3];
        let prior_sds = vec![1.0; 3];
        let coordinates = (0..3)
            .map(|index| JointLatentCoordinate {
                index,
                name: format!("eta:{index}"),
                kind: JointLatentCoordinateKind::Eta {
                    parameter_index: index,
                },
                prior_sd: 1.0,
            })
            .collect::<Vec<_>>();
        let diagnostics = conditional_mode_curvature(
            &mode,
            &prior_sds,
            &coordinates,
            &default_mode_metadata(),
            |x| {
                0.5 * (4.0 * x[0] * x[0]
                    + 3.0 * x[1] * x[1]
                    + 2.0 * x[2] * x[2]
                    + 2.0 * x[0] * x[1]
                    + x[0] * x[2]
                    + 0.5 * x[1] * x[2])
            },
        );
        assert_eq!(diagnostics.status, ConditionalCurvatureStatus::Available);
        let covariance = diagnostics.latent_covariance.unwrap();
        for (row, values) in covariance.iter().enumerate() {
            for (column, value) in values.iter().enumerate().take(row) {
                assert_eq!(value.to_bits(), covariance[column][row].to_bits());
            }
        }
    }

    // ── mode off-origin with high prior SD in step ────────────────────────

    #[test]
    fn quadratic_2d_mode_off_origin_with_high_prior_sd() {
        // f(eta) = 0.5 * eta^T H eta with mode at (5, -3).
        // Shift coordinates: let u = eta - mode, then f(u) = 0.5 * u^T H u.
        // The Hessian is unchanged.
        let mode = vec![5.0, -3.0];
        let prior_sds = vec![10.0, 20.0]; // high prior SDs → bigger steps
        let coords = two_eta_coordinates(10.0, 20.0);
        let metadata = default_mode_metadata();

        let diagnostics =
            conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, |eta| {
                let u0 = eta[0] - 5.0;
                let u1 = eta[1] + 3.0;
                0.5 * (4.0 * u0 * u0 + 2.0 * u0 * u1 + 2.0 * u1 * u1)
            });

        assert_eq!(diagnostics.status, ConditionalCurvatureStatus::Available);
        let fisher = diagnostics.hessian.unwrap();
        assert!(
            (fisher[0][0] - 4.0).abs() < 1e-6,
            "off-origin H[0,0] = {}",
            fisher[0][0]
        );
        assert!(
            (fisher[0][1] - 1.0).abs() < 1e-6,
            "off-origin H[0,1] = {}",
            fisher[0][1]
        );
    }

    // ── 1D quadratic ──────────────────────────────────────────────────────

    #[test]
    fn scalar_quadratic_hessian_and_covariance() {
        // f(x) = 0.5 * 9 * x^2, Hessian = 9, covariance = 1/9.
        let mode = vec![0.0];
        let prior_sds = vec![1.0];
        let coords = single_eta_coordinate(1.0);
        let metadata = default_mode_metadata();

        let diagnostics = conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, |x| {
            4.5 * x[0] * x[0]
        });

        assert_eq!(diagnostics.status, ConditionalCurvatureStatus::Available);
        let fisher = diagnostics.hessian.unwrap();
        assert!((fisher[0][0] - 9.0).abs() < 1e-8);
        let cov = diagnostics.latent_covariance.unwrap();
        assert!((cov[0][0] - 1.0 / 9.0).abs() < 1e-8);
        let se = diagnostics.latent_standard_errors.unwrap();
        assert!((se[0] - 1.0 / 3.0).abs() < 1e-8);
        assert_eq!(diagnostics.spectral_condition_number.unwrap(), 1.0);
    }

    // ── typed unavailable: non-SPD ────────────────────────────────────────

    #[test]
    fn non_positive_definite_hessian_yields_typed_unavailable() {
        // f(x,y) = 0.5 * (x^2 - y^2) has Hessian = [[1, 0], [0, -1]] (indefinite).
        let mode = vec![0.0, 0.0];
        let prior_sds = vec![1.0, 1.0];
        let coords = two_eta_coordinates(1.0, 1.0);
        let metadata = default_mode_metadata();

        let diagnostics =
            conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, |eta| {
                0.5 * (eta[0] * eta[0] - eta[1] * eta[1])
            });

        assert!(
            matches!(
                diagnostics.status,
                ConditionalCurvatureStatus::Unavailable(
                    ConditionalCurvatureUnavailableReason::NonSpdHessian
                )
            ),
            "expected NonSpdHessian, got {:?}",
            diagnostics.status
        );
        assert!(diagnostics.latent_covariance.is_none());
        assert!(diagnostics.latent_standard_errors.is_none());
        assert!(diagnostics.spectral_condition_number.is_none());
        // The Hessian is omitted because strict Cholesky classification failed.
    }

    // ── typed unavailable: NaN center ─────────────────────────────────────

    #[test]
    fn nan_center_yields_typed_unavailable() {
        let mode = vec![0.0];
        let prior_sds = vec![1.0];
        let coords = single_eta_coordinate(1.0);
        let metadata = default_mode_metadata();

        let diagnostics =
            conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, |_| f64::NAN);

        assert!(matches!(
            diagnostics.status,
            ConditionalCurvatureStatus::Unavailable(
                ConditionalCurvatureUnavailableReason::NonFiniteCenterObjective
            )
        ));
    }

    // ── typed unavailable: NaN perturbation ───────────────────────────────

    #[test]
    fn nan_perturbation_yields_typed_unavailable() {
        // Objective returns NaN when x[0] is perturbed away from 0.
        let mode = vec![0.0];
        let prior_sds = vec![1.0];
        let coords = single_eta_coordinate(1.0);
        let metadata = default_mode_metadata();

        let diagnostics = conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, |x| {
            if (x[0] - mode[0]).abs() > f64::EPSILON {
                f64::NAN
            } else {
                0.0
            }
        });

        assert!(matches!(
            diagnostics.status,
            ConditionalCurvatureStatus::Unavailable(
                ConditionalCurvatureUnavailableReason::NonFiniteModeOrPerturbation
            )
        ));
    }

    // ── typed unavailable: zero size ──────────────────────────────────────

    #[test]
    fn zero_size_yields_typed_unavailable() {
        let mode: Vec<f64> = vec![];
        let prior_sds: Vec<f64> = vec![];
        let coords: Vec<JointLatentCoordinate> = vec![];
        let metadata = default_mode_metadata();

        let diagnostics =
            conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, |_| 0.0);

        assert!(matches!(
            diagnostics.status,
            ConditionalCurvatureStatus::Unavailable(
                ConditionalCurvatureUnavailableReason::ZeroSize
            )
        ));
    }

    // ── typed unavailable: 2D non-SPD via zero eigenvalue ─────────────────

    #[test]
    fn rank_deficient_hessian_yields_typed_unavailable() {
        // f(x,y) = 0.5 * (x + y)^2 has Hessian = [[1,1],[1,1]] (rank 1).
        let mode = vec![0.0, 0.0];
        let prior_sds = vec![1.0, 1.0];
        let coords = two_eta_coordinates(1.0, 1.0);
        let metadata = default_mode_metadata();

        let diagnostics =
            conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, |eta| {
                let s = eta[0] + eta[1];
                0.5 * s * s
            });

        assert!(matches!(
            diagnostics.status,
            ConditionalCurvatureStatus::Unavailable(
                ConditionalCurvatureUnavailableReason::NonSpdHessian
            )
        ));
    }

    // ── serde roundtrip for all public types ──────────────────────────────

    #[test]
    fn joint_latent_coordinate_kind_serde_roundtrip() {
        for kind in [
            JointLatentCoordinateKind::Eta { parameter_index: 3 },
            JointLatentCoordinateKind::Kappa {
                occasion_index: 7,
                effect_index: 1,
                parameter_index: 3,
            },
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let roundtripped: JointLatentCoordinateKind = serde_json::from_str(&json).unwrap();
            assert_eq!(roundtripped, kind);
        }
    }

    #[test]
    fn conditional_curvature_status_serde_roundtrip() {
        let available = ConditionalCurvatureStatus::Available;
        let json = serde_json::to_string(&available).unwrap();
        assert!(json.contains("available"));
        let rt: ConditionalCurvatureStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(rt, ConditionalCurvatureStatus::Available);

        let unavailable = ConditionalCurvatureStatus::Unavailable(
            ConditionalCurvatureUnavailableReason::NonSpdHessian,
        );
        let json = serde_json::to_string(&unavailable).unwrap();
        assert!(json.contains("non_spd_hessian"));
        let rt: ConditionalCurvatureStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(rt, unavailable);
    }

    #[test]
    fn conditional_curvature_diagnostics_full_serde_roundtrip() {
        let coords = two_eta_coordinates(1.0, 1.0);
        let metadata = default_mode_metadata();
        let mode = vec![0.0, 0.0];
        let prior_sds = vec![1.0, 1.0];

        let diagnostics =
            conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, quadratic_2d);

        assert_eq!(diagnostics.status, ConditionalCurvatureStatus::Available);

        let json = serde_json::to_string(&diagnostics).unwrap();
        let roundtripped: ConditionalCurvatureDiagnostics = serde_json::from_str(&json).unwrap();

        assert_eq!(roundtripped.status, ConditionalCurvatureStatus::Available);
        assert_eq!(
            roundtripped.regularization,
            ConditionalCurvatureRegularization::None
        );
        assert!(roundtripped.latent_covariance.is_some());
        assert!(roundtripped.latent_standard_errors.is_some());
        assert!(roundtripped.spectral_condition_number.is_some());
        assert!(roundtripped.hessian.is_some());
        assert!(roundtripped.mode_metadata.converged);
        assert_eq!(roundtripped.mode_metadata.iterations, 42);
    }

    #[test]
    fn unavailable_diagnostics_serde_roundtrip() {
        let diagnostics = ConditionalCurvatureDiagnostics::unavailable(
            two_eta_coordinates(1.0, 1.0),
            ConditionalCurvatureUnavailableReason::NonSpdHessian,
            default_mode_metadata(),
        );

        let json = serde_json::to_string(&diagnostics).unwrap();
        assert!(json.contains("non_spd_hessian"));
        let roundtripped: ConditionalCurvatureDiagnostics = serde_json::from_str(&json).unwrap();

        assert!(matches!(
            roundtripped.status,
            ConditionalCurvatureStatus::Unavailable(
                ConditionalCurvatureUnavailableReason::NonSpdHessian
            )
        ));
        assert!(roundtripped.latent_covariance.is_none());
        assert!(roundtripped.latent_standard_errors.is_none());
    }

    #[test]
    fn conditional_curvature_regularization_serde_roundtrip() {
        let reg = ConditionalCurvatureRegularization::None;
        let json = serde_json::to_string(&reg).unwrap();
        assert!(json.contains("none"));
        let rt: ConditionalCurvatureRegularization = serde_json::from_str(&json).unwrap();
        assert_eq!(rt, ConditionalCurvatureRegularization::None);
    }

    // ── metadata passthrough ──────────────────────────────────────────────

    #[test]
    fn mode_metadata_is_preserved_in_output() {
        let mode = vec![0.0, 0.0];
        let prior_sds = vec![1.0, 1.0];
        let coords = two_eta_coordinates(1.0, 1.0);
        let metadata = ConditionalModeMetadata {
            converged: false,
            iterations: 100,
            objective_value: -123.456,
            termination_message: "max iterations".into(),
        };

        let diagnostics =
            conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, quadratic_2d);

        assert!(!diagnostics.mode_metadata.converged);
        assert_eq!(diagnostics.mode_metadata.iterations, 100);
        assert_eq!(diagnostics.mode_metadata.objective_value, -123.456);
        assert_eq!(
            diagnostics.mode_metadata.termination_message,
            "max iterations"
        );
        // Curvature should still be available even for non-converged mode.
        assert_eq!(diagnostics.status, ConditionalCurvatureStatus::Available);
    }

    // ── kappa coordinate kind ─────────────────────────────────────────────

    #[test]
    fn kappa_coordinates_are_handled() {
        // 1 eta + 1 kappa → 2D mode.
        let coords = vec![
            JointLatentCoordinate {
                index: 0,
                name: "eta:CL".into(),
                kind: JointLatentCoordinateKind::Eta { parameter_index: 0 },
                prior_sd: 1.0,
            },
            JointLatentCoordinate {
                index: 1,
                name: "kappa:CL:0".into(),
                kind: JointLatentCoordinateKind::Kappa {
                    occasion_index: 0,
                    effect_index: 0,
                    parameter_index: 0,
                },
                prior_sd: 0.5,
            },
        ];

        let mode = vec![0.2, -0.1];
        let prior_sds = vec![1.0, 0.5];
        let metadata = default_mode_metadata();

        let diagnostics =
            conditional_mode_curvature(&mode, &prior_sds, &coords, &metadata, |eta| {
                let x = eta[0];
                let y = eta[1];
                0.5 * (4.0 * x * x + 2.0 * x * y + 3.0 * y * y)
            });

        assert_eq!(diagnostics.status, ConditionalCurvatureStatus::Available);
        let fisher = diagnostics.hessian.unwrap();
        // Hessian = [[4, 1], [1, 3]]
        assert!((fisher[0][0] - 4.0).abs() < 1e-6);
        assert!((fisher[0][1] - 1.0).abs() < 1e-6);
        assert!((fisher[1][1] - 3.0).abs() < 1e-6);
    }
}
