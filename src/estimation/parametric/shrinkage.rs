//! Eta and kappa shrinkage diagnostics.
//!
//! Shrinkage quantifies how much individual empirical Bayes estimates are
//! pulled toward the population mean. A value near 100% indicates that
//! individual estimates are tightly clustered around the population mean
//! and provide little independent information; a value near 0% indicates
//! that individual estimates spread out about as much as the estimated
//! population variance.
//!
//! ## Units and denominators
//!
//! - **Eta (η) shrinkage**: one estimate per subject — the denominator
//!   `unit_count` equals the number of subjects.
//! - **Kappa (κ) shrinkage**: one estimate per subject-occasion pair,
//!   pooled across all subjects — the denominator `unit_count` equals
//!   the total number of subject-occasion pairs.
//!
//! ## Formula
//!
//! ```text
//! shrinkage = 100 × (1 − sample_variance_{N−1} / final_variance)
//! ```
//!
//! where `sample_variance_{N−1}` is the unbiased (N − 1) sample variance of
//! the individual estimates across units, and `final_variance` is the
//! corresponding diagonal element of the final population covariance matrix
//! (Ω for eta, inter-occasion covariance for kappa).
//!
//! No clamping is applied. Values below 0% (sample variance exceeds final
//! variance) and above 100% (floating-point edge cases) are preserved as-is.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shrinkage value and unavailable reasons
// ---------------------------------------------------------------------------

/// Computed shrinkage for one effect, or a typed reason the value is
/// unavailable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ShrinkageValue {
    /// Shrinkage was successfully computed.
    Available {
        /// Shrinkage percentage (no clamping applied).
        value: f64,
        /// Number of independent units in the denominator.
        ///
        /// For eta: number of subjects.
        /// For kappa: number of pooled subject-occasion pairs.
        unit_count: usize,
        /// Human-readable documentation of what `unit_count` counts.
        denominator_documentation: String,
    },
    /// Shrinkage could not be computed.
    Unavailable { reason: ShrinkageUnavailableReason },
}

/// Typed reason that a shrinkage value is unavailable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", content = "detail", rename_all = "snake_case")]
pub enum ShrinkageUnavailableReason {
    /// Fewer than 2 independent units.
    ///
    /// At least 2 units are required to compute a meaningful sample variance
    /// with the N−1 denominator.
    TooFewUnits { count: usize },
    /// MAP estimates were requested but are not available.
    MissingMap,
    /// One or more input values are non-finite (NaN or infinity).
    NonFiniteValue,
    /// The reference population variance is non-positive (zero, negative, or
    /// non-finite).
    NonPositiveReferenceVariance { variance: f64 },
    /// Width mismatch between effect names, posterior rows, and covariance
    /// diagonal.
    WidthMismatch {
        effect_names: usize,
        row_width: usize,
        variance_len: usize,
    },
}

// ---------------------------------------------------------------------------
// Per-effect shrinkage result types
// ---------------------------------------------------------------------------

/// Eta (η) shrinkage computed from subject posterior means.
///
/// One value per random effect. The denominator `unit_count` equals the
/// number of subjects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtaPosteriorMeanShrinkage {
    /// Random-effect name.
    pub effect: String,
    /// Computed shrinkage or unavailable reason.
    pub shrinkage: ShrinkageValue,
}

/// Eta (η) shrinkage computed from subject MAP (maximum a posteriori)
/// estimates.
///
/// One value per random effect. The denominator `unit_count` equals the
/// number of subjects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtaMapShrinkage {
    /// Random-effect name.
    pub effect: String,
    /// Computed shrinkage or unavailable reason.
    pub shrinkage: ShrinkageValue,
}

/// Kappa (κ) shrinkage computed from occasion posterior means.
///
/// One value per inter-occasion random effect. The denominator `unit_count`
/// equals the number of pooled subject-occasion pairs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KappaPosteriorMeanShrinkage {
    /// Random-effect name.
    pub effect: String,
    /// Computed shrinkage or unavailable reason.
    pub shrinkage: ShrinkageValue,
}

/// Kappa (κ) shrinkage computed from occasion MAP (maximum a posteriori)
/// estimates.
///
/// One value per inter-occasion random effect. The denominator `unit_count`
/// equals the number of pooled subject-occasion pairs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KappaMapShrinkage {
    /// Random-effect name.
    pub effect: String,
    /// Computed shrinkage or unavailable reason.
    pub shrinkage: ShrinkageValue,
}

/// Source-explicit shrinkage diagnostics retained by a parametric result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShrinkageDiagnostics {
    pub eta_posterior_mean: Vec<EtaPosteriorMeanShrinkage>,
    pub eta_map: Vec<EtaMapShrinkage>,
    pub kappa_posterior_mean: Vec<KappaPosteriorMeanShrinkage>,
    pub kappa_map: Vec<KappaMapShrinkage>,
}

// ---------------------------------------------------------------------------
// Internal constants
// ---------------------------------------------------------------------------

const ETA_DENOMINATOR: &str = "number of subjects";
const KAPPA_DENOMINATOR: &str = "number of pooled subject-occasion pairs";

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

/// Compute 100 × (1 − sample_variance_{N−1} / final_variance).
///
/// Returns `Ok(value)` on success or `Err(reason)` on any precondition
/// failure. No clamping is applied — the caller receives the raw result.
fn compute_shrinkage(
    unit_values: &[f64],
    final_variance: f64,
    unit_count: usize,
) -> Result<f64, ShrinkageUnavailableReason> {
    // Validate final variance must be strictly positive and finite.
    if !final_variance.is_finite() || final_variance <= 0.0 {
        return Err(ShrinkageUnavailableReason::NonPositiveReferenceVariance {
            variance: final_variance,
        });
    }

    // Validate all unit values are finite.
    if unit_values.iter().any(|v| !v.is_finite()) {
        return Err(ShrinkageUnavailableReason::NonFiniteValue);
    }

    // At least two units required for N−1 sample variance.
    if unit_count < 2 {
        return Err(ShrinkageUnavailableReason::TooFewUnits { count: unit_count });
    }

    // Sample mean.
    let mean = unit_values.iter().sum::<f64>() / unit_count as f64;

    // Unbiased sample variance (N−1 denominator).
    let sample_var = unit_values
        .iter()
        .map(|v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f64>()
        / (unit_count - 1) as f64;

    // No clamping: negative and >100 preserved if achievable.
    let shrinkage = 100.0 * (1.0 - sample_var / final_variance);
    Ok(shrinkage)
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

/// Validate that effect names, row widths, and final variance diagonal have
/// matching dimensions and that all rows are uniformly wide.
fn validate_widths(
    effect_names: &[String],
    rows: &[Vec<f64>],
    final_var_diag: &[f64],
) -> Result<(), ShrinkageUnavailableReason> {
    let effect_count = effect_names.len();
    let row_width = rows.first().map_or(effect_count, |r| r.len());
    let variance_len = final_var_diag.len();

    if effect_count != row_width || row_width != variance_len {
        return Err(ShrinkageUnavailableReason::WidthMismatch {
            effect_names: effect_count,
            row_width,
            variance_len,
        });
    }

    // Every row must have the same width.
    if rows.iter().any(|r| r.len() != row_width) {
        return Err(ShrinkageUnavailableReason::WidthMismatch {
            effect_names: effect_count,
            row_width,
            variance_len,
        });
    }

    Ok(())
}

/// Extract the column (by effect index) from all rows.
fn extract_column(rows: &[Vec<f64>], col: usize) -> Vec<f64> {
    rows.iter().map(|r| r[col]).collect()
}

// ---------------------------------------------------------------------------
// Public derivation functions
// ---------------------------------------------------------------------------

/// Derive eta (η) shrinkage for each random effect from subject posterior
/// means.
///
/// # Parameters
///
/// - `effect_names` — one name per random effect, in order.
/// - `final_var_diag` — diagonal of the final Ω covariance matrix, one entry
///   per effect.
/// - `posterior_mean_rows` — one row per subject, one column per effect.
///   Each row is the posterior mean η vector for that subject.
///
/// # Denominator
///
/// `unit_count` = number of rows = number of subjects.
pub(crate) fn derive_eta_posterior_mean_shrinkage(
    effect_names: &[String],
    final_var_diag: &[f64],
    posterior_mean_rows: &[Vec<f64>],
) -> Vec<EtaPosteriorMeanShrinkage> {
    if let Err(reason) = validate_widths(effect_names, posterior_mean_rows, final_var_diag) {
        return effect_names
            .iter()
            .map(|effect| EtaPosteriorMeanShrinkage {
                effect: effect.clone(),
                shrinkage: ShrinkageValue::Unavailable {
                    reason: reason.clone(),
                },
            })
            .collect();
    }

    let unit_count = posterior_mean_rows.len();
    effect_names
        .iter()
        .enumerate()
        .map(|(i, effect)| {
            let col = extract_column(posterior_mean_rows, i);
            let shrinkage = match compute_shrinkage(&col, final_var_diag[i], unit_count) {
                Ok(value) => ShrinkageValue::Available {
                    value,
                    unit_count,
                    denominator_documentation: ETA_DENOMINATOR.to_owned(),
                },
                Err(reason) => ShrinkageValue::Unavailable { reason },
            };
            EtaPosteriorMeanShrinkage {
                effect: effect.clone(),
                shrinkage,
            }
        })
        .collect()
}

/// Derive eta (η) shrinkage for each random effect from subject MAP estimates.
///
/// # Parameters
///
/// - `effect_names` — one name per random effect, in order.
/// - `final_var_diag` — diagonal of the final Ω covariance matrix, one entry
///   per effect.
/// - `map_rows` — one row per subject, one column per effect.
///   Each row is the MAP η vector for that subject.
///
/// # Denominator
///
/// `unit_count` = number of rows = number of subjects.
pub(crate) fn derive_eta_map_shrinkage(
    effect_names: &[String],
    final_var_diag: &[f64],
    map_rows: Option<&[Vec<f64>]>,
) -> Vec<EtaMapShrinkage> {
    let map_rows = match map_rows {
        Some(rows) => rows,
        None => {
            return effect_names
                .iter()
                .map(|effect| EtaMapShrinkage {
                    effect: effect.clone(),
                    shrinkage: ShrinkageValue::Unavailable {
                        reason: ShrinkageUnavailableReason::MissingMap,
                    },
                })
                .collect();
        }
    };

    if let Err(reason) = validate_widths(effect_names, map_rows, final_var_diag) {
        return effect_names
            .iter()
            .map(|effect| EtaMapShrinkage {
                effect: effect.clone(),
                shrinkage: ShrinkageValue::Unavailable {
                    reason: reason.clone(),
                },
            })
            .collect();
    }

    let unit_count = map_rows.len();
    effect_names
        .iter()
        .enumerate()
        .map(|(i, effect)| {
            let col = extract_column(map_rows, i);
            let shrinkage = match compute_shrinkage(&col, final_var_diag[i], unit_count) {
                Ok(value) => ShrinkageValue::Available {
                    value,
                    unit_count,
                    denominator_documentation: ETA_DENOMINATOR.to_owned(),
                },
                Err(reason) => ShrinkageValue::Unavailable { reason },
            };
            EtaMapShrinkage {
                effect: effect.clone(),
                shrinkage,
            }
        })
        .collect()
}

/// Derive kappa (κ) shrinkage for each inter-occasion random effect from
/// occasion posterior means.
///
/// # Parameters
///
/// - `effect_names` — one name per inter-occasion random effect, in order.
/// - `final_var_diag` — diagonal of the final inter-occasion covariance
///   matrix, one entry per effect.
/// - `posterior_mean_rows` — one row per subject-occasion pair, one column
///   per effect. Rows from all subjects are pooled together.
///
/// # Denominator
///
/// `unit_count` = number of rows = number of pooled subject-occasion pairs.
pub(crate) fn derive_kappa_posterior_mean_shrinkage(
    effect_names: &[String],
    final_var_diag: &[f64],
    posterior_mean_rows: &[Vec<f64>],
) -> Vec<KappaPosteriorMeanShrinkage> {
    if let Err(reason) = validate_widths(effect_names, posterior_mean_rows, final_var_diag) {
        return effect_names
            .iter()
            .map(|effect| KappaPosteriorMeanShrinkage {
                effect: effect.clone(),
                shrinkage: ShrinkageValue::Unavailable {
                    reason: reason.clone(),
                },
            })
            .collect();
    }

    let unit_count = posterior_mean_rows.len();
    effect_names
        .iter()
        .enumerate()
        .map(|(i, effect)| {
            let col = extract_column(posterior_mean_rows, i);
            let shrinkage = match compute_shrinkage(&col, final_var_diag[i], unit_count) {
                Ok(value) => ShrinkageValue::Available {
                    value,
                    unit_count,
                    denominator_documentation: KAPPA_DENOMINATOR.to_owned(),
                },
                Err(reason) => ShrinkageValue::Unavailable { reason },
            };
            KappaPosteriorMeanShrinkage {
                effect: effect.clone(),
                shrinkage,
            }
        })
        .collect()
}

/// Derive kappa (κ) shrinkage for each inter-occasion random effect from
/// occasion MAP estimates.
///
/// # Parameters
///
/// - `effect_names` — one name per inter-occasion random effect, in order.
/// - `final_var_diag` — diagonal of the final inter-occasion covariance
///   matrix, one entry per effect.
/// - `map_rows` — one row per subject-occasion pair, one column per effect.
///   Rows from all subjects are pooled together.
///
/// # Denominator
///
/// `unit_count` = number of rows = number of pooled subject-occasion pairs.
pub(crate) fn derive_kappa_map_shrinkage(
    effect_names: &[String],
    final_var_diag: &[f64],
    map_rows: Option<&[Vec<f64>]>,
) -> Vec<KappaMapShrinkage> {
    let map_rows = match map_rows {
        Some(rows) => rows,
        None => {
            return effect_names
                .iter()
                .map(|effect| KappaMapShrinkage {
                    effect: effect.clone(),
                    shrinkage: ShrinkageValue::Unavailable {
                        reason: ShrinkageUnavailableReason::MissingMap,
                    },
                })
                .collect();
        }
    };

    if let Err(reason) = validate_widths(effect_names, map_rows, final_var_diag) {
        return effect_names
            .iter()
            .map(|effect| KappaMapShrinkage {
                effect: effect.clone(),
                shrinkage: ShrinkageValue::Unavailable {
                    reason: reason.clone(),
                },
            })
            .collect();
    }

    let unit_count = map_rows.len();
    effect_names
        .iter()
        .enumerate()
        .map(|(i, effect)| {
            let col = extract_column(map_rows, i);
            let shrinkage = match compute_shrinkage(&col, final_var_diag[i], unit_count) {
                Ok(value) => ShrinkageValue::Available {
                    value,
                    unit_count,
                    denominator_documentation: KAPPA_DENOMINATOR.to_owned(),
                },
                Err(reason) => ShrinkageValue::Unavailable { reason },
            };
            KappaMapShrinkage {
                effect: effect.clone(),
                shrinkage,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parametric::shrinkage::ShrinkageUnavailableReason::{
        MissingMap, NonFiniteValue, NonPositiveReferenceVariance, TooFewUnits, WidthMismatch,
    };

    // ── Helpers ──

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn unpack_available(value: &ShrinkageValue) -> (f64, usize, &str) {
        match value {
            ShrinkageValue::Available {
                value,
                unit_count,
                denominator_documentation,
            } => (*value, *unit_count, denominator_documentation.as_str()),
            ShrinkageValue::Unavailable { reason } => {
                panic!("expected Available, got Unavailable: {reason:?}")
            }
        }
    }

    fn unpack_unavailable(value: &ShrinkageValue) -> &ShrinkageUnavailableReason {
        match value {
            ShrinkageValue::Unavailable { reason } => reason,
            ShrinkageValue::Available { .. } => panic!("expected Unavailable, got Available"),
        }
    }

    // ── Formula correctness ──

    #[test]
    fn shrinkage_zero_when_sample_variance_equals_final_variance() {
        // Two subjects with η values [-1, 1]. Sample variance:
        // mean = 0, var = ((-1)^2 + 1^2) / (2-1) = 2/1 = 2.
        // final_variance = 2 → shrinkage = 100*(1 - 2/2) = 0.
        let rows = vec![vec![-1.0], vec![1.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[2.0], &rows);
        let (val, count, doc) = unpack_available(&results[0].shrinkage);
        assert_eq!(results[0].effect, "CL");
        assert!((val - 0.0).abs() < 1e-12);
        assert_eq!(count, 2);
        assert_eq!(doc, "number of subjects");
    }

    #[test]
    fn shrinkage_fifty_when_sample_variance_is_half_of_final() {
        // Three subjects with values [0, 2, 4]. mean = 2.
        // sample var: ((0-2)^2 + (2-2)^2 + (4-2)^2) / 2 = (4 + 0 + 4)/2 = 4.
        // final_variance = 8 → shrinkage = 100*(1 - 4/8) = 50.
        let rows = vec![vec![0.0], vec![2.0], vec![4.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["V"]), &[8.0], &rows);
        let (val, _, _) = unpack_available(&results[0].shrinkage);
        assert!((val - 50.0).abs() < 1e-12);
    }

    #[test]
    fn shrinkage_one_hundred_when_all_posterior_means_identical() {
        // All identical → sample variance = 0 → shrinkage = 100%.
        let rows = vec![vec![3.0], vec![3.0], vec![3.0], vec![3.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["KA"]), &[2.0], &rows);
        let (val, count, _) = unpack_available(&results[0].shrinkage);
        assert!((val - 100.0).abs() < 1e-12);
        assert_eq!(count, 4);
    }

    #[test]
    fn negative_shrinkage_preserved_when_sample_variance_exceeds_final_variance() {
        // Two subjects: values [-3, 3]. mean = 0.
        // sample var: ((-3)^2 + 3^2) / 1 = 18.
        // final_variance = 2 → shrinkage = 100*(1 - 18/2) = 100*(1 - 9) = -800.
        let rows = vec![vec![-3.0], vec![3.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[2.0], &rows);
        let (val, _, _) = unpack_available(&results[0].shrinkage);
        assert!((val - (-800.0)).abs() < 1e-12);
        assert!(val < 0.0, "negative shrinkage must be preserved, got {val}");
    }

    #[test]
    fn posterior_mean_differs_from_map() {
        // Two effects, three subjects.
        // Posterior means and MAP differ → shrinkage values must differ.
        let effect_names = names(&["CL", "V"]);
        let final_var_diag = vec![4.0, 9.0];

        // Posterior mean rows
        let post_mean_rows = vec![
            vec![1.0, 3.0], // subject 1
            vec![2.0, 6.0], // subject 2
            vec![3.0, 9.0], // subject 3
        ];
        // MAP rows (different values)
        let map_rows = vec![
            vec![0.5, 1.0],  // subject 1
            vec![2.5, 5.0],  // subject 2
            vec![4.0, 14.0], // subject 3
        ];

        let post_results =
            derive_eta_posterior_mean_shrinkage(&effect_names, &final_var_diag, &post_mean_rows);
        let map_results = derive_eta_map_shrinkage(&effect_names, &final_var_diag, Some(&map_rows));

        let (post_cl, _, _) = unpack_available(&post_results[0].shrinkage);
        let (map_cl, _, _) = unpack_available(&map_results[0].shrinkage);
        let (post_v, _, _) = unpack_available(&post_results[1].shrinkage);
        let (map_v, _, _) = unpack_available(&map_results[1].shrinkage);

        // Both effects should differ between posterior mean and MAP.
        assert!(
            (post_cl - map_cl).abs() > 1e-12,
            "CL shrinkage should differ: posterior_mean={post_cl}, map={map_cl}"
        );
        assert!(
            (post_v - map_v).abs() > 1e-12,
            "V shrinkage should differ: posterior_mean={post_v}, map={map_v}"
        );
    }

    #[test]
    fn kappa_pooled_units_use_correct_denominator() {
        // 2 subjects × 3 occasions = 6 pooled rows.
        let rows = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0], // subject 1 occasions
            vec![3.0],
            vec![4.0],
            vec![5.0], // subject 2 occasions
        ];
        let results = derive_kappa_posterior_mean_shrinkage(&names(&["IOV_CL"]), &[10.0], &rows);
        let (val, count, doc) = unpack_available(&results[0].shrinkage);
        assert_eq!(count, 6);
        assert_eq!(doc, "number of pooled subject-occasion pairs");
        // mean = 2.5, sample var = ((0-2.5)^2 + ... + (5-2.5)^2) / 5
        // = (6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25) / 5 = 17.5/5 = 3.5
        // shrinkage = 100*(1 - 3.5/10) = 65
        assert!((val - 65.0).abs() < 1e-12);
    }

    // ── Multiple effects ──

    #[test]
    fn multiple_effects_each_have_independent_shrinkage() {
        let effect_names = names(&["CL", "V", "KA"]);
        let final_var_diag = vec![4.0, 16.0, 1.0];
        // 4 subjects
        let rows = vec![
            vec![1.0, 2.0, 0.5],
            vec![3.0, 6.0, 1.5],
            vec![1.0, 2.0, 0.5],
            vec![3.0, 6.0, 1.5],
        ];
        let results = derive_eta_posterior_mean_shrinkage(&effect_names, &final_var_diag, &rows);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].effect, "CL");
        assert_eq!(results[1].effect, "V");
        assert_eq!(results[2].effect, "KA");

        // CL: post means [1,3,1,3], mean=2, sample var = (1+1+1+1)/3 = 4/3 ≈ 1.333
        // shrinkage = 100*(1 - 1.333/4) = 100*(1 - 0.3333) ≈ 66.667
        let (cl_val, _, _) = unpack_available(&results[0].shrinkage);
        assert!((cl_val - 100.0 * (1.0 - 4.0 / 3.0 / 4.0)).abs() < 1e-12);

        // V: squared deviations sum to 16, so sample variance = 16/3.
        let (v_val, _, _) = unpack_available(&results[1].shrinkage);
        assert!((v_val - 100.0 * (1.0 - 16.0 / 3.0 / 16.0)).abs() < 1e-12);

        // KA: squared deviations sum to 1, so sample variance = 1/3.
        let (ka_val, _, _) = unpack_available(&results[2].shrinkage);
        assert!((ka_val - 100.0 * (1.0 - 1.0 / 3.0 / 1.0)).abs() < 1e-12);
    }

    // ── Unavailable: TooFewUnits ──

    #[test]
    fn single_subject_yields_too_few_units() {
        let rows = vec![vec![5.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[2.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &TooFewUnits { count: 1 });
    }

    #[test]
    fn zero_subjects_yields_too_few_units() {
        let rows: Vec<Vec<f64>> = vec![];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[2.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &TooFewUnits { count: 0 });
    }

    // ── Unavailable: MissingMap ──

    #[test]
    fn missing_map_yields_missing_map_reason() {
        let results = derive_eta_map_shrinkage(&names(&["CL"]), &[2.0], None);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &MissingMap);
    }

    #[test]
    fn missing_kappa_map_yields_missing_map_reason() {
        let results = derive_kappa_map_shrinkage(&names(&["IOV_V"]), &[3.0], None);
        assert_eq!(results.len(), 1);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &MissingMap);
    }

    // ── Unavailable: NonFiniteValue ──

    #[test]
    fn nan_in_posterior_mean_yields_non_finite() {
        let rows = vec![vec![f64::NAN], vec![1.0], vec![2.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[4.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &NonFiniteValue);
    }

    #[test]
    fn infinity_in_posterior_mean_yields_non_finite() {
        let rows = vec![vec![1.0], vec![f64::INFINITY], vec![3.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[4.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &NonFiniteValue);
    }

    #[test]
    fn neg_infinity_in_posterior_mean_yields_non_finite() {
        let rows = vec![vec![1.0], vec![f64::NEG_INFINITY], vec![3.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[4.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &NonFiniteValue);
    }

    // ── Unavailable: NonPositiveReferenceVariance ──

    #[test]
    fn zero_final_variance_yields_non_positive_reference() {
        let rows = vec![vec![1.0], vec![2.0], vec![3.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[0.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &NonPositiveReferenceVariance { variance: 0.0 });
    }

    #[test]
    fn negative_final_variance_yields_non_positive_reference() {
        let rows = vec![vec![1.0], vec![2.0], vec![3.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[-1.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &NonPositiveReferenceVariance { variance: -1.0 });
    }

    #[test]
    fn nan_final_variance_yields_non_positive_reference() {
        let rows = vec![vec![1.0], vec![2.0], vec![3.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[f64::NAN], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert!(matches!(reason, NonPositiveReferenceVariance { .. }));
    }

    // ── Unavailable: WidthMismatch ──

    #[test]
    fn effect_names_fewer_than_row_columns_yields_width_mismatch() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[4.0, 9.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(
            reason,
            &WidthMismatch {
                effect_names: 1,
                row_width: 2,
                variance_len: 2,
            }
        );
    }

    #[test]
    fn variance_diag_shorter_than_effects_yields_width_mismatch() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL", "V"]), &[4.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(
            reason,
            &WidthMismatch {
                effect_names: 2,
                row_width: 2,
                variance_len: 1,
            }
        );
    }

    #[test]
    fn jagged_rows_yield_width_mismatch() {
        let rows = vec![vec![1.0], vec![2.0, 3.0], vec![4.0]];
        let results = derive_kappa_posterior_mean_shrinkage(&names(&["IOV_CL"]), &[5.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert!(matches!(reason, WidthMismatch { .. }));
    }

    #[test]
    fn empty_rows_with_effects_yields_too_few_units() {
        let rows: Vec<Vec<f64>> = vec![];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL", "V"]), &[4.0, 9.0], &rows);
        let reason = unpack_unavailable(&results[0].shrinkage);
        assert_eq!(reason, &TooFewUnits { count: 0 });
    }

    // ── Kappa-specific: pooled subject-occasion pairs ──

    #[test]
    fn kappa_map_differs_from_posterior_mean() {
        let effect_names = names(&["IOV_CL", "IOV_V"]);
        let final_var_diag = vec![4.0, 9.0];

        // 2 subjects × 3 occasions = 6 pooled rows
        let post_mean_rows = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
            vec![5.0, 10.0],
            vec![6.0, 12.0],
        ];
        let map_rows = vec![
            vec![0.0, 1.0],
            vec![1.5, 3.0],
            vec![3.5, 7.0],
            vec![4.5, 9.0],
            vec![4.0, 8.0],
            vec![7.0, 14.0],
        ];

        let post_results =
            derive_kappa_posterior_mean_shrinkage(&effect_names, &final_var_diag, &post_mean_rows);
        let map_results =
            derive_kappa_map_shrinkage(&effect_names, &final_var_diag, Some(&map_rows));

        let (post_cl, post_count, post_doc) = unpack_available(&post_results[0].shrinkage);
        let (map_cl, map_count, map_doc) = unpack_available(&map_results[0].shrinkage);

        assert_eq!(post_count, 6);
        assert_eq!(map_count, 6);
        assert_eq!(post_doc, "number of pooled subject-occasion pairs");
        assert_eq!(map_doc, "number of pooled subject-occasion pairs");
        assert!(
            (post_cl - map_cl).abs() > 1e-12,
            "kappa CL shrinkage should differ: posterior_mean={post_cl}, map={map_cl}"
        );
    }

    // ── MAP paths with valid data work identically to posterior-mean paths ──

    #[test]
    fn eta_map_with_valid_rows_computes_correctly() {
        let rows = vec![vec![0.0], vec![2.0], vec![4.0]];
        let results = derive_eta_map_shrinkage(&names(&["V"]), &[8.0], Some(&rows));
        let (val, count, doc) = unpack_available(&results[0].shrinkage);
        assert!((val - 50.0).abs() < 1e-12);
        assert_eq!(count, 3);
        assert_eq!(doc, "number of subjects");
    }

    #[test]
    fn kappa_map_with_valid_rows_computes_correctly() {
        let rows = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let results = derive_kappa_map_shrinkage(&names(&["IOV_CL"]), &[5.0], Some(&rows));
        let (val, count, doc) = unpack_available(&results[0].shrinkage);
        // mean = 2.5, sample var = (2.25+0.25+0.25+2.25)/3 = 5/3 ≈ 1.667
        // shrinkage = 100*(1 - 1.667/5) ≈ 66.667
        assert!((val - 100.0 * (1.0 - 5.0 / 3.0 / 5.0)).abs() < 1e-12);
        assert_eq!(count, 4);
        assert_eq!(doc, "number of pooled subject-occasion pairs");
    }

    // ── Serde round-trip ──

    #[test]
    fn serde_round_trip_available() {
        let value = ShrinkageValue::Available {
            value: 42.5,
            unit_count: 10,
            denominator_documentation: "number of subjects".to_owned(),
        };
        let json = serde_json::to_string(&value).unwrap();
        let round_tripped: ShrinkageValue = serde_json::from_str(&json).unwrap();
        assert_eq!(value, round_tripped);
    }

    #[test]
    fn serde_round_trip_unavailable() {
        let value = ShrinkageValue::Unavailable {
            reason: ShrinkageUnavailableReason::TooFewUnits { count: 1 },
        };
        let json = serde_json::to_string(&value).unwrap();
        let round_tripped: ShrinkageValue = serde_json::from_str(&json).unwrap();
        assert_eq!(value, round_tripped);
    }

    #[test]
    fn serde_round_trip_missing_map() {
        let value = ShrinkageValue::Unavailable { reason: MissingMap };
        let json = serde_json::to_string(&value).unwrap();
        let round_tripped: ShrinkageValue = serde_json::from_str(&json).unwrap();
        assert_eq!(value, round_tripped);
    }

    #[test]
    fn serde_json_structure_is_stable() {
        let effect = EtaPosteriorMeanShrinkage {
            effect: "CL".to_owned(),
            shrinkage: ShrinkageValue::Available {
                value: 25.0,
                unit_count: 3,
                denominator_documentation: "number of subjects".to_owned(),
            },
        };
        let json = serde_json::to_string_pretty(&effect).unwrap();
        // Verify key fields are present with expected values.
        assert!(json.contains("\"effect\": \"CL\""));
        assert!(json.contains("\"status\": \"available\""));
        assert!(json.contains("\"value\": 25.0"));
        assert!(json.contains("\"unit_count\": 3"));
        assert!(json.contains("\"denominator_documentation\": \"number of subjects\""));
    }

    // ── No clamping: edge cases ──

    #[test]
    fn near_zero_sample_variance_produces_near_100_shrinkage_no_clamp() {
        // Very tiny but non-zero sample variance → shrinkage < 100 but very close.
        // Two values almost identical.
        let rows = vec![vec![5.0], vec![5.0 + 1e-6]];
        let results = derive_eta_posterior_mean_shrinkage(&names(&["CL"]), &[1.0], &rows);
        let (val, _, _) = unpack_available(&results[0].shrinkage);
        // The nonzero variance is small enough to be near 100%, but large
        // enough that the subtraction remains representable in binary64.
        assert!(val < 100.0, "should be slightly below 100, got {val}");
        assert!(val > 99.999, "should be very close to 100, got {val}");
    }

    #[test]
    fn negative_shrinkage_round_trips_through_serde() {
        let value = ShrinkageValue::Available {
            value: -150.0,
            unit_count: 5,
            denominator_documentation: "number of subjects".to_owned(),
        };
        let json = serde_json::to_string(&value).unwrap();
        assert!(json.contains("-150.0"));
        let round_tripped: ShrinkageValue = serde_json::from_str(&json).unwrap();
        assert_eq!(value, round_tripped);
    }
}
