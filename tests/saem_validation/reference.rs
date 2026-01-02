//! Reference data structures for loading R saemix results
//!
//! These structures match the JSON output from generate_reference.R

use serde::Deserialize;
use std::path::Path;

/// Reference results from R saemix for a single test case
#[derive(Debug, Deserialize)]
pub struct SaemixReference {
    pub test_case: String,
    pub description: String,

    /// True parameter values (if synthetic data)
    #[serde(default)]
    pub true_values: Option<TrueValues>,

    /// Population mean in φ (transformed/unconstrained) space
    pub mu_phi: Vec<f64>,

    /// Population mean in ψ (original/constrained) space
    pub mu_psi: Vec<f64>,

    /// Full covariance matrix Ω
    pub omega: Vec<Vec<f64>>,

    /// Diagonal of Ω (variances)
    pub omega_diag: Vec<f64>,

    /// Residual error standard deviation
    pub sigma: f64,

    /// Log-likelihood (linearization approximation)
    pub ll_lin: f64,

    /// Objective function (-2LL)
    pub objf: f64,

    /// MAP individual parameter estimates (n_subjects × n_params)
    pub map_psi: Vec<Vec<f64>>,

    /// MAP random effects (n_subjects × n_params)
    pub map_eta: Vec<Vec<f64>>,

    /// Conditional mean in φ space
    pub cond_mean_phi: Vec<Vec<f64>>,

    /// Settings used for this run
    pub settings: SaemixSettings,

    /// Number of subjects
    pub n_subjects: usize,

    /// Total number of observations
    pub n_observations: usize,
}

#[derive(Debug, Deserialize, Default)]
pub struct TrueValues {
    #[serde(default)]
    pub ke: Option<f64>,
    #[serde(default)]
    pub v: Option<f64>,
    #[serde(default)]
    pub ka: Option<f64>,
    #[serde(default)]
    pub cl: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct SaemixSettings {
    pub seed: u64,
    pub n_burn: usize,
    pub n_sa: usize,
    pub n_smooth: usize,
    pub n_chains: usize,
    pub transform_par: Vec<u8>,
    pub error_model: String,
    pub initial_psi: Vec<f64>,
    pub initial_omega_diag: Vec<f64>,
}

/// Component-level reference values for exact matching
#[derive(Debug, Deserialize)]
pub struct ComponentReference {
    pub transforms: TransformReference,
    pub sufficient_stats: SufficientStatsReference,
    pub step_size: StepSizeReference,
}

#[derive(Debug, Deserialize)]
pub struct TransformReference {
    pub log_normal: LogNormalTransform,
}

#[derive(Debug, Deserialize)]
pub struct LogNormalTransform {
    pub psi: Vec<f64>,
    pub phi: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct SufficientStatsReference {
    pub phi_samples: Vec<Vec<f64>>,
    pub s1: Vec<f64>,
    pub s2: Vec<Vec<f64>>,
    pub mu: Vec<f64>,
    pub omega: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct StepSizeReference {
    pub n_burn: usize,
    pub n_smooth: usize,
    pub schedule: Vec<f64>,
}

/// Load reference from JSON file
pub fn load_reference<P: AsRef<Path>>(path: P) -> Result<SaemixReference, String> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| format!("Failed to read file: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))
}

/// Load component reference from JSON file
pub fn load_component_reference<P: AsRef<Path>>(path: P) -> Result<ComponentReference, String> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| format!("Failed to read file: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))
}

/// Assertion helper: check values are close within relative tolerance
pub fn assert_close(actual: f64, expected: f64, rtol: f64, name: &str) {
    let abs_expected = expected.abs().max(1e-10);
    let rel_error = (actual - expected).abs() / abs_expected;
    assert!(
        rel_error < rtol,
        "{}: actual={:.6}, expected={:.6}, rel_error={:.4} (tolerance={:.4})",
        name,
        actual,
        expected,
        rel_error,
        rtol
    );
}

/// Assertion helper: check vectors are close
pub fn assert_vec_close(actual: &[f64], expected: &[f64], rtol: f64, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", name);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(*a, *e, rtol, &format!("{}[{}]", name, i));
    }
}

/// Assertion helper: check matrices are close (flattened comparison)
pub fn assert_matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>], rtol: f64, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: row count mismatch", name);
    for (i, (row_a, row_e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            row_a.len(),
            row_e.len(),
            "{}: row {} length mismatch",
            name,
            i
        );
        for (j, (a, e)) in row_a.iter().zip(row_e.iter()).enumerate() {
            assert_close(*a, *e, rtol, &format!("{}[{},{}]", name, i, j));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_close() {
        // Should pass
        assert_close(1.0, 1.0, 0.01, "exact");
        assert_close(1.01, 1.0, 0.02, "within 2%");

        // Should fail - commented out to not break tests
        // assert_close(1.1, 1.0, 0.05, "beyond 5%");
    }

    #[test]
    fn test_assert_vec_close() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.01, 2.02, 3.03];
        assert_vec_close(&a, &b, 0.02, "test_vec");
    }
}
