//! Covariate model for parametric population modeling
//!
//! This module implements the covariate model that defines how subject-level
//! covariates (like weight, age, sex) affect population parameters.
//!
//! # Mathematical Background
//!
//! In parametric mixed-effects models, the individual parameter for subject i is:
//!
//! ```text
//! φᵢ = μᵢ + ηᵢ
//! ```
//!
//! where:
//! - `μᵢ`: Subject-specific population mean (depends on covariates)
//! - `ηᵢ ~ N(0, Ω)`: Random effects
//!
//! The subject-specific population mean is computed as:
//!
//! ```text
//! μᵢ = Xᵢ × β
//! ```
//!
//! where:
//! - `Xᵢ`: Design matrix row for subject i (includes intercept and covariate values)
//! - `β`: Fixed effect coefficients (intercepts + covariate effects)
//!
//! # R saemix Correspondence
//!
//! This corresponds to R saemix's:
//! - `covariate.model`: Matrix indicating which parameters depend on which covariates
//! - `betaest.model`: Which β coefficients to estimate
//! - Design matrix construction in `model_pred.R`
//!
//! # Example
//!
//! A model with 2 parameters (CL, V) and 2 covariates (WT, SEX):
//!
//! ```text
//! covariate_mask:
//!        WT   SEX
//!   CL   1    0     <- CL depends on WT only
//!   V    1    1     <- V depends on both WT and SEX
//!
//! β = [β_CL, β_CL_WT, β_V, β_V_WT, β_V_SEX]
//!
//! For subject i with WT=70, SEX=1:
//! μ_CL,i = β_CL + β_CL_WT × 70
//! μ_V,i  = β_V + β_V_WT × 70 + β_V_SEX × 1
//! ```

use anyhow::{bail, Result};
use faer::{Col, Mat};
use serde::Serialize;
use std::collections::HashMap;

/// Covariate model specification
///
/// Defines how covariates affect population parameters through a design matrix formulation.
#[derive(Debug, Clone)]
pub struct CovariateModel {
    /// Names of the model parameters (e.g., ["CL", "V", "KA"])
    param_names: Vec<String>,

    /// Names of the covariates in order (e.g., ["WT", "SEX", "AGE"])
    covariate_names: Vec<String>,

    /// Covariate influence mask: param_mask[i][j] = true if parameter i depends on covariate j
    /// Dimensions: n_params × n_covariates
    covariate_mask: Vec<Vec<bool>>,

    /// Fixed effect coefficients (β)
    /// Order: for each parameter, [intercept, cov1_effect, cov2_effect, ...] if mask is true
    beta: Col<f64>,

    /// Which β coefficients to estimate (vs hold fixed)
    /// Same length as beta
    estimate_beta: Vec<bool>,

    /// Reference values for covariates (for centering)
    /// Optional: if provided, covariates are centered before applying effects
    reference_values: HashMap<String, f64>,
}

impl CovariateModel {
    /// Create a new covariate model
    ///
    /// # Arguments
    /// * `param_names` - Names of the model parameters
    /// * `covariate_names` - Names of the covariates
    /// * `covariate_mask` - Boolean matrix indicating parameter-covariate dependencies
    ///
    /// # Example
    /// ```ignore
    /// let model = CovariateModel::new(
    ///     vec!["CL", "V"],
    ///     vec!["WT", "SEX"],
    ///     vec![
    ///         vec![true, false],  // CL depends on WT
    ///         vec![true, true],   // V depends on WT and SEX
    ///     ],
    /// )?;
    /// ```
    pub fn new(
        param_names: Vec<impl Into<String>>,
        covariate_names: Vec<impl Into<String>>,
        covariate_mask: Vec<Vec<bool>>,
    ) -> Result<Self> {
        let param_names: Vec<String> = param_names.into_iter().map(|s| s.into()).collect();
        let covariate_names: Vec<String> = covariate_names.into_iter().map(|s| s.into()).collect();

        let n_params = param_names.len();
        let n_covs = covariate_names.len();

        // Validate mask dimensions
        if covariate_mask.len() != n_params {
            bail!(
                "Covariate mask rows ({}) must match number of parameters ({})",
                covariate_mask.len(),
                n_params
            );
        }

        for (i, row) in covariate_mask.iter().enumerate() {
            if row.len() != n_covs {
                bail!(
                    "Covariate mask row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    n_covs
                );
            }
        }

        // Calculate total number of beta coefficients
        // For each parameter: 1 (intercept) + sum of covariate effects
        let n_beta = Self::count_beta_coefficients(&covariate_mask, n_params);

        let beta = Col::zeros(n_beta);
        let estimate_beta = vec![true; n_beta];

        Ok(Self {
            param_names,
            covariate_names,
            covariate_mask,
            beta,
            estimate_beta,
            reference_values: HashMap::new(),
        })
    }

    /// Create a model with no covariates (intercept only)
    ///
    /// This is equivalent to having all parameters independent of covariates.
    pub fn intercept_only(param_names: Vec<impl Into<String>>) -> Result<Self> {
        let param_names: Vec<String> = param_names.into_iter().map(|s| s.into()).collect();
        let n_params = param_names.len();

        Ok(Self {
            param_names,
            covariate_names: Vec::new(),
            covariate_mask: vec![Vec::new(); n_params],
            beta: Col::zeros(n_params), // Just intercepts
            estimate_beta: vec![true; n_params],
            reference_values: HashMap::new(),
        })
    }

    /// Create from R saemix-style covariate.model matrix
    ///
    /// # Arguments
    /// * `param_names` - Parameter names
    /// * `covariate_names` - Covariate names  
    /// * `matrix` - Flattened covariate model matrix (row-major, 0/1 values)
    pub fn from_saemix_matrix(
        param_names: Vec<impl Into<String>>,
        covariate_names: Vec<impl Into<String>>,
        matrix: &[f64],
    ) -> Result<Self> {
        let param_names: Vec<String> = param_names.into_iter().map(|s| s.into()).collect();
        let covariate_names: Vec<String> = covariate_names.into_iter().map(|s| s.into()).collect();

        let n_params = param_names.len();
        let n_covs = covariate_names.len();

        if matrix.len() != n_params * n_covs {
            bail!(
                "Matrix length ({}) doesn't match n_params × n_covs ({} × {} = {})",
                matrix.len(),
                n_params,
                n_covs,
                n_params * n_covs
            );
        }

        let covariate_mask: Vec<Vec<bool>> = (0..n_params)
            .map(|i| (0..n_covs).map(|j| matrix[i * n_covs + j] != 0.0).collect())
            .collect();

        Self::new(param_names, covariate_names, covariate_mask)
    }

    // ========== Configuration Methods ==========

    /// Set the fixed effect coefficients (β)
    pub fn set_beta(&mut self, beta: Col<f64>) -> Result<()> {
        let expected = self.n_beta();
        if beta.nrows() != expected {
            bail!(
                "Beta length ({}) doesn't match expected ({})",
                beta.nrows(),
                expected
            );
        }
        self.beta = beta;
        Ok(())
    }

    /// Set initial beta values from parameter intercepts
    ///
    /// This sets the intercept term for each parameter, leaving covariate effects at zero.
    pub fn set_intercepts(&mut self, intercepts: &[f64]) -> Result<()> {
        if intercepts.len() != self.n_params() {
            bail!(
                "Intercepts length ({}) doesn't match n_params ({})",
                intercepts.len(),
                self.n_params()
            );
        }

        let mut idx = 0;
        for (i, &intercept) in intercepts.iter().enumerate() {
            self.beta[idx] = intercept;
            idx += 1;

            // Skip covariate effect slots (leave at 0)
            idx += self.covariate_mask[i].iter().filter(|&&x| x).count();
        }

        Ok(())
    }

    /// Set which beta coefficients to estimate
    pub fn set_estimate_beta(&mut self, estimate: Vec<bool>) -> Result<()> {
        if estimate.len() != self.beta.nrows() {
            bail!(
                "estimate_beta length ({}) doesn't match n_beta ({})",
                estimate.len(),
                self.beta.nrows()
            );
        }
        self.estimate_beta = estimate;
        Ok(())
    }

    /// Fix an intercept (don't estimate it)
    pub fn fix_intercept(&mut self, param_idx: usize) -> Result<()> {
        let beta_idx = self.intercept_beta_index(param_idx)?;
        self.estimate_beta[beta_idx] = false;
        Ok(())
    }

    /// Set reference values for centering covariates
    pub fn set_reference(&mut self, covariate: &str, value: f64) -> Result<()> {
        if !self.covariate_names.contains(&covariate.to_string()) {
            bail!("Unknown covariate: {}", covariate);
        }
        self.reference_values.insert(covariate.to_string(), value);
        Ok(())
    }

    // ========== Computation Methods ==========

    /// Compute subject-specific population mean μᵢ given covariate values
    ///
    /// # Arguments
    /// * `covariates` - Map of covariate name → value for this subject
    ///
    /// # Returns
    /// * Vector of population means for each parameter
    pub fn compute_mu(&self, covariates: &HashMap<String, f64>) -> Col<f64> {
        let n_params = self.param_names.len();
        let mut mu = Col::zeros(n_params);

        let mut beta_idx = 0;

        for i in 0..n_params {
            // Start with intercept
            mu[i] = self.beta[beta_idx];
            beta_idx += 1;

            // Add covariate effects
            for (j, cov_name) in self.covariate_names.iter().enumerate() {
                if self.covariate_mask[i][j] {
                    let cov_value = covariates.get(cov_name).copied().unwrap_or(0.0);
                    let reference = self.reference_values.get(cov_name).copied().unwrap_or(0.0);
                    let centered_value = cov_value - reference;

                    mu[i] += self.beta[beta_idx] * centered_value;
                    beta_idx += 1;
                }
            }
        }

        mu
    }

    /// Build the design matrix row for a subject
    ///
    /// Returns a row vector X such that μ = X × β
    pub fn build_design_row(&self, covariates: &HashMap<String, f64>) -> Col<f64> {
        let n_beta = self.n_beta();
        let mut x = Col::zeros(n_beta);

        let mut beta_idx = 0;

        for i in 0..self.n_params() {
            // Intercept column
            x[beta_idx] = 1.0;
            beta_idx += 1;

            // Covariate columns
            for (j, cov_name) in self.covariate_names.iter().enumerate() {
                if self.covariate_mask[i][j] {
                    let cov_value = covariates.get(cov_name).copied().unwrap_or(0.0);
                    let reference = self.reference_values.get(cov_name).copied().unwrap_or(0.0);
                    x[beta_idx] = cov_value - reference;
                    beta_idx += 1;
                }
            }
        }

        x
    }

    /// Build the full design matrix for multiple subjects
    ///
    /// # Arguments
    /// * `all_covariates` - Vector of covariate maps, one per subject
    ///
    /// # Returns
    /// * Design matrix with shape (n_subjects × n_params, n_beta)
    pub fn build_design_matrix(&self, all_covariates: &[HashMap<String, f64>]) -> Mat<f64> {
        let n_subjects = all_covariates.len();
        let n_params = self.n_params();
        let n_beta = self.n_beta();

        // Each subject contributes n_params rows (one per parameter)
        let n_rows = n_subjects * n_params;
        let mut x = Mat::zeros(n_rows, n_beta);

        for (subj_idx, covs) in all_covariates.iter().enumerate() {
            let _row = self.build_design_row(covs);

            // This subject's rows start at subj_idx * n_params
            // But we need to extract the relevant portion for each parameter
            let mut beta_idx = 0;
            for param_idx in 0..n_params {
                let row_idx = subj_idx * n_params + param_idx;

                // Intercept
                x[(row_idx, beta_idx)] = 1.0;
                beta_idx += 1;

                // Covariate effects for this parameter
                for (j, cov_name) in self.covariate_names.iter().enumerate() {
                    if self.covariate_mask[param_idx][j] {
                        let cov_value = covs.get(cov_name).copied().unwrap_or(0.0);
                        let reference = self.reference_values.get(cov_name).copied().unwrap_or(0.0);
                        x[(row_idx, beta_idx)] = cov_value - reference;
                        beta_idx += 1;
                    }
                }
            }
        }

        x
    }

    // ========== Accessors ==========

    /// Get the number of parameters
    pub fn n_params(&self) -> usize {
        self.param_names.len()
    }

    /// Get the number of covariates
    pub fn n_covariates(&self) -> usize {
        self.covariate_names.len()
    }

    /// Get the total number of β coefficients
    pub fn n_beta(&self) -> usize {
        Self::count_beta_coefficients(&self.covariate_mask, self.param_names.len())
    }

    /// Get the number of β coefficients to estimate
    pub fn n_beta_estimated(&self) -> usize {
        self.estimate_beta.iter().filter(|&&x| x).count()
    }

    /// Get the parameter names
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// Get the covariate names
    pub fn covariate_names(&self) -> &[String] {
        &self.covariate_names
    }

    /// Get the covariate mask
    pub fn covariate_mask(&self) -> &[Vec<bool>] {
        &self.covariate_mask
    }

    /// Get the current β coefficients
    pub fn beta(&self) -> &Col<f64> {
        &self.beta
    }

    /// Get mutable reference to β coefficients
    pub fn beta_mut(&mut self) -> &mut Col<f64> {
        &mut self.beta
    }

    /// Get the estimation flags
    pub fn estimate_beta(&self) -> &[bool] {
        &self.estimate_beta
    }

    /// Get the intercept value for a parameter
    pub fn intercept(&self, param_idx: usize) -> Option<f64> {
        let beta_idx = self.intercept_beta_index(param_idx).ok()?;
        Some(self.beta[beta_idx])
    }

    /// Check if a parameter has any covariate effects
    pub fn has_covariates(&self, param_idx: usize) -> bool {
        param_idx < self.covariate_mask.len() && self.covariate_mask[param_idx].iter().any(|&x| x)
    }

    // ========== Helper Methods ==========

    /// Count total beta coefficients given mask
    fn count_beta_coefficients(mask: &[Vec<bool>], n_params: usize) -> usize {
        let mut count = n_params; // Intercepts
        for row in mask {
            count += row.iter().filter(|&&x| x).count();
        }
        count
    }

    /// Get the beta index for a parameter's intercept
    fn intercept_beta_index(&self, param_idx: usize) -> Result<usize> {
        if param_idx >= self.n_params() {
            bail!("Parameter index {} out of range", param_idx);
        }

        let mut idx = 0;
        for i in 0..param_idx {
            idx += 1; // Intercept
            idx += self.covariate_mask[i].iter().filter(|&&x| x).count();
        }
        Ok(idx)
    }

    /// Get indices of β coefficients to estimate
    pub fn estimated_beta_indices(&self) -> Vec<usize> {
        self.estimate_beta
            .iter()
            .enumerate()
            .filter_map(|(i, &est)| if est { Some(i) } else { None })
            .collect()
    }
}

impl Default for CovariateModel {
    fn default() -> Self {
        Self {
            param_names: Vec::new(),
            covariate_names: Vec::new(),
            covariate_mask: Vec::new(),
            beta: Col::zeros(0),
            estimate_beta: Vec::new(),
            reference_values: HashMap::new(),
        }
    }
}

impl Serialize for CovariateModel {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("CovariateModel", 6)?;
        state.serialize_field("param_names", &self.param_names)?;
        state.serialize_field("covariate_names", &self.covariate_names)?;
        state.serialize_field("covariate_mask", &self.covariate_mask)?;

        // Convert Col<f64> to Vec<f64> for serialization
        let beta_vec: Vec<f64> = (0..self.beta.nrows()).map(|i| self.beta[i]).collect();
        state.serialize_field("beta", &beta_vec)?;

        state.serialize_field("estimate_beta", &self.estimate_beta)?;
        state.serialize_field("reference_values", &self.reference_values)?;
        state.end()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intercept_only() {
        let model = CovariateModel::intercept_only(vec!["CL", "V"]).unwrap();

        assert_eq!(model.n_params(), 2);
        assert_eq!(model.n_covariates(), 0);
        assert_eq!(model.n_beta(), 2); // Just intercepts

        // Set intercepts
        let mut model = model;
        model.set_intercepts(&[5.0, 50.0]).unwrap();

        // Compute mu (should just be intercepts)
        let mu = model.compute_mu(&HashMap::new());
        assert_eq!(mu[0], 5.0);
        assert_eq!(mu[1], 50.0);
    }

    #[test]
    fn test_with_covariates() {
        let model = CovariateModel::new(
            vec!["CL", "V"],
            vec!["WT", "SEX"],
            vec![
                vec![true, false], // CL depends on WT only
                vec![true, true],  // V depends on both
            ],
        )
        .unwrap();

        assert_eq!(model.n_params(), 2);
        assert_eq!(model.n_covariates(), 2);
        // Beta: CL_intercept, CL_WT, V_intercept, V_WT, V_SEX
        assert_eq!(model.n_beta(), 5);
    }

    #[test]
    fn test_compute_mu() {
        let mut model = CovariateModel::new(
            vec!["CL", "V"],
            vec!["WT"],
            vec![
                vec![true], // CL depends on WT
                vec![true], // V depends on WT
            ],
        )
        .unwrap();

        // Set beta: [CL_intercept, CL_WT, V_intercept, V_WT]
        model
            .set_beta(Col::from_fn(4, |i| match i {
                0 => 5.0,  // CL intercept
                1 => 0.1,  // CL_WT effect
                2 => 50.0, // V intercept
                3 => 1.0,  // V_WT effect
                _ => 0.0,
            }))
            .unwrap();

        let mut covs = HashMap::new();
        covs.insert("WT".to_string(), 70.0);

        let mu = model.compute_mu(&covs);

        // CL = 5.0 + 0.1 * 70 = 12.0
        assert!((mu[0] - 12.0).abs() < 1e-10);
        // V = 50.0 + 1.0 * 70 = 120.0
        assert!((mu[1] - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_centering() {
        let mut model = CovariateModel::new(vec!["CL"], vec!["WT"], vec![vec![true]]).unwrap();

        // Set reference value for centering
        model.set_reference("WT", 70.0).unwrap();

        // Set beta: [CL_intercept, CL_WT]
        model
            .set_beta(Col::from_fn(2, |i| if i == 0 { 5.0 } else { 0.1 }))
            .unwrap();

        // At reference weight, mu should equal intercept
        let mut covs = HashMap::new();
        covs.insert("WT".to_string(), 70.0);
        let mu = model.compute_mu(&covs);
        assert!((mu[0] - 5.0).abs() < 1e-10);

        // At 80 kg: CL = 5.0 + 0.1 * (80 - 70) = 6.0
        covs.insert("WT".to_string(), 80.0);
        let mu = model.compute_mu(&covs);
        assert!((mu[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_saemix_matrix() {
        // R matrix: matrix(c(1,0, 1,1), nrow=2, byrow=TRUE)
        // means: param1 depends on cov1 only, param2 depends on both
        let model = CovariateModel::from_saemix_matrix(
            vec!["CL", "V"],
            vec!["WT", "SEX"],
            &[1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();

        assert!(model.covariate_mask[0][0]); // CL-WT
        assert!(!model.covariate_mask[0][1]); // CL-SEX
        assert!(model.covariate_mask[1][0]); // V-WT
        assert!(model.covariate_mask[1][1]); // V-SEX
    }

    #[test]
    fn test_fix_intercept() {
        let mut model = CovariateModel::intercept_only(vec!["CL", "V"]).unwrap();

        assert!(model.estimate_beta[0]);
        model.fix_intercept(0).unwrap();
        assert!(!model.estimate_beta[0]);
        assert!(model.estimate_beta[1]); // V still estimated
    }
}
