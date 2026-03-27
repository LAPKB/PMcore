//! Structured covariate model for population parameter regression.

use anyhow::{bail, Result};
use faer::{Col, Mat};
use serde::Serialize;
use std::collections::HashMap;

/// Covariate model specification.
#[derive(Debug, Clone)]
pub struct CovariateModel {
    param_names: Vec<String>,
    covariate_names: Vec<String>,
    covariate_mask: Vec<Vec<bool>>,
    beta: Col<f64>,
    estimate_beta: Vec<bool>,
    reference_values: HashMap<String, f64>,
}

impl CovariateModel {
    pub fn new(
        param_names: Vec<impl Into<String>>,
        covariate_names: Vec<impl Into<String>>,
        covariate_mask: Vec<Vec<bool>>,
    ) -> Result<Self> {
        let param_names: Vec<String> = param_names.into_iter().map(|s| s.into()).collect();
        let covariate_names: Vec<String> = covariate_names.into_iter().map(|s| s.into()).collect();

        let n_params = param_names.len();
        let n_covs = covariate_names.len();

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

        let n_beta = Self::count_beta_coefficients(&covariate_mask, n_params);

        Ok(Self {
            param_names,
            covariate_names,
            covariate_mask,
            beta: Col::zeros(n_beta),
            estimate_beta: vec![true; n_beta],
            reference_values: HashMap::new(),
        })
    }

    pub fn intercept_only(param_names: Vec<impl Into<String>>) -> Result<Self> {
        let param_names: Vec<String> = param_names.into_iter().map(|s| s.into()).collect();
        let n_params = param_names.len();

        Ok(Self {
            param_names,
            covariate_names: Vec::new(),
            covariate_mask: vec![Vec::new(); n_params],
            beta: Col::zeros(n_params),
            estimate_beta: vec![true; n_params],
            reference_values: HashMap::new(),
        })
    }

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
            idx += self.covariate_mask[i].iter().filter(|&&x| x).count();
        }

        Ok(())
    }

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

    pub fn fix_intercept(&mut self, param_idx: usize) -> Result<()> {
        let beta_idx = self.intercept_beta_index(param_idx)?;
        self.estimate_beta[beta_idx] = false;
        Ok(())
    }

    pub fn set_reference(&mut self, covariate: &str, value: f64) -> Result<()> {
        if !self.covariate_names.contains(&covariate.to_string()) {
            bail!("Unknown covariate: {}", covariate);
        }
        self.reference_values.insert(covariate.to_string(), value);
        Ok(())
    }

    pub fn compute_mu(&self, covariates: &HashMap<String, f64>) -> Col<f64> {
        let n_params = self.param_names.len();
        let mut mu = Col::zeros(n_params);
        let mut beta_idx = 0;

        for i in 0..n_params {
            mu[i] = self.beta[beta_idx];
            beta_idx += 1;

            for (j, cov_name) in self.covariate_names.iter().enumerate() {
                if self.covariate_mask[i][j] {
                    let cov_value = covariates.get(cov_name).copied().unwrap_or(0.0);
                    let reference = self.reference_values.get(cov_name).copied().unwrap_or(0.0);
                    mu[i] += self.beta[beta_idx] * (cov_value - reference);
                    beta_idx += 1;
                }
            }
        }

        mu
    }

    pub fn build_design_row(&self, covariates: &HashMap<String, f64>) -> Col<f64> {
        let n_beta = self.n_beta();
        let mut x = Col::zeros(n_beta);
        let mut beta_idx = 0;

        for i in 0..self.n_params() {
            x[beta_idx] = 1.0;
            beta_idx += 1;

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

    pub fn build_design_matrix(&self, all_covariates: &[HashMap<String, f64>]) -> Mat<f64> {
        let n_subjects = all_covariates.len();
        let n_params = self.n_params();
        let n_beta = self.n_beta();
        let n_rows = n_subjects * n_params;
        let mut x = Mat::zeros(n_rows, n_beta);

        for (subject_idx, covs) in all_covariates.iter().enumerate() {
            let mut beta_idx = 0;
            for param_idx in 0..n_params {
                let row_idx = subject_idx * n_params + param_idx;
                x[(row_idx, beta_idx)] = 1.0;
                beta_idx += 1;

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

    pub fn n_params(&self) -> usize {
        self.param_names.len()
    }

    pub fn n_covariates(&self) -> usize {
        self.covariate_names.len()
    }

    pub fn n_beta(&self) -> usize {
        Self::count_beta_coefficients(&self.covariate_mask, self.param_names.len())
    }

    pub fn n_beta_estimated(&self) -> usize {
        self.estimate_beta.iter().filter(|&&x| x).count()
    }

    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    pub fn covariate_names(&self) -> &[String] {
        &self.covariate_names
    }

    pub fn covariate_mask(&self) -> &[Vec<bool>] {
        &self.covariate_mask
    }

    pub fn beta(&self) -> &Col<f64> {
        &self.beta
    }

    pub fn beta_mut(&mut self) -> &mut Col<f64> {
        &mut self.beta
    }

    pub fn estimate_beta(&self) -> &[bool] {
        &self.estimate_beta
    }

    pub fn intercept(&self, param_idx: usize) -> Option<f64> {
        let beta_idx = self.intercept_beta_index(param_idx).ok()?;
        Some(self.beta[beta_idx])
    }

    pub fn has_covariates(&self, param_idx: usize) -> bool {
        param_idx < self.covariate_mask.len() && self.covariate_mask[param_idx].iter().any(|&x| x)
    }

    fn count_beta_coefficients(mask: &[Vec<bool>], n_params: usize) -> usize {
        let mut count = n_params;
        for row in mask {
            count += row.iter().filter(|&&x| x).count();
        }
        count
    }

    fn intercept_beta_index(&self, param_idx: usize) -> Result<usize> {
        if param_idx >= self.n_params() {
            bail!("Parameter index {} out of range", param_idx);
        }

        let mut idx = 0;
        for i in 0..param_idx {
            idx += 1;
            idx += self.covariate_mask[i].iter().filter(|&&x| x).count();
        }
        Ok(idx)
    }

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
        let beta_vec: Vec<f64> = (0..self.beta.nrows()).map(|i| self.beta[i]).collect();
        state.serialize_field("beta", &beta_vec)?;
        state.serialize_field("estimate_beta", &self.estimate_beta)?;
        state.serialize_field("reference_values", &self.reference_values)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intercept_only() {
        let model = CovariateModel::intercept_only(vec!["CL", "V"]).unwrap();

        assert_eq!(model.n_params(), 2);
        assert_eq!(model.n_covariates(), 0);
        assert_eq!(model.n_beta(), 2);

        let mut model = model;
        model.set_intercepts(&[5.0, 50.0]).unwrap();

        let mu = model.compute_mu(&HashMap::new());
        assert_eq!(mu[0], 5.0);
        assert_eq!(mu[1], 50.0);
    }

    #[test]
    fn test_with_covariates() {
        let model = CovariateModel::new(
            vec!["CL", "V"],
            vec!["WT", "SEX"],
            vec![vec![true, false], vec![true, true]],
        )
        .unwrap();

        assert_eq!(model.n_params(), 2);
        assert_eq!(model.n_covariates(), 2);
        assert_eq!(model.n_beta(), 5);
    }

    #[test]
    fn test_compute_mu() {
        let mut model =
            CovariateModel::new(vec!["CL", "V"], vec!["WT"], vec![vec![true], vec![true]]).unwrap();

        model
            .set_beta(Col::from_fn(4, |i| match i {
                0 => 5.0,
                1 => 0.1,
                2 => 50.0,
                3 => 1.0,
                _ => 0.0,
            }))
            .unwrap();

        let mut covs = HashMap::new();
        covs.insert("WT".to_string(), 70.0);
        let mu = model.compute_mu(&covs);

        assert!((mu[0] - 12.0).abs() < 1e-10);
        assert!((mu[1] - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_centering() {
        let mut model = CovariateModel::new(vec!["CL"], vec!["WT"], vec![vec![true]]).unwrap();
        model.set_reference("WT", 70.0).unwrap();
        model
            .set_beta(Col::from_fn(2, |i| if i == 0 { 5.0 } else { 0.1 }))
            .unwrap();

        let mut covs = HashMap::new();
        covs.insert("WT".to_string(), 70.0);
        let mu = model.compute_mu(&covs);
        assert!((mu[0] - 5.0).abs() < 1e-10);

        covs.insert("WT".to_string(), 80.0);
        let mu = model.compute_mu(&covs);
        assert!((mu[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_saemix_matrix() {
        let model = CovariateModel::from_saemix_matrix(
            vec!["CL", "V"],
            vec!["WT", "SEX"],
            &[1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();

        assert!(model.covariate_mask[0][0]);
        assert!(!model.covariate_mask[0][1]);
        assert!(model.covariate_mask[1][0]);
        assert!(model.covariate_mask[1][1]);
    }

    #[test]
    fn test_fix_intercept() {
        let mut model = CovariateModel::intercept_only(vec!["CL", "V"]).unwrap();

        assert!(model.estimate_beta[0]);
        model.fix_intercept(0).unwrap();
        assert!(!model.estimate_beta[0]);
        assert!(model.estimate_beta[1]);
    }
}
