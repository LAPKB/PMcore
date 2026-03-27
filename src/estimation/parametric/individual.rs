//! Individual parameter estimates.

use anyhow::{bail, Result};
use faer::{Col, Mat};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Individual {
    subject_id: String,
    eta: Col<f64>,
    psi: Col<f64>,
    conditional_variance: Option<Mat<f64>>,
    objective_function: Option<f64>,
}

impl Individual {
    pub fn new(subject_id: impl Into<String>, eta: Col<f64>, psi: Col<f64>) -> Result<Self> {
        if eta.nrows() != psi.nrows() {
            bail!(
                "Random effects length ({}) must match parameter length ({})",
                eta.nrows(),
                psi.nrows()
            );
        }

        Ok(Self {
            subject_id: subject_id.into(),
            eta,
            psi,
            conditional_variance: None,
            objective_function: None,
        })
    }

    pub fn with_variance(
        subject_id: impl Into<String>,
        eta: Col<f64>,
        psi: Col<f64>,
        variance: Mat<f64>,
    ) -> Result<Self> {
        let n = eta.nrows();
        if variance.nrows() != n || variance.ncols() != n {
            bail!(
                "Variance matrix dimensions ({}x{}) must match parameter count ({})",
                variance.nrows(),
                variance.ncols(),
                n
            );
        }

        let mut individual = Self::new(subject_id, eta, psi)?;
        individual.conditional_variance = Some(variance);
        Ok(individual)
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub fn eta(&self) -> &Col<f64> {
        &self.eta
    }

    pub fn psi(&self) -> &Col<f64> {
        &self.psi
    }

    pub fn conditional_variance(&self) -> Option<&Mat<f64>> {
        self.conditional_variance.as_ref()
    }

    pub fn objective_function(&self) -> Option<f64> {
        self.objective_function
    }

    pub fn npar(&self) -> usize {
        self.eta.nrows()
    }

    pub fn standard_errors(&self) -> Option<Col<f64>> {
        self.conditional_variance
            .as_ref()
            .map(|var| Col::from_fn(self.npar(), |i| var[(i, i)].sqrt()))
    }

    pub fn set_conditional_variance(&mut self, variance: Mat<f64>) -> Result<()> {
        let n = self.npar();
        if variance.nrows() != n || variance.ncols() != n {
            bail!(
                "Variance matrix dimensions ({}x{}) must match parameter count ({})",
                variance.nrows(),
                variance.ncols(),
                n
            );
        }
        self.conditional_variance = Some(variance);
        Ok(())
    }

    pub fn set_objective_function(&mut self, objf: f64) {
        self.objective_function = Some(objf);
    }
}

#[derive(Debug, Clone, Default)]
pub struct IndividualEstimates {
    estimates: Vec<Individual>,
}

impl IndividualEstimates {
    pub fn new() -> Self {
        Self {
            estimates: Vec::new(),
        }
    }

    pub fn from_vec(estimates: Vec<Individual>) -> Self {
        Self { estimates }
    }

    pub fn add(&mut self, individual: Individual) {
        self.estimates.push(individual);
    }

    pub fn nsubjects(&self) -> usize {
        self.estimates.len()
    }

    pub fn get(&self, index: usize) -> Option<&Individual> {
        self.estimates.get(index)
    }

    pub fn get_by_id(&self, id: &str) -> Option<&Individual> {
        self.estimates.iter().find(|estimate| estimate.subject_id() == id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Individual> {
        self.estimates.iter()
    }

    pub fn eta_matrix(&self) -> Option<Mat<f64>> {
        if self.estimates.is_empty() {
            return None;
        }

        let n_subjects = self.estimates.len();
        let n_params = self.estimates[0].npar();

        Some(Mat::from_fn(n_subjects, n_params, |i, j| {
            self.estimates[i].eta()[j]
        }))
    }

    pub fn psi_matrix(&self) -> Option<Mat<f64>> {
        if self.estimates.is_empty() {
            return None;
        }

        let n_subjects = self.estimates.len();
        let n_params = self.estimates[0].npar();

        Some(Mat::from_fn(n_subjects, n_params, |i, j| {
            self.estimates[i].psi()[j]
        }))
    }

    pub fn eta_mean(&self) -> Option<Col<f64>> {
        if self.estimates.is_empty() {
            return None;
        }

        let n_subjects = self.estimates.len() as f64;
        let n_params = self.estimates[0].npar();

        Some(Col::from_fn(n_params, |j| {
            self.estimates.iter().map(|estimate| estimate.eta()[j]).sum::<f64>() / n_subjects
        }))
    }

    pub fn eta_covariance(&self) -> Option<Mat<f64>> {
        let mean = self.eta_mean()?;
        let n_subjects = self.estimates.len() as f64;
        let n_params = self.estimates[0].npar();

        Some(Mat::from_fn(n_params, n_params, |i, j| {
            self.estimates
                .iter()
                .map(|estimate| (estimate.eta()[i] - mean[i]) * (estimate.eta()[j] - mean[j]))
                .sum::<f64>()
                / (n_subjects - 1.0)
        }))
    }

    pub fn shrinkage(&self, population_variance: &Col<f64>) -> Option<Col<f64>> {
        let eta_cov = self.eta_covariance()?;
        let n_params = self.estimates[0].npar();

        Some(Col::from_fn(n_params, |i| {
            let eta_var = eta_cov[(i, i)];
            let pop_var = population_variance[i];
            if pop_var > 0.0 {
                1.0 - (eta_var / pop_var)
            } else {
                0.0
            }
        }))
    }
}

impl Serialize for Individual {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("Individual", 4)?;
        state.serialize_field("subject_id", &self.subject_id)?;

        let eta_vec: Vec<f64> = (0..self.eta.nrows()).map(|i| self.eta[i]).collect();
        state.serialize_field("eta", &eta_vec)?;

        let psi_vec: Vec<f64> = (0..self.psi.nrows()).map(|i| self.psi[i]).collect();
        state.serialize_field("psi", &psi_vec)?;

        state.serialize_field("objective_function", &self.objective_function)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Individual {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct IndividualData {
            subject_id: String,
            eta: Vec<f64>,
            psi: Vec<f64>,
            objective_function: Option<f64>,
        }

        let data = IndividualData::deserialize(deserializer)?;

        let eta = Col::from_fn(data.eta.len(), |i| data.eta[i]);
        let psi = Col::from_fn(data.psi.len(), |i| data.psi[i]);

        let mut individual =
            Individual::new(data.subject_id, eta, psi).map_err(serde::de::Error::custom)?;
        individual.objective_function = data.objective_function;
        Ok(individual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_creation() {
        let eta = Col::from_fn(2, |i| if i == 0 { 0.1 } else { -0.2 });
        let psi = Col::from_fn(2, |i| if i == 0 { 5.5 } else { 45.0 });

        let ind = Individual::new("SUBJ001", eta, psi).unwrap();

        assert_eq!(ind.subject_id(), "SUBJ001");
        assert_eq!(ind.npar(), 2);
        assert_eq!(ind.eta()[0], 0.1);
        assert_eq!(ind.psi()[1], 45.0);
    }

    #[test]
    fn test_individual_estimates_collection() {
        let mut estimates = IndividualEstimates::new();

        for i in 0..3 {
            let eta = Col::from_fn(2, |j| (i as f64) * 0.1 + (j as f64) * 0.05);
            let psi = Col::from_fn(2, |j| 5.0 + (i as f64) + (j as f64) * 10.0);
            let ind = Individual::new(format!("SUBJ{:03}", i), eta, psi).unwrap();
            estimates.add(ind);
        }

        assert_eq!(estimates.nsubjects(), 3);
        assert!(estimates.get_by_id("SUBJ001").is_some());
        assert!(estimates.eta_matrix().is_some());
    }
}