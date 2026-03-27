use serde::Serialize;

use crate::model::CovariateModel;

#[derive(Debug, Clone, Default, Serialize)]
pub enum CovariateSpec {
    #[default]
    InEquation,
    Structured(CovariateEffectsSpec),
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CovariateEffectsSpec {
    pub subject_effects: Option<CovariateModel>,
    pub occasion_effects: Option<CovariateModel>,
}

impl CovariateEffectsSpec {
    pub fn subject_columns(&self) -> Vec<String> {
        self.subject_effects
            .as_ref()
            .map(|model| model.covariate_names().to_vec())
            .unwrap_or_default()
    }

    pub fn occasion_columns(&self) -> Vec<String> {
        self.occasion_effects
            .as_ref()
            .map(|model| model.covariate_names().to_vec())
            .unwrap_or_default()
    }
}
