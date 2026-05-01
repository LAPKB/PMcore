use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct VariabilityModel {
    pub subject: RandomEffectsSpec,
    pub occasion: Option<RandomEffectsSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RandomEffectsSpec {
    pub enabled_for: Vec<bool>,
    pub covariance: CovarianceStructure,
}

impl Default for RandomEffectsSpec {
    fn default() -> Self {
        Self {
            enabled_for: Vec::new(),
            covariance: CovarianceStructure::Diagonal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CovarianceStructure {
    Diagonal,
    Full,
}
