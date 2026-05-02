use pharmsol::AssayErrorModels;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "family", content = "models", rename_all = "snake_case")]
pub enum ErrorModels {
    Nonparametric(AssayErrorModels),
}

impl ErrorModels {
    pub fn models(&self) -> &AssayErrorModels {
        match self {
            Self::Nonparametric(models) => models,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.models().iter().next().is_none()
    }
}
