use serde::{Deserialize, Serialize};

use super::{AssayErrorModels, ResidualErrorModel, ResidualErrorModels};

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

impl From<ErrorModels> for AssayErrorModels {
    fn from(val: ErrorModels) -> Self {
        val.models().clone()
    }
}

/// One parametric residual-error declaration and its estimation status.
///
/// Fixedness and residual-distribution semantics are both owned by PMcore.
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricErrorModel {
    model: ResidualErrorModel,
    estimate: bool,
    combined_component_estimated: [bool; 2],
    correlated_combined_component_estimated: [bool; 3],
}

impl ParametricErrorModel {
    pub fn new(model: ResidualErrorModel) -> Self {
        model.into()
    }

    pub fn with_estimate(mut self, estimate: bool) -> Self {
        self.estimate = estimate;
        self.combined_component_estimated = [estimate; 2];
        self.correlated_combined_component_estimated = [estimate; 3];
        self
    }

    /// Configure estimation of the additive component of a combined model.
    pub fn with_combined_additive_estimate(mut self, estimate: bool) -> Self {
        self.combined_component_estimated[0] = estimate;
        self.estimate = self.combined_component_estimated.iter().any(|value| *value);
        self.correlated_combined_component_estimated = [self.estimate; 3];
        self
    }

    /// Configure estimation of the proportional component of a combined model.
    pub fn with_combined_proportional_estimate(mut self, estimate: bool) -> Self {
        self.combined_component_estimated[1] = estimate;
        self.estimate = self.combined_component_estimated.iter().any(|value| *value);
        self.correlated_combined_component_estimated = [self.estimate; 3];
        self
    }

    pub fn fixed_combined_additive(self) -> Self {
        self.with_combined_additive_estimate(false)
    }

    pub fn fixed_combined_proportional(self) -> Self {
        self.with_combined_proportional_estimate(false)
    }

    /// Configure estimation of the additive SD of a correlated-combined model.
    pub fn with_correlated_combined_additive_estimate(mut self, estimate: bool) -> Self {
        self.correlated_combined_component_estimated[0] = estimate;
        self.estimate = self
            .correlated_combined_component_estimated
            .iter()
            .any(|value| *value);
        self.combined_component_estimated = [self.estimate; 2];
        self
    }

    /// Configure estimation of the proportional SD of a correlated-combined model.
    pub fn with_correlated_combined_proportional_estimate(mut self, estimate: bool) -> Self {
        self.correlated_combined_component_estimated[1] = estimate;
        self.estimate = self
            .correlated_combined_component_estimated
            .iter()
            .any(|value| *value);
        self.combined_component_estimated = [self.estimate; 2];
        self
    }

    /// Configure estimation of rho of a correlated-combined model.
    pub fn with_correlated_combined_correlation_estimate(mut self, estimate: bool) -> Self {
        self.correlated_combined_component_estimated[2] = estimate;
        self.estimate = self
            .correlated_combined_component_estimated
            .iter()
            .any(|value| *value);
        self.combined_component_estimated = [self.estimate; 2];
        self
    }

    pub fn fixed_correlated_combined_additive(self) -> Self {
        self.with_correlated_combined_additive_estimate(false)
    }

    pub fn fixed_correlated_combined_proportional(self) -> Self {
        self.with_correlated_combined_proportional_estimate(false)
    }

    pub fn fixed_correlated_combined_correlation(self) -> Self {
        self.with_correlated_combined_correlation_estimate(false)
    }

    pub fn fixed(self) -> Self {
        self.with_estimate(false)
    }

    pub fn model(&self) -> &ResidualErrorModel {
        &self.model
    }

    pub fn is_estimated(&self) -> bool {
        self.estimate
    }

    pub fn combined_component_estimated(&self) -> [bool; 2] {
        self.combined_component_estimated
    }

    pub fn correlated_combined_component_estimated(&self) -> [bool; 3] {
        self.correlated_combined_component_estimated
    }
}

impl From<ResidualErrorModel> for ParametricErrorModel {
    fn from(model: ResidualErrorModel) -> Self {
        Self {
            model,
            estimate: true,
            combined_component_estimated: [true, true],
            correlated_combined_component_estimated: [true, true, true],
        }
    }
}

/// Resolved parametric residual models plus estimation masks.
#[derive(Debug, Clone)]
pub struct ParametricErrorModels {
    models: ResidualErrorModels,
    estimated: Vec<bool>,
    combined_component_estimated: Vec<[bool; 2]>,
    correlated_combined_component_estimated: Vec<[bool; 3]>,
    output_names: Vec<Option<String>>,
}

impl ParametricErrorModels {
    pub(crate) fn new() -> Self {
        Self {
            models: ResidualErrorModels::new(),
            estimated: Vec::new(),
            combined_component_estimated: Vec::new(),
            correlated_combined_component_estimated: Vec::new(),
            output_names: Vec::new(),
        }
    }

    pub(crate) fn add(
        mut self,
        outeq: usize,
        output_name: impl Into<String>,
        declaration: ParametricErrorModel,
    ) -> Self {
        if self.estimated.len() <= outeq {
            self.estimated.resize(outeq + 1, false);
            self.combined_component_estimated
                .resize(outeq + 1, [false, false]);
            self.correlated_combined_component_estimated
                .resize(outeq + 1, [false, false, false]);
            self.output_names.resize(outeq + 1, None);
        }
        self.estimated[outeq] = declaration.estimate;
        self.combined_component_estimated[outeq] = declaration.combined_component_estimated;
        self.correlated_combined_component_estimated[outeq] =
            declaration.correlated_combined_component_estimated;
        self.output_names[outeq] = Some(output_name.into());
        self.models = self.models.add(outeq, declaration.model);
        self
    }

    pub fn models(&self) -> &ResidualErrorModels {
        &self.models
    }

    pub fn get(&self, outeq: usize) -> Option<&ResidualErrorModel> {
        self.models.get(outeq)
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub(crate) fn models_mut(&mut self) -> &mut ResidualErrorModels {
        &mut self.models
    }

    pub fn is_estimated(&self, outeq: usize) -> bool {
        self.estimated.get(outeq).copied().unwrap_or(false)
    }

    pub fn combined_component_estimated(&self, outeq: usize) -> [bool; 2] {
        self.combined_component_estimated
            .get(outeq)
            .copied()
            .unwrap_or([false, false])
    }

    pub fn correlated_combined_component_estimated(&self, outeq: usize) -> [bool; 3] {
        self.correlated_combined_component_estimated
            .get(outeq)
            .copied()
            .unwrap_or([false, false, false])
    }

    pub fn output_name(&self, outeq: usize) -> Option<&str> {
        self.output_names.get(outeq).and_then(Option::as_deref)
    }
}
