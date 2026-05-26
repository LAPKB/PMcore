use pharmsol::equation::Equation;

use crate::algorithms::Algorithm;
use crate::api::{ErrorModels, RuntimeOptions};
use crate::compile::{DesignContext, ExecutionCaches, ObservationIndex};
use crate::model::ModelDefinition;
use pharmsol::Data;

#[derive(Debug, Clone)]
pub struct CompiledProblem<E: Equation> {
    pub model: ModelDefinition<E>,
    pub data: Data,
    error_models: ErrorModels,
    algorithm: Algorithm,
    runtime: RuntimeOptions,
    pub design: DesignContext,
    pub observation_index: ObservationIndex,
    pub caches: ExecutionCaches,
}

impl<E: Equation> CompiledProblem<E> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: ModelDefinition<E>,
        data: Data,
        error_models: ErrorModels,
        algorithm: Algorithm,
        runtime: RuntimeOptions,
        design: DesignContext,
        observation_index: ObservationIndex,
        caches: ExecutionCaches,
    ) -> Self {
        Self {
            model,
            data,
            error_models,
            algorithm,
            runtime,
            design,
            observation_index,
            caches,
        }
    }

    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    pub fn error_models(&self) -> &ErrorModels {
        &self.error_models
    }

    pub fn runtime_options(&self) -> &RuntimeOptions {
        &self.runtime
    }

    pub fn into_parts(self) -> (ModelDefinition<E>, Data) {
        (self.model, self.data)
    }
}
