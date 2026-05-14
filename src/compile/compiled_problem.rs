use pharmsol::equation::Equation;

use crate::algorithms::Algorithm;
use crate::api::estimation_problem::NonparametricMethod;
use crate::api::{ErrorModels, OutputPlan, RuntimeOptions};
use crate::compile::{DesignContext, ExecutionCaches, ObservationIndex};
use crate::model::ModelDefinition;
use pharmsol::Data;

#[derive(Debug, Clone)]
pub struct CompiledProblem<E: Equation> {
    pub model: ModelDefinition<E>,
    pub data: Data,
    error_models: ErrorModels,
    method: NonparametricMethod,
    output: OutputPlan,
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
        method: NonparametricMethod,
        output: OutputPlan,
        runtime: RuntimeOptions,
        design: DesignContext,
        observation_index: ObservationIndex,
        caches: ExecutionCaches,
    ) -> Self {
        Self {
            model,
            data,
            error_models,
            method,
            output,
            runtime,
            design,
            observation_index,
            caches,
        }
    }

    pub(crate) fn method(&self) -> NonparametricMethod {
        self.method
    }

    pub fn algorithm(&self) -> Algorithm {
        self.method.algorithm()
    }

    pub fn error_models(&self) -> &ErrorModels {
        &self.error_models
    }

    pub fn output_plan(&self) -> &OutputPlan {
        &self.output
    }

    pub fn runtime_options(&self) -> &RuntimeOptions {
        &self.runtime
    }

    pub fn into_parts(self) -> (ModelDefinition<E>, Data) {
        (self.model, self.data)
    }
}
