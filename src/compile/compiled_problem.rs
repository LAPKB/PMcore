use pharmsol::equation::Equation;

use crate::api::{EstimationMethod, OutputPlan, RuntimeOptions};
use crate::compile::{DesignContext, ExecutionCaches, ObservationIndex};
use crate::model::ModelDefinition;
use pharmsol::Data;

#[derive(Debug, Clone)]
pub struct CompiledProblem<E: Equation> {
    pub model: ModelDefinition<E>,
    pub data: Data,
    method: EstimationMethod,
    output: OutputPlan,
    runtime: RuntimeOptions,
    pub design: DesignContext,
    pub observation_index: ObservationIndex,
    pub caches: ExecutionCaches,
}

impl<E: Equation> CompiledProblem<E> {
    pub fn new(
        model: ModelDefinition<E>,
        data: Data,
        method: EstimationMethod,
        output: OutputPlan,
        runtime: RuntimeOptions,
        design: DesignContext,
        observation_index: ObservationIndex,
        caches: ExecutionCaches,
    ) -> Self {
        Self {
            model,
            data,
            method,
            output,
            runtime,
            design,
            observation_index,
            caches,
        }
    }

    pub fn method(&self) -> EstimationMethod {
        self.method
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
