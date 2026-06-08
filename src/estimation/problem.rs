use anyhow::{anyhow, Result};
use pharmsol::{
    AssayErrorModel, AssayErrorModels, Data, Equation, ResidualErrorModel, ResidualErrorModels,
};
use std::collections::HashSet;

use crate::model::parameter_space::{BoundedParameter, ParameterSpace, UnboundedParameter};
use crate::model::{EquationMetadataSource, Model, ModelBuilder};

pub trait Framework {
    type ErrorModels;
    type Parameters;
}

pub struct Parametric;

impl Framework for Parametric {
    type ErrorModels = ResidualErrorModels;
    type Parameters = ParameterSpace<UnboundedParameter>;
}

pub struct NonParametric;

impl Framework for NonParametric {
    type ErrorModels = AssayErrorModels;
    type Parameters = ParameterSpace<BoundedParameter>;
}

#[derive(Debug, Clone)]
pub struct EstimationProblem<E: Equation, F: Framework> {
    pub(crate) model: Model<E>,
    pub(crate) data: Data,
    pub(crate) error_models: F::ErrorModels,
    pub(crate) parameters: F::Parameters,
}

pub struct EstimationProblemBuilder<E: Equation> {
    model: ModelBuilder<E>,
    data: Data,
}

impl<E: Equation> EstimationProblemBuilder<E> {
    pub fn nonparametric(self) -> NonParametricBuilder<E> {
        NonParametricBuilder {
            model: self.model,
            data: self.data,
            parameters: ParameterSpace::<BoundedParameter>::new(),
            error_models: Vec::new(),
        }
    }

    pub fn parametric(self) -> ParametricBuilder<E> {
        ParametricBuilder {
            model: self.model,
            data: self.data,
            parameters: ParameterSpace::<UnboundedParameter>::new(),
            error_models: Vec::new(),
        }
    }
}

impl<E: Equation> EstimationProblem<E, NonParametric> {
    pub fn builder(equation: E, data: Data) -> EstimationProblemBuilder<E> {
        EstimationProblemBuilder {
            model: Model::builder(equation),
            data,
        }
    }
}

pub struct NonParametricBuilder<E: Equation> {
    model: ModelBuilder<E>,
    data: Data,
    parameters: ParameterSpace<BoundedParameter>,
    error_models: Vec<(String, AssayErrorModel)>,
}

impl<E: Equation> NonParametricBuilder<E> {
    pub fn parameter(mut self, parameter: impl Into<BoundedParameter>) -> Self {
        self.parameters.push(parameter.into());
        self
    }

    pub fn parameters<P, I>(mut self, parameters: I) -> Self
    where
        P: Into<BoundedParameter>,
        I: IntoIterator<Item = P>,
    {
        for param in parameters {
            self.parameters.push(param.into());
        }
        self
    }

    pub fn error(mut self, name: impl Into<String>, model: AssayErrorModel) -> Self {
        self.error_models.push((name.into(), model));
        self
    }
}

impl<E: Equation + EquationMetadataSource> NonParametricBuilder<E> {
    pub fn build(self) -> Result<EstimationProblem<E, NonParametric>> {
        validate_nonparametric_parameters(&self.model, &self.parameters)?;
        validate_nonparametric_error_models(&self.model, &self.error_models)?;

        let mut all_errors = AssayErrorModels::new();
        for (name, error_model) in self.error_models {
            let outeq = self
                .model
                .output_index(&name)
                .ok_or_else(|| anyhow!("unknown equation output label: {name}"))?;

            all_errors = all_errors.add(outeq, error_model)?;
        }

        Ok(EstimationProblem {
            model: self.model.build()?,
            data: self.data,
            error_models: all_errors,
            parameters: self.parameters,
        })
    }
}

pub struct ParametricBuilder<E: Equation> {
    model: ModelBuilder<E>,
    data: Data,
    parameters: ParameterSpace<UnboundedParameter>,
    error_models: Vec<(String, ResidualErrorModel)>,
}

impl<E: Equation> ParametricBuilder<E> {
    pub fn parameter(mut self, parameter: impl Into<UnboundedParameter>) -> Self {
        self.parameters.push(parameter.into());
        self
    }

    pub fn parameters<P, I>(mut self, parameters: I) -> Self
    where
        P: Into<UnboundedParameter>,
        I: IntoIterator<Item = P>,
    {
        for param in parameters {
            self.parameters.push(param.into());
        }
        self
    }

    pub fn error(mut self, name: impl Into<String>, model: ResidualErrorModel) -> Self {
        self.error_models.push((name.into(), model));
        self
    }
}

impl<E: Equation + EquationMetadataSource> ParametricBuilder<E> {
    pub fn build(self) -> Result<EstimationProblem<E, Parametric>> {
        validate_parametric_parameters(&self.model, &self.parameters)?;
        validate_parametric_error_models(&self.model, &self.error_models)?;

        let mut all_errors = ResidualErrorModels::new();
        for (name, error_model) in self.error_models {
            let outeq = self
                .model
                .output_index(&name)
                .ok_or_else(|| anyhow!("unknown equation output label: {name}"))?;

            all_errors = all_errors.add(outeq, error_model);
        }

        Ok(EstimationProblem {
            model: self.model.build()?,
            data: self.data,
            error_models: all_errors,
            parameters: self.parameters,
        })
    }
}

fn validate_nonparametric_parameters<E: Equation + EquationMetadataSource>(
    model: &ModelBuilder<E>,
    parameters: &ParameterSpace<BoundedParameter>,
) -> Result<()> {
    if parameters.is_empty() {
        anyhow::bail!("at least one parameter is required for non-parametric models");
    }

    for parameter in parameters.iter() {
        if !parameter.lower.is_finite() || !parameter.upper.is_finite() {
            anyhow::bail!(
                "invalid bounds for parameter '{}': bounds must be finite numbers",
                parameter.name
            );
        }

        if parameter.lower >= parameter.upper {
            anyhow::bail!(
                "invalid bounds for parameter '{}': lower bound ({}) must be strictly less than upper bound ({})",
                parameter.name,
                parameter.lower,
                parameter.upper
            );
        }
    }

    let names: Vec<String> = parameters
        .iter()
        .map(|parameter| parameter.name.clone())
        .collect();
    validate_parameter_declarations(model, &names)
}

fn validate_parametric_parameters<E: Equation + EquationMetadataSource>(
    model: &ModelBuilder<E>,
    parameters: &ParameterSpace<UnboundedParameter>,
) -> Result<()> {
    if parameters.is_empty() {
        anyhow::bail!("at least one parameter is required for parametric models");
    }

    let names: Vec<String> = parameters
        .iter()
        .map(|parameter| parameter.name.clone())
        .collect();
    validate_parameter_declarations(model, &names)
}

fn validate_parameter_declarations<E: Equation + EquationMetadataSource>(
    model: &ModelBuilder<E>,
    provided_names: &[String],
) -> Result<()> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut duplicates: Vec<String> = Vec::new();
    for name in provided_names {
        if !seen.insert(name.as_str()) {
            duplicates.push(name.clone());
        }
    }

    if !duplicates.is_empty() {
        duplicates.sort();
        duplicates.dedup();
        anyhow::bail!(
            "duplicate parameter declarations found: {}",
            duplicates.join(", ")
        );
    }

    let declared = model.parameter_names();

    let unknown: Vec<String> = provided_names
        .iter()
        .filter(|name| model.parameter_index(name).is_none())
        .cloned()
        .collect();
    if !unknown.is_empty() {
        anyhow::bail!(
            "unknown parameter name(s): {}. Valid parameters are: {}",
            unknown.join(", "),
            declared.join(", ")
        );
    }

    let provided: HashSet<&str> = provided_names.iter().map(|name| name.as_str()).collect();
    let missing: Vec<String> = declared
        .iter()
        .filter(|name| !provided.contains(name.as_str()))
        .cloned()
        .collect();

    if !missing.is_empty() {
        anyhow::bail!("missing parameter declaration(s): {}", missing.join(", "));
    }

    Ok(())
}

fn validate_nonparametric_error_models<E: Equation + EquationMetadataSource>(
    model: &ModelBuilder<E>,
    error_models: &[(String, AssayErrorModel)],
) -> Result<()> {
    if error_models.is_empty() {
        anyhow::bail!("at least one assay error model is required");
    }

    validate_error_model_labels(model, error_models.iter().map(|(name, _)| name.as_str()))
}

fn validate_parametric_error_models<E: Equation + EquationMetadataSource>(
    model: &ModelBuilder<E>,
    error_models: &[(String, ResidualErrorModel)],
) -> Result<()> {
    if error_models.is_empty() {
        anyhow::bail!("at least one residual error model is required");
    }

    validate_error_model_labels(model, error_models.iter().map(|(name, _)| name.as_str()))
}

fn validate_error_model_labels<'a, E, I>(model: &ModelBuilder<E>, labels: I) -> Result<()>
where
    E: Equation + EquationMetadataSource,
    I: IntoIterator<Item = &'a str>,
{
    let valid_outputs = model.output_names();
    let mut seen_output_indexes: HashSet<usize> = HashSet::new();

    for name in labels {
        let outeq = model.output_index(name).ok_or_else(|| {
            anyhow!(
                "unknown equation output label: {}. Valid outputs are: {}",
                name,
                valid_outputs.join(", ")
            )
        })?;

        if !seen_output_indexes.insert(outeq) {
            anyhow::bail!(
                "duplicate error model declaration for output '{}' (index {})",
                name,
                outeq
            );
        }
    }

    Ok(())
}
