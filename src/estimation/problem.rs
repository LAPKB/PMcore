use anyhow::{anyhow, Result};
use pharmsol::{
    AssayErrorModel, AssayErrorModels, Data, Equation, Event, ResidualErrorModel,
    ResidualErrorModels,
};
use std::collections::{BTreeSet, HashSet};

use crate::estimation::nonparametric::Theta;
use crate::model::parameter_space::{BoundedParameter, ParameterSpace, UnboundedParameter};
use crate::model::{EquationMetadataSource, Model, ModelBuilder};

pub trait Framework {
    type ErrorModels;
    /// The prior that seeds the algorithm.
    ///
    /// For the non-parametric framework this is a [`Theta`] (a discrete prior
    /// distribution that also carries the parameter space). For the parametric
    /// framework it is the [`ParameterSpace`] of unbounded parameters.
    type Prior;
}

pub struct Parametric;

impl Framework for Parametric {
    type ErrorModels = ResidualErrorModels;
    type Prior = ParameterSpace<UnboundedParameter>;
}

pub struct NonParametric;

impl Framework for NonParametric {
    type ErrorModels = AssayErrorModels;
    type Prior = Theta;
}

#[derive(Debug, Clone)]
pub struct EstimationProblem<E: Equation, F: Framework> {
    pub(crate) model: Model<E>,
    pub(crate) data: Data,
    pub(crate) error_models: F::ErrorModels,
    /// The prior that seeds the algorithm.
    ///
    /// For the non-parametric framework this is the prior [`Theta`], which also
    /// carries the parameter space. The parameter space is therefore not stored
    /// separately.
    pub(crate) prior: F::Prior,
}

impl<E: Equation + EquationMetadataSource> EstimationProblem<E, NonParametric> {
    /// Creates a non-parametric estimation problem.
    ///
    /// The `prior` is a [`Theta`] holding the prior distribution (the initial
    /// set of support points) together with the [`ParameterSpace`] it was built
    /// from. The parameter space is taken directly from the prior, so there is
    /// no separate parameter-declaration step.
    pub fn nonparametric(
        equation: E,
        data: Data,
        prior: Theta,
        error_models: AssayErrorModels,
    ) -> Result<Self> {
        let model_builder = Model::builder(equation);

        validate_nonparametric_parameters(&model_builder, prior.parameters())?;

        let model = model_builder.build()?;

        validate_nonparametric_error_models(&model, &data, &error_models)?;

        Ok(EstimationProblem {
            model,
            data,
            error_models,
            prior,
        })
    }
}

impl<E: Equation> EstimationProblem<E, Parametric> {
    /// Begins building a parametric estimation problem.
    pub fn parametric(equation: E, data: Data) -> ParametricBuilder<E> {
        ParametricBuilder {
            model: Model::builder(equation),
            data,
            parameters: ParameterSpace::<UnboundedParameter>::new(),
            error_models: Vec::new(),
        }
    }

    /// Returns the parameter space defined for this problem.
    pub fn parameters(&self) -> &ParameterSpace<UnboundedParameter> {
        &self.prior
    }
}

impl<E: Equation> EstimationProblem<E, NonParametric> {
    /// Returns the parameter space carried by the prior [`Theta`].
    pub fn parameters(&self) -> &ParameterSpace<BoundedParameter> {
        self.prior.parameters()
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

    pub fn error_model(mut self, name: impl Into<String>, model: ResidualErrorModel) -> Self {
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
            prior: self.parameters,
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
    model: &Model<E>,
    data: &Data,
    error_models: &AssayErrorModels,
) -> Result<()> {
    // Bind the (label-first) error models to the equation. This resolves and
    // validates that every declared output label maps to a valid model output.
    let bound = model
        .equation
        .bind_error_models(error_models)
        .map_err(|e| anyhow!("invalid assay error model output(s): {e}"))?;

    // Collect the set of model output indices that are actually observed in the
    // data, resolving each observation's output label the same way the simulator
    // does (exact name, then the `outeq_<N>` numeric alias).
    let mut observed_outputs: BTreeSet<usize> = BTreeSet::new();
    let mut unresolved_labels: BTreeSet<String> = BTreeSet::new();
    for subject in data.subjects() {
        for occasion in subject.occasions() {
            for event in occasion.events() {
                if let Event::Observation(obs) = event {
                    let label = obs.outeq().to_string();
                    match resolve_output_index(model, &label) {
                        Some(outeq) => {
                            observed_outputs.insert(outeq);
                        }
                        None => {
                            unresolved_labels.insert(label);
                        }
                    }
                }
            }
        }
    }

    if !unresolved_labels.is_empty() {
        let labels: Vec<String> = unresolved_labels.into_iter().collect();
        anyhow::bail!(
            "the data references output label(s) that are not defined by the model: {}",
            labels.join(", ")
        );
    }

    if observed_outputs.is_empty() {
        anyhow::bail!("the data contains no observations to fit");
    }

    // Every observed output must have a (non-`None`) assay error model.
    for &outeq in &observed_outputs {
        let has_model = matches!(
            bound.error_model(outeq),
            Ok(error_model) if *error_model != AssayErrorModel::None
        );

        if !has_model {
            let label = model
                .output_name(outeq)
                .map(|name| name.to_string())
                .unwrap_or_else(|| outeq.to_string());
            anyhow::bail!(
                "no assay error model defined for output '{}' (index {}), which is observed in the data",
                label,
                outeq
            );
        }
    }

    Ok(())
}

/// Resolves an observation output `label` to a model output index, mirroring the
/// simulator: first by exact output name, then via the `outeq_<N>` numeric alias.
fn resolve_output_index<E: Equation + EquationMetadataSource>(
    model: &Model<E>,
    label: &str,
) -> Option<usize> {
    model.output_index(label).or_else(|| {
        if !label.is_empty() && label.bytes().all(|b| b.is_ascii_digit()) {
            model.output_index(&format!("outeq_{label}"))
        } else {
            None
        }
    })
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
