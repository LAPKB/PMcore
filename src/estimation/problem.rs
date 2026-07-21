use anyhow::{anyhow, Result};
use pharmsol::{Data, Equation, Event};
use std::collections::{BTreeSet, HashSet};

use crate::estimation::error_models::{ParametricErrorModel, ParametricErrorModels};
use crate::estimation::nonparametric::Theta;
use crate::estimation::parametric::residual::RESIDUAL_OPTIMIZER_MAX_SIGMA;
use crate::estimation::parametric::transforms::psi_to_phi;
use crate::estimation::parametric::{
    reject_constraints, CovariateEffect, CovariateModel, Iov, Omega, ParametricConstraint,
    ParametricPrior,
};
use crate::estimation::{AssayErrorModel, AssayErrorModels, ResidualErrorModel};
use crate::model::parameter_space::{
    BoundedParameter, ParameterScale, ParameterSpace, UnboundedParameter,
};
use crate::model::{EquationMetadataSource, Model, ModelBuilder};

pub trait Framework {
    type ErrorModels;
    /// The prior that seeds the algorithm.
    ///
    /// For the non-parametric framework this is a [`Theta`] (a discrete prior
    /// distribution that also carries the parameter space). For the parametric
    /// framework it is a [`ParametricPrior`] carrying the population parameters
    /// and initial IIV covariance model.
    type Prior;
}

#[derive(Debug, Clone, Copy)]
pub struct Parametric;

impl Framework for Parametric {
    type ErrorModels = ParametricErrorModels;
    type Prior = ParametricPrior;
}

#[derive(Debug, Clone, Copy)]
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
        reject_sde_estimation::<E>()?;
        let model_builder = Model::builder(equation);

        validate_nonparametric_parameters(&model_builder, prior.parameters())?;

        let model = model_builder.build()?;

        let error_models = validate_nonparametric_error_models(&model, &data, &error_models)?;

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
            omega: None,
            iov: None,
            covariate_effects: Vec::new(),
            constraints: Vec::new(),
            error_models: Vec::new(),
        }
    }

    /// Returns the parameter space defined for this problem.
    pub fn parameters(&self) -> &ParameterSpace<UnboundedParameter> {
        self.prior.parameters()
    }

    /// Returns the initial IIV covariance matrix in random-effect order.
    pub fn omega(&self) -> &ndarray::Array2<f64> {
        self.prior.omega()
    }

    /// Returns random-effect names in η/Ω order.
    pub fn random_effect_names(&self) -> &[String] {
        self.prior.random_effect_names()
    }

    /// Returns IOV effect names in κ/Ω_IOV order, when configured.
    pub fn iov_effect_names(&self) -> Option<&[String]> {
        self.prior.iov_effect_names()
    }

    /// Returns the initial IOV covariance matrix, when configured.
    pub fn omega_iov(&self) -> Option<&ndarray::Array2<f64>> {
        self.prior.omega_iov()
    }

    /// Returns the declared residual-error models and estimation masks.
    pub fn residual_error_models(&self) -> &ParametricErrorModels {
        &self.error_models
    }

    /// Returns the fully validated subject-static covariate model, when declared.
    pub fn covariates(&self) -> Option<&CovariateModel> {
        self.prior.covariates()
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
    omega: Option<Omega>,
    iov: Option<Iov>,
    covariate_effects: Vec<CovariateEffect>,
    constraints: Vec<ParametricConstraint>,
    error_models: Vec<(String, ParametricErrorModel)>,
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

    /// Defines the initial IIV covariance structure. If omitted, PMcore uses an
    /// estimated diagonal identity Ω over declared random effects.
    pub fn omega(mut self, omega: Omega) -> Self {
        self.omega = Some(omega);
        self
    }

    /// Defines inter-occasion variability. κ effects are additive in φ-space
    /// for every occasion and have their own Ω_IOV.
    pub fn iov(mut self, iov: Iov) -> Self {
        self.iov = Some(iov);
        self
    }

    /// Adds one named subject-static transformed-space covariate effect.
    pub fn covariate_effect(mut self, effect: CovariateEffect) -> Self {
        self.covariate_effects.push(effect);
        self
    }

    /// Adds named subject-static covariate effects in stable iterator order.
    pub fn covariate_effects<I>(mut self, effects: I) -> Self
    where
        I: IntoIterator<Item = CovariateEffect>,
    {
        self.covariate_effects.extend(effects);
        self
    }

    /// Declares a parametric constraint. Unsupported nonlinear constraints are
    /// retained only long enough to fail explicitly during `build`.
    pub fn constraint(mut self, constraint: ParametricConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn error_model(
        mut self,
        name: impl Into<String>,
        model: impl Into<ParametricErrorModel>,
    ) -> Self {
        self.error_models.push((name.into(), model.into()));
        self
    }
}

impl<E: Equation + EquationMetadataSource> ParametricBuilder<E> {
    pub fn build(self) -> Result<EstimationProblem<E, Parametric>> {
        reject_sde_estimation::<E>()?;
        validate_parametric_parameters(&self.model, &self.parameters)?;
        validate_parametric_error_models(&self.model, &self.error_models)?;
        reject_constraints(&self.constraints)?;
        let covariates = if self.covariate_effects.is_empty() {
            None
        } else {
            Some(CovariateModel::resolve(
                self.covariate_effects,
                &self.parameters,
                &self.data,
            )?)
        };

        let model = self.model.build()?;
        let mut all_errors = ParametricErrorModels::new();
        for (name, error_model) in self.error_models {
            let outeq = model
                .output_index(&name)
                .ok_or_else(|| anyhow!("unknown equation output label: {name}"))?;

            all_errors = all_errors.add(outeq, name, error_model);
        }
        validate_parametric_data(&model, &self.data, &all_errors)?;

        let prior = ParametricPrior::new_with_covariates(
            self.parameters,
            self.omega,
            self.iov,
            covariates,
        )?;
        if let Some(covariates) = prior.covariates() {
            covariates.validate_initial_gls_rank(
                prior.parameters(),
                prior.random_effect_names(),
                prior.omega(),
            )?;
            let scales: Vec<_> = prior
                .parameters()
                .iter()
                .map(|parameter| parameter.scale)
                .collect();
            let population_phi: Vec<_> = prior
                .parameters()
                .iter()
                .map(|parameter| {
                    let psi = parameter.initial.unwrap_or(match parameter.scale {
                        ParameterScale::Identity | ParameterScale::Log => 1.0,
                        ParameterScale::Logit { lower, upper }
                        | ParameterScale::Probit { lower, upper } => 0.5 * (lower + upper),
                    });
                    psi_to_phi(psi, parameter.scale)
                })
                .collect();
            covariates.subject_population_parameters(&population_phi, &scales)?;
        }
        Ok(EstimationProblem {
            model,
            data: self.data,
            error_models: all_errors,
            prior,
        })
    }
}

fn reject_sde_estimation<E: Equation>() -> Result<()> {
    if matches!(E::kind(), pharmsol::equation::EqnKind::SDE) {
        anyhow::bail!(
            "EstimationProblem does not support SDE models; use SdeParticleFilter for \
             observation-conditioned filtering."
        );
    }
    Ok(())
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

    for parameter in parameters.iter() {
        if let Some(initial) = parameter.initial {
            if !initial.is_finite() {
                anyhow::bail!(
                    "invalid initial value for parameter '{}': initial values must be finite",
                    parameter.name
                );
            }
        }

        match parameter.scale {
            ParameterScale::Identity => {}
            ParameterScale::Log => {
                if parameter.initial.is_some_and(|initial| initial <= 0.0) {
                    anyhow::bail!(
                        "invalid initial value for log-scale parameter '{}': the initial value must be greater than zero",
                        parameter.name
                    );
                }
            }
            ParameterScale::Logit { lower, upper } | ParameterScale::Probit { lower, upper } => {
                if !lower.is_finite() || !upper.is_finite() || lower >= upper {
                    anyhow::bail!(
                        "invalid bounds for parameter '{}': bounds must be finite and lower ({lower}) must be strictly less than upper ({upper})",
                        parameter.name
                    );
                }
                if parameter
                    .initial
                    .is_some_and(|initial| initial <= lower || initial >= upper)
                {
                    anyhow::bail!(
                        "invalid initial value for bounded parameter '{}': the initial value must lie strictly inside ({lower}, {upper})",
                        parameter.name
                    );
                }
            }
        }
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
) -> Result<AssayErrorModels> {
    // Bind the label-first PMcore error models against neutral equation metadata.
    let output_names = (0..model.output_count())
        .filter_map(|index| model.output_name(index))
        .collect::<Vec<_>>();
    let bound = error_models
        .bind_outputs(output_names)
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

    Ok((*bound).clone())
}

/// Resolves an observation output `label` to a model output index, mirroring the
/// simulator: exact metadata name, then numeric `N` only for a declared `outeq_N`.
fn resolve_output_index<E: Equation + EquationMetadataSource>(
    model: &Model<E>,
    label: &str,
) -> Option<usize> {
    model.output_index(label).or_else(|| {
        (!label.is_empty() && label.chars().all(|ch| ch.is_ascii_digit()))
            .then(|| format!("outeq_{label}"))
            .and_then(|alias| model.output_index(&alias))
    })
}

fn validate_parametric_data<E: Equation + EquationMetadataSource>(
    model: &Model<E>,
    data: &Data,
    error_models: &ParametricErrorModels,
) -> Result<()> {
    if data.subjects().is_empty() {
        anyhow::bail!("parametric estimation requires at least one subject");
    }

    let mut measured_observations = 0;
    for subject in data.subjects() {
        for occasion in subject.occasions() {
            for event in occasion.events() {
                let Event::Observation(observation) = event else {
                    continue;
                };
                let label = observation.outeq().to_string();
                let output_index = resolve_output_index(model, &label).ok_or_else(|| {
                    anyhow!(
                        "parametric observation for subject '{}' at time {} references unknown model output '{}'; valid outputs are: {}",
                        subject.id(),
                        observation.time(),
                        label,
                        (0..model.output_count())
                            .filter_map(|index| model.output_name(index))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
                if observation.censored() {
                    anyhow::bail!(
                        "parametric estimation does not support {:?} censoring for subject '{}' at time {} on output '{}'; only missing and uncensored observations are supported",
                        observation.censoring(),
                        subject.id(),
                        observation.time(),
                        label
                    );
                }
                if observation.value().is_none() {
                    continue;
                }
                measured_observations += 1;
                if error_models.output_name(output_index).is_none() {
                    anyhow::bail!(
                        "no parametric residual model is defined for measured output '{}' (index {}) referenced by subject '{}' at time {}",
                        model.output_name(output_index).unwrap_or(&label),
                        output_index,
                        subject.id(),
                        observation.time()
                    );
                }
            }
        }
    }
    if measured_observations == 0 {
        anyhow::bail!("parametric estimation requires at least one measured observation");
    }
    Ok(())
}

fn validate_parametric_error_models<E: Equation + EquationMetadataSource>(
    model: &ModelBuilder<E>,
    error_models: &[(String, ParametricErrorModel)],
) -> Result<()> {
    if error_models.is_empty() {
        anyhow::bail!("at least one residual error model is required");
    }

    validate_error_model_labels(model, error_models.iter().map(|(name, _)| name.as_str()))?;

    for (output, declaration) in error_models {
        let combined_component_estimated = declaration.combined_component_estimated();
        let correlated_component_estimated = declaration.correlated_combined_component_estimated();
        if !matches!(declaration.model(), ResidualErrorModel::Combined { .. })
            && combined_component_estimated != [declaration.is_estimated(); 2]
        {
            anyhow::bail!(
                "combined-component estimation controls for output '{output}' require a combined residual model"
            );
        }
        if !matches!(
            declaration.model(),
            ResidualErrorModel::CorrelatedCombined { .. }
        ) && correlated_component_estimated != [declaration.is_estimated(); 3]
        {
            anyhow::bail!(
                "correlated-combined component estimation controls for output '{output}' require a correlated-combined residual model"
            );
        }
        match declaration.model() {
            ResidualErrorModel::Constant { a } if !a.is_finite() || *a <= 0.0 => {
                anyhow::bail!(
                    "constant residual SD for output '{output}' must be finite and greater than zero"
                )
            }
            ResidualErrorModel::Proportional { b } if !b.is_finite() || *b <= 0.0 => {
                anyhow::bail!(
                    "proportional residual SD coefficient for output '{output}' must be finite and greater than zero"
                )
            }
            ResidualErrorModel::Exponential { sigma } if !sigma.is_finite() || *sigma <= 0.0 => {
                anyhow::bail!(
                    "exponential residual log-scale SD for output '{output}' must be finite and greater than zero"
                )
            }
            ResidualErrorModel::Combined { a, b }
                if !a.is_finite()
                    || !b.is_finite()
                    || *a < 0.0
                    || *b < 0.0
                    || (*a == 0.0 && *b == 0.0) =>
            {
                anyhow::bail!(
                    "combined residual SD coefficients for output '{output}' must be finite, non-negative, and not both zero"
                )
            }
            ResidualErrorModel::Combined { a, .. }
                if combined_component_estimated[0] && *a == 0.0 =>
            {
                anyhow::bail!(
                    "estimated combined additive SD for output '{output}' must be greater than zero"
                )
            }
            ResidualErrorModel::Combined { b, .. }
                if combined_component_estimated[1] && *b == 0.0 =>
            {
                anyhow::bail!(
                    "estimated combined proportional SD for output '{output}' must be greater than zero"
                )
            }
            ResidualErrorModel::Combined { a, .. }
                if combined_component_estimated[0] && *a > RESIDUAL_OPTIMIZER_MAX_SIGMA =>
            {
                anyhow::bail!(
                    "estimated combined additive SD for output '{output}' must not exceed the optimizer maximum {RESIDUAL_OPTIMIZER_MAX_SIGMA}"
                )
            }
            ResidualErrorModel::Combined { b, .. }
                if combined_component_estimated[1] && *b > RESIDUAL_OPTIMIZER_MAX_SIGMA =>
            {
                anyhow::bail!(
                    "estimated combined proportional SD for output '{output}' must not exceed the optimizer maximum {RESIDUAL_OPTIMIZER_MAX_SIGMA}"
                )
            }
            ResidualErrorModel::CorrelatedCombined { a, b, rho }
                if !a.is_finite()
                    || !b.is_finite()
                    || *a <= 0.0
                    || *b <= 0.0
                    || !rho.is_finite()
                    || *rho <= -1.0
                    || *rho >= 1.0 =>
            {
                anyhow::bail!(
                    "correlated-combined residual declaration for output '{output}' requires finite positive a and b and finite rho strictly inside (-1, 1)"
                )
            }
            ResidualErrorModel::CorrelatedCombined { a, .. }
                if correlated_component_estimated[0] && *a > RESIDUAL_OPTIMIZER_MAX_SIGMA =>
            {
                anyhow::bail!(
                    "estimated correlated-combined additive SD for output '{output}' must not exceed the optimizer maximum {RESIDUAL_OPTIMIZER_MAX_SIGMA}"
                )
            }
            ResidualErrorModel::CorrelatedCombined { b, .. }
                if correlated_component_estimated[1] && *b > RESIDUAL_OPTIMIZER_MAX_SIGMA =>
            {
                anyhow::bail!(
                    "estimated correlated-combined proportional SD for output '{output}' must not exceed the optimizer maximum {RESIDUAL_OPTIMIZER_MAX_SIGMA}"
                )
            }
            _ => {}
        }
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{reject_sde_estimation, EstimationProblem, RESIDUAL_OPTIMIZER_MAX_SIGMA};
    use crate::estimation::ParametricErrorModel;
    use crate::model::parameter_space::Parameter;
    use crate::ResidualErrorModel;
    use pharmsol::prelude::*;
    use pharmsol::{Censor, Data, Subject, SubjectBuilderExt};

    fn equation_with_outputs(outputs: [&str; 2]) -> pharmsol::ODE {
        pharmsol::equation::ODE::new(
            |_x, _p, _t, dx, _b, _rateiv, _cov| dx[0] = 0.0,
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |_x, p, _t, _cov, y| {
                y[0] = p[0];
                y[1] = p[0];
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(2)
        .with_metadata(
            equation::metadata::new("parametric_validation")
                .parameters(["value"])
                .states(["state"])
                .outputs(outputs)
                .route(equation::Route::bolus("dose").to_state("state")),
        )
        .unwrap()
    }

    fn equation() -> pharmsol::ODE {
        equation_with_outputs(["cp", "effect"])
    }

    fn measured_data(output: &str) -> Data {
        Data::new(vec![Subject::builder("subject-1")
            .observation(3.5, 1.25, output)
            .build()])
    }

    #[test]
    fn deterministic_model_kind_support_is_fail_closed() {
        assert!(reject_sde_estimation::<pharmsol::Analytical>().is_ok());
        assert!(reject_sde_estimation::<pharmsol::ODE>().is_ok());

        let error = reject_sde_estimation::<pharmsol::SDE>()
            .expect_err("EstimationProblem must reject SDE models")
            .to_string();
        assert!(error.contains("SDE"));
        assert!(error.contains("SdeParticleFilter"));
    }

    #[test]
    fn parametric_parameter_domains_fail_closed() {
        let invalid = [
            Parameter::real("value").with_initial(f64::NAN),
            Parameter::log("value").with_initial(0.0),
            Parameter::logit("value", f64::NEG_INFINITY, 1.0).with_initial(0.5),
            Parameter::probit("value", 1.0, 1.0).with_initial(1.0),
            Parameter::logit("value", 0.0, 1.0).with_initial(0.0),
            Parameter::probit("value", 0.0, 1.0).with_initial(1.0),
        ];

        for parameter in invalid {
            let error = EstimationProblem::parametric(equation(), measured_data("cp"))
                .parameter(parameter)
                .error_model("cp", ResidualErrorModel::constant(1.0))
                .build()
                .expect_err("invalid parameter domain must fail")
                .to_string();
            assert!(error.contains("value"));
        }
    }

    #[test]
    fn valid_parameter_scales_and_default_initials_are_preserved() {
        for parameter in [
            Parameter::real("value"),
            Parameter::log("value"),
            Parameter::logit("value", 0.0, 2.0),
            Parameter::probit("value", 0.0, 2.0),
        ] {
            assert!(
                EstimationProblem::parametric(equation(), measured_data("cp"))
                    .parameter(parameter)
                    .error_model("cp", ResidualErrorModel::constant(1.0))
                    .build()
                    .is_ok(),
                "supported parameter scale must build"
            );
        }
    }

    #[test]
    fn combined_estimated_components_respect_optimizer_bound() {
        let too_large = RESIDUAL_OPTIMIZER_MAX_SIGMA * 2.0;
        let error = EstimationProblem::parametric(equation(), measured_data("cp"))
            .parameter(Parameter::log("value"))
            .error_model("cp", ResidualErrorModel::combined(too_large, 0.1))
            .build()
            .expect_err("estimated combined components above the optimizer bound must fail")
            .to_string();
        assert!(error.contains("optimizer maximum"));

        assert!(
            EstimationProblem::parametric(equation(), measured_data("cp"))
                .parameter(Parameter::log("value"))
                .error_model(
                    "cp",
                    ParametricErrorModel::new(ResidualErrorModel::combined(too_large, 0.1))
                        .fixed_combined_additive(),
                )
                .build()
                .is_ok()
        );
    }

    #[test]
    fn parametric_construction_rejects_empty_or_unmeasured_data() {
        let empty_error = EstimationProblem::parametric(equation(), Data::new(vec![]))
            .parameter(Parameter::log("value"))
            .error_model("cp", ResidualErrorModel::constant(1.0))
            .build()
            .expect_err("parametric data must contain a subject")
            .to_string();
        assert!(empty_error.contains("at least one subject"));

        let missing_only = Data::new(vec![Subject::builder("missing-subject")
            .missing_observation(2.0, "cp")
            .build()]);
        let missing_error = EstimationProblem::parametric(equation(), missing_only)
            .parameter(Parameter::log("value"))
            .error_model("cp", ResidualErrorModel::constant(1.0))
            .build()
            .expect_err("parametric data must contain a measured observation")
            .to_string();
        assert!(missing_error.contains("measured observation"));
    }

    #[test]
    fn parametric_observation_outputs_fail_closed() {
        let unknown = EstimationProblem::parametric(equation(), measured_data("unknown"))
            .parameter(Parameter::log("value"))
            .error_model("cp", ResidualErrorModel::constant(1.0))
            .build()
            .expect_err("unknown observation output must fail")
            .to_string();
        assert!(unknown.contains("subject-1"));
        assert!(unknown.contains("3.5"));
        assert!(unknown.contains("unknown"));

        let unknown_missing_data = Data::new(vec![Subject::builder("missing-subject")
            .missing_observation(4.5, "unknown")
            .build()]);
        let unknown_missing = EstimationProblem::parametric(equation(), unknown_missing_data)
            .parameter(Parameter::log("value"))
            .error_model("cp", ResidualErrorModel::constant(1.0))
            .build()
            .expect_err("unknown missing-observation output must fail")
            .to_string();
        assert!(unknown_missing.contains("missing-subject"));
        assert!(unknown_missing.contains("4.5"));
        assert!(unknown_missing.contains("unknown"));

        let missing_residual = EstimationProblem::parametric(equation(), measured_data("cp"))
            .parameter(Parameter::log("value"))
            .error_model("effect", ResidualErrorModel::constant(1.0))
            .build()
            .expect_err("measured output without residual model must fail")
            .to_string();
        assert!(missing_residual.contains("cp"));
        assert!(missing_residual.contains("subject-1"));
    }

    #[test]
    fn missing_observations_with_declared_output_names_remain_supported() {
        let data = Data::new(vec![Subject::builder("subject-1")
            .observation(1.0, 1.0, "cp")
            .missing_observation(2.0, "effect")
            .build()]);
        assert!(
            EstimationProblem::parametric(equation(), data)
                .parameter(Parameter::log("value"))
                .error_model("cp", ResidualErrorModel::constant(1.0))
                .build()
                .is_ok(),
            "missing values need no residual model"
        );

        let numeric = EstimationProblem::parametric(equation(), measured_data("0"))
            .parameter(Parameter::log("value"))
            .error_model("cp", ResidualErrorModel::constant(1.0))
            .build()
            .expect_err("numeric labels must not alias arbitrarily named metadata outputs")
            .to_string();
        assert!(numeric.contains("unknown model output '0'"));
    }

    #[test]
    fn numeric_output_aliases_preserve_leading_zeroes() {
        assert!(EstimationProblem::parametric(
            equation_with_outputs(["outeq_00", "effect"]),
            measured_data("00"),
        )
        .parameter(Parameter::log("value"))
        .error_model("outeq_00", ResidualErrorModel::constant(1.0))
        .build()
        .is_ok());

        let error = EstimationProblem::parametric(
            equation_with_outputs(["outeq_0", "effect"]),
            measured_data("00"),
        )
        .parameter(Parameter::log("value"))
        .error_model("outeq_0", ResidualErrorModel::constant(1.0))
        .build()
        .expect_err("00 must not resolve to outeq_0")
        .to_string();
        assert!(error.contains("unknown model output '00'"));
    }

    #[test]
    fn estimated_and_fixed_parameters_without_iiv_are_declared_independently() {
        let estimated = EstimationProblem::parametric(equation(), measured_data("cp"))
            .parameter(
                Parameter::log("value")
                    .with_initial(1.0)
                    .without_random_effect(),
            )
            .error_model("cp", ResidualErrorModel::constant(1.0))
            .build()
            .expect("estimated non-IIV theta should use the numerical M-step");
        assert!(estimated.parameters().items[0].estimate);
        assert!(!estimated.parameters().items[0].random_effect);

        assert!(
            EstimationProblem::parametric(equation(), measured_data("cp"))
                .parameter(
                    Parameter::log("value")
                        .with_initial(1.0)
                        .fixed()
                        .without_random_effect(),
                )
                .error_model("cp", ResidualErrorModel::constant(1.0))
                .build()
                .is_ok()
        );
    }

    #[test]
    fn parametric_construction_rejects_bloq_observations() {
        assert_parametric_censoring_rejected(Censor::BLOQ, "BLOQ");
    }

    #[test]
    fn parametric_construction_rejects_aloq_observations() {
        assert_parametric_censoring_rejected(Censor::ALOQ, "ALOQ");
    }

    fn assert_parametric_censoring_rejected(censoring: Censor, label: &str) {
        let data = Data::new(vec![Subject::builder("censored-subject")
            .censored_observation(3.5, 1.25, "cp", censoring)
            .build()]);

        let error = EstimationProblem::parametric(equation(), data)
            .parameter(Parameter::log("value").with_initial(1.0))
            .error_model("cp", ResidualErrorModel::constant(1.0))
            .build()
            .expect_err("ParametricBuilder::build must reject censored observations")
            .to_string();

        assert!(error.contains(label));
        assert!(error.contains("censored-subject"));
        assert!(error.contains("3.5"));
        assert!(error.contains("cp"));
    }
}
