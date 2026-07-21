//! Named subject-static covariate effects in transformed population space.
//!
//! This module deliberately owns declaration, validation, design, and exact
//! GLS primitives without depending on pharmsol's interpolation semantics.

use std::collections::{BTreeMap, HashMap, HashSet};

use ndarray::Array2;
use pharmsol::Data;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::{ParameterScale, ParameterSpace, UnboundedParameter};

use super::{covariance::cholesky_lower, transforms::phi_to_psi};

/// One named transformed-space population covariate effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovariateEffect {
    parameter: String,
    covariate: String,
    kind: CovariateEffectKind,
    initial: Option<f64>,
    estimated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case", deny_unknown_fields)]
enum CovariateEffectKind {
    Continuous { center: f64 },
    Categorical { reference: f64, level: f64 },
}

/// Public family metadata without exposing mutable declaration fields.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum CovariateEffectFamily {
    Continuous { center: f64 },
    Categorical { reference: f64, level: f64 },
}

impl CovariateEffect {
    pub fn continuous(
        parameter: impl Into<String>,
        covariate: impl Into<String>,
        center: f64,
    ) -> Self {
        Self {
            parameter: parameter.into(),
            covariate: covariate.into(),
            kind: CovariateEffectKind::Continuous {
                center: canonical_zero(center),
            },
            initial: None,
            estimated: true,
        }
    }

    pub fn categorical(
        parameter: impl Into<String>,
        covariate: impl Into<String>,
        reference: f64,
        level: f64,
    ) -> Self {
        Self {
            parameter: parameter.into(),
            covariate: covariate.into(),
            kind: CovariateEffectKind::Categorical {
                reference: canonical_zero(reference),
                level: canonical_zero(level),
            },
            initial: None,
            estimated: true,
        }
    }

    pub fn with_initial(mut self, beta: f64) -> Self {
        self.initial = Some(beta);
        self
    }

    pub fn fixed(mut self) -> Self {
        self.estimated = false;
        self
    }

    pub fn parameter(&self) -> &str {
        &self.parameter
    }

    pub fn covariate(&self) -> &str {
        &self.covariate
    }

    pub fn family(&self) -> CovariateEffectFamily {
        match self.kind {
            CovariateEffectKind::Continuous { center } => {
                CovariateEffectFamily::Continuous { center }
            }
            CovariateEffectKind::Categorical { reference, level } => {
                CovariateEffectFamily::Categorical { reference, level }
            }
        }
    }

    pub fn initial(&self) -> Option<f64> {
        self.initial
    }

    pub fn estimated(&self) -> bool {
        self.estimated
    }

    pub fn name(&self) -> String {
        match self.kind {
            CovariateEffectKind::Continuous { .. } => {
                format!("beta:{}:{}", self.parameter, self.covariate)
            }
            CovariateEffectKind::Categorical { level, .. } => format!(
                "beta:{}:{}:{}",
                self.parameter,
                self.covariate,
                stable_number(level)
            ),
        }
    }

    fn design_value(&self, value: f64) -> f64 {
        match self.kind {
            CovariateEffectKind::Continuous { center } => value - center,
            CovariateEffectKind::Categorical { level, .. } => {
                if value == level {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn stable_number(value: f64) -> String {
    canonical_zero(value).to_string()
}

/// Rejection-only declaration for constraints outside the supported domains.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParametricConstraint {
    Nonlinear { description: String },
}

impl ParametricConstraint {
    pub fn nonlinear(description: impl Into<String>) -> Self {
        Self::Nonlinear {
            description: description.into(),
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Self::Nonlinear { description } => description,
        }
    }
}

/// Immutable coefficient estimate in canonical declaration order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovariateEstimate {
    name: String,
    declaration_index: usize,
    estimate: f64,
    estimated: bool,
}

impl CovariateEstimate {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn declaration_index(&self) -> usize {
        self.declaration_index
    }
    pub fn estimate(&self) -> f64 {
        self.estimate
    }
    pub fn estimated(&self) -> bool {
        self.estimated
    }
}

/// One exact subject-static covariate value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubjectCovariateValue {
    subject: String,
    covariate: String,
    value: f64,
}

impl SubjectCovariateValue {
    pub fn subject(&self) -> &str {
        &self.subject
    }
    pub fn covariate(&self) -> &str {
        &self.covariate
    }
    pub fn value(&self) -> f64 {
        self.value
    }
}

/// One subject's design values in canonical effect declaration order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubjectCovariateDesign {
    subject: String,
    values: Vec<f64>,
}

impl SubjectCovariateDesign {
    pub fn subject(&self) -> &str {
        &self.subject
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Subject-specific transformed and execution-space population parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubjectPopulationParameters {
    subject: String,
    phi: Vec<f64>,
    psi: Vec<f64>,
}

impl SubjectPopulationParameters {
    pub fn subject(&self) -> &str {
        &self.subject
    }
    pub fn phi(&self) -> &[f64] {
        &self.phi
    }
    pub fn psi(&self) -> &[f64] {
        &self.psi
    }
}

/// Fully validated subject-static covariate model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovariateModel {
    declarations: Vec<CovariateEffect>,
    estimates: Vec<CovariateEstimate>,
    parameter_indices: Vec<usize>,
    subject_values: Vec<SubjectCovariateValue>,
    subject_design: Vec<SubjectCovariateDesign>,
}

impl CovariateModel {
    pub fn resolve(
        declarations: Vec<CovariateEffect>,
        parameters: &ParameterSpace<UnboundedParameter>,
        data: &Data,
    ) -> Result<Self, CovariateValidationError> {
        // Normalize signed zero even for declarations produced by serde rather
        // than the public constructors. This makes equality, bit-set keys,
        // names, design coding, and persisted source metadata agree.
        let declarations = declarations
            .into_iter()
            .map(|mut effect| {
                effect.kind = match effect.kind {
                    CovariateEffectKind::Continuous { center } => CovariateEffectKind::Continuous {
                        center: canonical_zero(center),
                    },
                    CovariateEffectKind::Categorical { reference, level } => {
                        CovariateEffectKind::Categorical {
                            reference: canonical_zero(reference),
                            level: canonical_zero(level),
                        }
                    }
                };
                effect
            })
            .collect::<Vec<_>>();
        validate_declarations(&declarations, parameters)?;
        let covariate_names: Vec<String> = declarations
            .iter()
            .map(|effect| effect.covariate.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let mut covariate_names = covariate_names;
        covariate_names.sort();

        let mut subject_values = Vec::new();
        let mut subject_maps = Vec::new();
        for subject in data.subjects() {
            let mut values = BTreeMap::new();
            for name in &covariate_names {
                let mut exact: Option<f64> = None;
                for occasion in subject.occasions() {
                    let covariate = occasion.covariates().get_covariate(name).ok_or_else(|| {
                        CovariateValidationError::MissingSubjectCovariate {
                            subject: subject.id().clone(),
                            covariate: name.clone(),
                            occasion: occasion.index(),
                        }
                    })?;
                    let observations = covariate.observations();
                    if observations.is_empty() {
                        return Err(CovariateValidationError::MissingSubjectCovariate {
                            subject: subject.id().clone(),
                            covariate: name.clone(),
                            occasion: occasion.index(),
                        });
                    }
                    for (_, value) in observations {
                        if !value.is_finite() {
                            return Err(CovariateValidationError::NonFiniteSubjectCovariate {
                                subject: subject.id().clone(),
                                covariate: name.clone(),
                            });
                        }
                        if exact.is_some_and(|prior| prior != value) {
                            return Err(CovariateValidationError::TimeVaryingSubjectCovariate {
                                subject: subject.id().clone(),
                                covariate: name.clone(),
                            });
                        }
                        exact = Some(canonical_zero(value));
                    }
                }
                let value =
                    exact.ok_or_else(|| CovariateValidationError::MissingSubjectCovariate {
                        subject: subject.id().clone(),
                        covariate: name.clone(),
                        occasion: 0,
                    })?;
                values.insert(name.clone(), value);
                subject_values.push(SubjectCovariateValue {
                    subject: subject.id().clone(),
                    covariate: name.clone(),
                    value,
                });
            }
            subject_maps.push((subject.id().clone(), values));
        }
        validate_observed_categories(&declarations, &subject_maps)?;

        let mut subject_design = Vec::with_capacity(subject_maps.len());
        for (subject, values) in &subject_maps {
            subject_design.push(SubjectCovariateDesign {
                subject: subject.clone(),
                values: declarations
                    .iter()
                    .map(|effect| effect.design_value(values[effect.covariate()]))
                    .collect(),
            });
        }
        let parameter_indices = declarations
            .iter()
            .map(|effect| {
                parameters
                    .iter()
                    .position(|parameter| parameter.name == effect.parameter)
                    .expect("declaration parameter validated")
            })
            .collect();
        let estimates = declarations
            .iter()
            .enumerate()
            .map(|(index, effect)| CovariateEstimate {
                name: effect.name(),
                declaration_index: index,
                estimate: effect.initial.expect("initial validated"),
                estimated: effect.estimated,
            })
            .collect();
        Ok(Self {
            declarations,
            estimates,
            parameter_indices,
            subject_values,
            subject_design,
        })
    }

    pub fn declarations(&self) -> &[CovariateEffect] {
        &self.declarations
    }
    pub fn estimates(&self) -> &[CovariateEstimate] {
        &self.estimates
    }
    pub fn subject_values(&self) -> &[SubjectCovariateValue] {
        &self.subject_values
    }
    pub fn subject_design(&self) -> &[SubjectCovariateDesign] {
        &self.subject_design
    }
    pub fn parameter_indices(&self) -> &[usize] {
        &self.parameter_indices
    }

    pub(crate) fn validate_initial_gls_rank(
        &self,
        parameters: &ParameterSpace<UnboundedParameter>,
        random_effect_names: &[String],
        omega: &Array2<f64>,
    ) -> Result<(), CovariateValidationError> {
        let random_rows: HashMap<&str, usize> = random_effect_names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let intercepts: Vec<usize> = parameters
            .iter()
            .enumerate()
            .filter_map(|(parameter_index, parameter)| {
                (parameter.estimate && parameter.random_effect).then_some(parameter_index)
            })
            .collect();
        let estimated_effects: Vec<usize> = self
            .declarations
            .iter()
            .enumerate()
            .filter_map(|(index, effect)| {
                (effect.estimated && parameters.items[self.parameter_indices[index]].random_effect)
                    .then_some(index)
            })
            .collect();
        let width = intercepts.len() + estimated_effects.len();
        if width == 0 {
            return Ok(());
        }
        let mut designs = Vec::with_capacity(self.subject_design.len());
        for subject in &self.subject_design {
            let mut design = Array2::<f64>::zeros((random_effect_names.len(), width));
            for (column, parameter_index) in intercepts.iter().copied().enumerate() {
                let parameter = &parameters.items[parameter_index];
                let row = random_rows[parameter.name.as_str()];
                design[[row, column]] = 1.0;
            }
            for (effect_column, effect_index) in estimated_effects.iter().copied().enumerate() {
                let row = random_rows[self.declarations[effect_index].parameter()];
                design[[row, intercepts.len() + effect_column]] = subject.values[effect_index];
            }
            designs.push(design);
        }
        let expected = vec![vec![0.0; random_effect_names.len()]; designs.len()];
        let offsets = expected.clone();
        solve_covariate_gls(CovariateGlsProblem {
            design: &designs,
            expected_phi: &expected,
            offset: &offsets,
            omega,
        })
        .map(|_| ())
        .map_err(|error| CovariateValidationError::SingularDesign {
            detail: error.to_string(),
        })
    }

    pub fn with_estimates(&self, values: &[f64]) -> Result<Self, CovariateMstepError> {
        if values.len() != self.estimates.len() {
            return Err(CovariateMstepError::DimensionMismatch {
                detail: format!(
                    "expected {} coefficients, got {}",
                    self.estimates.len(),
                    values.len()
                ),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CovariateMstepError::NonFiniteSolution);
        }
        let mut updated = self.clone();
        for (estimate, value) in updated.estimates.iter_mut().zip(values) {
            if estimate.estimated {
                estimate.estimate = *value;
            } else if estimate.estimate != *value {
                return Err(CovariateMstepError::FixedCoefficientChanged {
                    name: estimate.name.clone(),
                });
            }
        }
        Ok(updated)
    }

    pub fn subject_population_parameters(
        &self,
        population_phi: &[f64],
        scales: &[ParameterScale],
    ) -> Result<Vec<SubjectPopulationParameters>, CovariateMstepError> {
        if population_phi.len() != scales.len() {
            return Err(CovariateMstepError::DimensionMismatch {
                detail: "population phi and scale widths differ".to_string(),
            });
        }
        let mut rows = Vec::with_capacity(self.subject_design.len());
        for design in &self.subject_design {
            let mut phi = population_phi.to_vec();
            for (effect_index, design_value) in design.values.iter().enumerate() {
                phi[self.parameter_indices[effect_index]] +=
                    design_value * self.estimates[effect_index].estimate;
            }
            let psi: Vec<f64> = phi
                .iter()
                .zip(scales)
                .map(|(value, scale)| phi_to_psi(*value, *scale))
                .collect();
            if phi.iter().chain(&psi).any(|value| !value.is_finite()) {
                return Err(CovariateMstepError::NonFiniteSubjectMean {
                    subject: design.subject.clone(),
                });
            }
            rows.push(SubjectPopulationParameters {
                subject: design.subject.clone(),
                phi,
                psi,
            });
        }
        Ok(rows)
    }
}

fn validate_declarations(
    declarations: &[CovariateEffect],
    parameters: &ParameterSpace<UnboundedParameter>,
) -> Result<(), CovariateValidationError> {
    let by_parameter: HashMap<&str, &UnboundedParameter> = parameters
        .iter()
        .map(|parameter| (parameter.name.as_str(), parameter))
        .collect();
    let mut keys = HashSet::new();
    let mut families: HashMap<&str, CovariateEffectFamily> = HashMap::new();
    for effect in declarations {
        if effect.parameter.is_empty() {
            return Err(CovariateValidationError::UnknownParameter(
                effect.parameter.clone(),
            ));
        }
        if effect.covariate.is_empty() {
            return Err(CovariateValidationError::UnknownCovariate(
                effect.covariate.clone(),
            ));
        }
        if !by_parameter.contains_key(effect.parameter.as_str()) {
            return Err(CovariateValidationError::UnknownParameter(
                effect.parameter.clone(),
            ));
        }
        let initial = effect
            .initial
            .ok_or_else(|| CovariateValidationError::MissingInitial(effect.name()))?;
        if !initial.is_finite() {
            return Err(CovariateValidationError::NonFiniteInitial(effect.name()));
        }
        let family = effect.family();
        match family {
            CovariateEffectFamily::Continuous { center } if !center.is_finite() => {
                return Err(CovariateValidationError::NonFiniteCenter {
                    covariate: effect.covariate.clone(),
                });
            }
            CovariateEffectFamily::Categorical { reference, level }
                if !reference.is_finite() || !level.is_finite() =>
            {
                return Err(CovariateValidationError::NonFiniteCategory {
                    covariate: effect.covariate.clone(),
                });
            }
            CovariateEffectFamily::Categorical { reference, level } if reference == level => {
                return Err(CovariateValidationError::ReferenceLevelCollision {
                    covariate: effect.covariate.clone(),
                    value: reference,
                });
            }
            _ => {}
        }
        if let Some(existing) = families.get(effect.covariate.as_str()) {
            let compatible = match (*existing, family) {
                (
                    CovariateEffectFamily::Continuous { center: left },
                    CovariateEffectFamily::Continuous { center: right },
                ) => left == right,
                (
                    CovariateEffectFamily::Categorical {
                        reference: left, ..
                    },
                    CovariateEffectFamily::Categorical {
                        reference: right, ..
                    },
                ) => left == right,
                _ => false,
            };
            if !compatible {
                return Err(CovariateValidationError::InconsistentFamily {
                    covariate: effect.covariate.clone(),
                });
            }
        } else {
            families.insert(effect.covariate.as_str(), family);
        }
        let level = match effect.kind {
            CovariateEffectKind::Continuous { .. } => None,
            CovariateEffectKind::Categorical { level, .. } => Some(level.to_bits()),
        };
        if !keys.insert((effect.parameter.as_str(), effect.covariate.as_str(), level)) {
            return Err(CovariateValidationError::DuplicateEffect(effect.name()));
        }
    }
    Ok(())
}

fn validate_observed_categories(
    declarations: &[CovariateEffect],
    subjects: &[(String, BTreeMap<String, f64>)],
) -> Result<(), CovariateValidationError> {
    let categorical_covariates: HashSet<&str> = declarations
        .iter()
        .filter_map(|effect| match effect.kind {
            CovariateEffectKind::Categorical { .. } => Some(effect.covariate.as_str()),
            _ => None,
        })
        .collect();
    for covariate in categorical_covariates {
        let Some(reference) = declarations.iter().find_map(|effect| {
            (effect.covariate == covariate)
                .then_some(effect)
                .and_then(|effect| match effect.kind {
                    CovariateEffectKind::Categorical { reference, .. } => Some(reference),
                    _ => None,
                })
        }) else {
            return Err(CovariateValidationError::InconsistentFamily {
                covariate: covariate.to_string(),
            });
        };
        let observed: HashSet<u64> = subjects
            .iter()
            .map(|(_, values)| values[covariate].to_bits())
            .collect();
        let declared_all: HashSet<u64> = declarations
            .iter()
            .filter_map(|effect| match effect.kind {
                CovariateEffectKind::Categorical { level, .. } if effect.covariate == covariate => {
                    Some(level.to_bits())
                }
                _ => None,
            })
            .collect();
        for value in &observed {
            if *value != reference.to_bits() && !declared_all.contains(value) {
                return Err(CovariateValidationError::UnknownCategory {
                    covariate: covariate.to_string(),
                    value: f64::from_bits(*value),
                });
            }
        }
        let targets: HashSet<&str> = declarations
            .iter()
            .filter(|effect| effect.covariate == covariate)
            .map(|effect| effect.parameter.as_str())
            .collect();
        for parameter in targets {
            let levels: HashSet<u64> = declarations
                .iter()
                .filter_map(|effect| match effect.kind {
                    CovariateEffectKind::Categorical { level, .. }
                        if effect.covariate == covariate && effect.parameter == parameter =>
                    {
                        Some(level.to_bits())
                    }
                    _ => None,
                })
                .collect();
            for value in &observed {
                if *value != reference.to_bits() && !levels.contains(value) {
                    return Err(CovariateValidationError::IncompleteCategoricalLevels {
                        parameter: parameter.to_string(),
                        covariate: covariate.to_string(),
                        level: f64::from_bits(*value),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Exact joint GLS input. `design[i]` is `A_i`, `expected_phi[i]` is `Ephi_i`,
/// and `offset[i]` contains all fixed-coordinate contributions.
#[derive(Debug, Clone)]
pub struct CovariateGlsProblem<'a> {
    pub design: &'a [Array2<f64>],
    pub expected_phi: &'a [Vec<f64>],
    pub offset: &'a [Vec<f64>],
    pub omega: &'a Array2<f64>,
}

/// Strict finite Cholesky GLS with no tolerance, repair, or generalized inverse.
pub fn solve_covariate_gls(
    problem: CovariateGlsProblem<'_>,
) -> Result<Vec<f64>, CovariateMstepError> {
    let n = problem.design.len();
    if n == 0 || problem.expected_phi.len() != n || problem.offset.len() != n {
        return Err(CovariateMstepError::DimensionMismatch {
            detail: "subject GLS inputs differ".to_string(),
        });
    }
    let q = problem.omega.nrows();
    if q == 0 || problem.omega.ncols() != q {
        return Err(CovariateMstepError::DimensionMismatch {
            detail: "Omega must be nonempty and square".to_string(),
        });
    }
    let p = problem.design[0].ncols();
    if p == 0 {
        return Ok(Vec::new());
    }
    let omega_lower = cholesky_lower(problem.omega)
        .map_err(|error| CovariateMstepError::InvalidOmega(error.to_string()))?;
    let mut h = Array2::<f64>::zeros((p, p));
    let mut g = vec![0.0; p];
    for subject in 0..n {
        let a = &problem.design[subject];
        if a.nrows() != q
            || a.ncols() != p
            || problem.expected_phi[subject].len() != q
            || problem.offset[subject].len() != q
        {
            return Err(CovariateMstepError::DimensionMismatch {
                detail: format!("subject {subject} GLS width differs"),
            });
        }
        let d: Vec<f64> = problem.expected_phi[subject]
            .iter()
            .zip(&problem.offset[subject])
            .map(|(mean, offset)| mean - offset)
            .collect();
        let wd = solve_spd_from_lower(&omega_lower, &d)?;
        let mut wa = Array2::<f64>::zeros((q, p));
        for column in 0..p {
            let rhs: Vec<f64> = (0..q).map(|row| a[[row, column]]).collect();
            let solved = solve_spd_from_lower(&omega_lower, &rhs)?;
            for row in 0..q {
                wa[[row, column]] = solved[row];
            }
        }
        for left in 0..p {
            g[left] += (0..q).map(|row| a[[row, left]] * wd[row]).sum::<f64>();
            for right in 0..=left {
                h[[left, right]] += (0..q)
                    .map(|row| a[[row, left]] * wa[[row, right]])
                    .sum::<f64>();
            }
        }
    }
    for left in 0..p {
        for right in 0..left {
            h[[right, left]] = h[[left, right]];
        }
    }
    let lower = cholesky_lower(&h).map_err(|_| CovariateMstepError::SingularDesign)?;
    let solution = solve_spd_from_lower(&lower, &g)?;
    if solution.iter().any(|value| !value.is_finite()) {
        return Err(CovariateMstepError::NonFiniteSolution);
    }
    Ok(solution)
}

fn solve_spd_from_lower(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, CovariateMstepError> {
    if lower.len() != rhs.len() || lower.iter().any(|row| row.len() != rhs.len()) {
        return Err(CovariateMstepError::DimensionMismatch {
            detail: "Cholesky solve width differs".to_string(),
        });
    }
    let n = rhs.len();
    let mut y = vec![0.0; n];
    for row in 0..n {
        let residual = rhs[row] - (0..row).map(|col| lower[row][col] * y[col]).sum::<f64>();
        y[row] = residual / lower[row][row];
        if !y[row].is_finite() {
            return Err(CovariateMstepError::NonFiniteSolution);
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let residual = y[row]
            - ((row + 1)..n)
                .map(|col| lower[col][row] * x[col])
                .sum::<f64>();
        x[row] = residual / lower[row][row];
        if !x[row].is_finite() {
            return Err(CovariateMstepError::NonFiniteSolution);
        }
    }
    Ok(x)
}

/// Rebase one eta vector while preserving absolute transformed parameters.
pub fn rebase_eta(
    eta: &mut [f64],
    old_mu: &[f64],
    new_mu: &[f64],
) -> Result<(), CovariateMstepError> {
    if eta.len() != old_mu.len() || eta.len() != new_mu.len() {
        return Err(CovariateMstepError::DimensionMismatch {
            detail: "eta and subject means differ".to_string(),
        });
    }
    for index in 0..eta.len() {
        eta[index] -= new_mu[index] - old_mu[index];
        if !eta[index].is_finite() {
            return Err(CovariateMstepError::NonFiniteSolution);
        }
    }
    Ok(())
}

/// Subject-specific raw covariance candidate from SA-updated moments.
pub fn subject_centered_omega(
    global_second_moment: &Array2<f64>,
    expected_phi: &[Vec<f64>],
    subject_mu: &[Vec<f64>],
) -> Result<Array2<f64>, CovariateMstepError> {
    let q = global_second_moment.nrows();
    if q == 0
        || global_second_moment.ncols() != q
        || expected_phi.is_empty()
        || expected_phi.len() != subject_mu.len()
    {
        return Err(CovariateMstepError::DimensionMismatch {
            detail: "Omega moment inputs differ".to_string(),
        });
    }
    let mut candidate = global_second_moment.clone();
    let n = expected_phi.len() as f64;
    for (mean, mu) in expected_phi.iter().zip(subject_mu) {
        if mean.len() != q || mu.len() != q {
            return Err(CovariateMstepError::DimensionMismatch {
                detail: "subject mean width differs".to_string(),
            });
        }
        for row in 0..q {
            for column in 0..q {
                candidate[[row, column]] +=
                    (-mean[row] * mu[column] - mu[row] * mean[column] + mu[row] * mu[column]) / n;
            }
        }
    }
    if candidate.iter().any(|value| !value.is_finite()) {
        return Err(CovariateMstepError::NonFiniteOmegaCandidate);
    }
    Ok(candidate)
}

#[derive(Debug, Clone, PartialEq, Error, Serialize, Deserialize)]
#[serde(tag = "failure", rename_all = "snake_case")]
pub enum CovariateValidationError {
    #[error("unknown parameter '{0}' in covariate effect")]
    UnknownParameter(String),
    #[error("unknown or empty covariate '{0}'")]
    UnknownCovariate(String),
    #[error("covariate effect '{0}' has no initial coefficient")]
    MissingInitial(String),
    #[error("covariate effect '{0}' has a nonfinite initial coefficient")]
    NonFiniteInitial(String),
    #[error("continuous covariate '{covariate}' has a nonfinite center")]
    NonFiniteCenter { covariate: String },
    #[error("categorical covariate '{covariate}' has a nonfinite reference or level")]
    NonFiniteCategory { covariate: String },
    #[error("categorical covariate '{covariate}' reference collides with level {value}")]
    ReferenceLevelCollision { covariate: String, value: f64 },
    #[error("covariate '{covariate}' has inconsistent family, center, or reference declarations")]
    InconsistentFamily { covariate: String },
    #[error("duplicate covariate effect '{0}'")]
    DuplicateEffect(String),
    #[error("subject '{subject}' is missing covariate '{covariate}' in occasion {occasion}")]
    MissingSubjectCovariate {
        subject: String,
        covariate: String,
        occasion: usize,
    },
    #[error("subject '{subject}' covariate '{covariate}' is nonfinite")]
    NonFiniteSubjectCovariate { subject: String, covariate: String },
    #[error("subject '{subject}' covariate '{covariate}' is time-varying")]
    TimeVaryingSubjectCovariate { subject: String, covariate: String },
    #[error("covariate '{covariate}' has undeclared observed category {value}")]
    UnknownCategory { covariate: String, value: f64 },
    #[error("parameter '{parameter}' does not declare observed level {level} for covariate '{covariate}'")]
    IncompleteCategoricalLevels {
        parameter: String,
        covariate: String,
        level: f64,
    },
    #[error("covariate GLS design is not strict full rank: {detail}")]
    SingularDesign { detail: String },
    #[error("unsupported nonlinear parametric constraint: {description}")]
    UnsupportedNonlinearConstraint { description: String },
}

#[derive(Debug, Clone, PartialEq, Error, Serialize, Deserialize)]
#[serde(tag = "failure", rename_all = "snake_case")]
pub enum CovariateMstepError {
    #[error("covariate M-step dimension mismatch: {detail}")]
    DimensionMismatch { detail: String },
    #[error("invalid accepted Omega for covariate GLS: {0}")]
    InvalidOmega(String),
    #[error("covariate GLS design is singular or collinear")]
    SingularDesign,
    #[error("covariate GLS produced a nonfinite solution")]
    NonFiniteSolution,
    #[error("fixed coefficient '{name}' changed")]
    FixedCoefficientChanged { name: String },
    #[error("subject '{subject}' has a nonfinite population mean")]
    NonFiniteSubjectMean { subject: String },
    #[error("subject-centered Omega candidate is nonfinite")]
    NonFiniteOmegaCandidate,
}

pub fn reject_constraints(
    constraints: &[ParametricConstraint],
) -> Result<(), CovariateValidationError> {
    if let Some(constraint) = constraints.first() {
        return Err(CovariateValidationError::UnsupportedNonlinearConstraint {
            description: constraint.description().to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_names_and_design_are_stable() {
        let continuous = CovariateEffect::continuous("cl", "wt", 70.0).with_initial(0.0);
        let categorical = CovariateEffect::categorical("v", "sex", 0.0, 2.0).with_initial(0.1);
        assert_eq!(continuous.name(), "beta:cl:wt");
        assert_eq!(categorical.name(), "beta:v:sex:2");
        assert_eq!(continuous.design_value(82.0), 12.0);
        assert_eq!(categorical.design_value(2.0), 1.0);
        assert_eq!(categorical.design_value(0.0), 0.0);
    }

    #[test]
    fn signed_zero_is_canonical_before_names_bits_and_design() {
        let positive = CovariateEffect::categorical("v", "group", 1.0, 0.0);
        let negative = CovariateEffect::categorical("v", "group", 1.0, -0.0);
        assert_eq!(positive, negative);
        assert_eq!(negative.name(), "beta:v:group:0");
        assert_eq!(negative.design_value(0.0), 1.0);
        assert_eq!(negative.design_value(-0.0), 1.0);
        match negative.family() {
            CovariateEffectFamily::Categorical { reference, level } => {
                assert_eq!(reference.to_bits(), 1.0f64.to_bits());
                assert_eq!(level.to_bits(), 0.0f64.to_bits());
            }
            _ => panic!("expected categorical effect"),
        }

        let collision = CovariateEffect::categorical("v", "group", -0.0, 0.0);
        match collision.family() {
            CovariateEffectFamily::Categorical { reference, level } => {
                assert_eq!(reference.to_bits(), level.to_bits());
            }
            _ => panic!("expected categorical effect"),
        }
        let centered = CovariateEffect::continuous("v", "wt", -0.0);
        match centered.family() {
            CovariateEffectFamily::Continuous { center } => {
                assert_eq!(center.to_bits(), 0.0f64.to_bits());
            }
            _ => panic!("expected continuous effect"),
        }
    }

    #[test]
    fn correlated_gls_with_fixed_offsets_is_exact() {
        let omega = Array2::from_shape_vec((2, 2), vec![2.0, 0.6, 0.6, 1.0]).unwrap();
        let design = vec![
            Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 0.0, 1.0]).unwrap(),
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 0.0, -0.5]).unwrap(),
            Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.0, 2.0]).unwrap(),
        ];
        let truth = [0.7, -0.2];
        let offsets = vec![vec![0.1, 1.5]; 3];
        let expected_phi: Vec<Vec<f64>> = design
            .iter()
            .map(|a| {
                (0..2)
                    .map(|row| {
                        offsets[0][row]
                            + (0..2)
                                .map(|column| a[[row, column]] * truth[column])
                                .sum::<f64>()
                    })
                    .collect()
            })
            .collect();
        let estimate = solve_covariate_gls(CovariateGlsProblem {
            design: &design,
            expected_phi: &expected_phi,
            offset: &offsets,
            omega: &omega,
        })
        .unwrap();
        assert!((estimate[0] - truth[0]).abs() <= 1e-10);
        assert!((estimate[1] - truth[1]).abs() <= 1e-10);
    }

    #[test]
    fn singular_design_fails_without_repair() {
        let omega = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let design = vec![Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap()];
        let means = vec![vec![2.0]];
        let offsets = vec![vec![0.0]];
        assert_eq!(
            solve_covariate_gls(CovariateGlsProblem {
                design: &design,
                expected_phi: &means,
                offset: &offsets,
                omega: &omega
            }),
            Err(CovariateMstepError::SingularDesign)
        );
    }

    #[test]
    fn eta_rebase_preserves_absolute_phi() {
        let old_mu = [1.0, -2.0];
        let new_mu = [1.25, -2.5];
        let mut eta = [0.4, 0.8];
        let absolute = [old_mu[0] + eta[0], old_mu[1] + eta[1]];
        rebase_eta(&mut eta, &old_mu, &new_mu).unwrap();
        assert!((new_mu[0] + eta[0] - absolute[0]).abs() <= 1e-10);
        assert!((new_mu[1] + eta[1] - absolute[1]).abs() <= 1e-10);
    }

    #[test]
    fn subject_specific_omega_formula_is_exact() {
        let m2 = Array2::from_shape_vec((1, 1), vec![10.0]).unwrap();
        let means = vec![vec![2.0], vec![4.0]];
        let mu = vec![vec![1.0], vec![3.0]];
        let omega = subject_centered_omega(&m2, &means, &mu).unwrap();
        let expected = 10.0 + (-2.0 - 2.0 + 1.0 - 12.0 - 12.0 + 9.0) / 2.0;
        assert!((omega[[0, 0]] - expected).abs() <= 1e-10);
    }
}
