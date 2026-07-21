use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use ndarray::Array2;

use crate::model::{ParameterSpace, UnboundedParameter};
use crate::results::{CovarianceTrialRejectionReason, CovarianceUpdateRejectionReason};

use super::{
    covariance::{cholesky_log_determinant, cholesky_lower, identity_matrix},
    covariates::CovariateModel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OmegaEntryUnit {
    VarianceOrCovariance,
    StandardDeviation,
}

#[derive(Debug, Clone, PartialEq)]
struct OmegaEntry {
    left: String,
    right: String,
    value: f64,
    unit: OmegaEntryUnit,
    estimated: bool,
}

/// Named declaration of the initial IIV covariance matrix.
///
/// Variances and covariances omitted from an explicit declaration are
/// structural zeros. Fixed entries remain part of Ω but are not updated by an
/// estimation algorithm. This preserves the distinction between covariance
/// structure and structural/free and fixed masks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Omega {
    entries: Vec<OmegaEntry>,
}

impl Omega {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an estimated diagonal Ω declaration from variances.
    ///
    /// This compatibility constructor retains its original variance semantics.
    pub fn diagonal<N, I>(variances: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, f64)>,
    {
        Self::diagonal_variances(variances)
    }

    /// Builds an estimated diagonal Ω declaration from variances.
    pub fn diagonal_variances<N, I>(variances: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, f64)>,
    {
        variances
            .into_iter()
            .fold(Self::new(), |omega, (name, variance)| {
                omega.variance(name, variance)
            })
    }

    /// Builds an estimated diagonal Ω declaration from standard deviations.
    ///
    /// SD domain validation and conversion to variances occur during final
    /// problem construction, preserving fail-closed builder semantics.
    pub fn diagonal_standard_deviations<N, I>(standard_deviations: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, f64)>,
    {
        standard_deviations.into_iter().fold(
            Self::new(),
            |mut omega, (name, standard_deviation)| {
                let name = name.into();
                omega.entries.push(OmegaEntry {
                    left: name.clone(),
                    right: name,
                    value: standard_deviation,
                    unit: OmegaEntryUnit::StandardDeviation,
                    estimated: true,
                });
                omega
            },
        )
    }

    /// Declares an estimated variance for one random effect.
    pub fn variance(mut self, name: impl Into<String>, value: f64) -> Self {
        let name = name.into();
        self.entries.push(OmegaEntry {
            left: name.clone(),
            right: name,
            value,
            unit: OmegaEntryUnit::VarianceOrCovariance,
            estimated: true,
        });
        self
    }

    /// Declares a fixed variance for one random effect.
    pub fn fixed_variance(mut self, name: impl Into<String>, value: f64) -> Self {
        let name = name.into();
        self.entries.push(OmegaEntry {
            left: name.clone(),
            right: name,
            value,
            unit: OmegaEntryUnit::VarianceOrCovariance,
            estimated: false,
        });
        self
    }

    /// Declares an estimated covariance. Undeclared covariances remain
    /// structural zeros.
    pub fn covariance(
        mut self,
        left: impl Into<String>,
        right: impl Into<String>,
        value: f64,
    ) -> Self {
        self.entries.push(OmegaEntry {
            left: left.into(),
            right: right.into(),
            value,
            unit: OmegaEntryUnit::VarianceOrCovariance,
            estimated: true,
        });
        self
    }

    /// Declares a fixed covariance.
    pub fn fixed_covariance(
        mut self,
        left: impl Into<String>,
        right: impl Into<String>,
        value: f64,
    ) -> Self {
        self.entries.push(OmegaEntry {
            left: left.into(),
            right: right.into(),
            value,
            unit: OmegaEntryUnit::VarianceOrCovariance,
            estimated: false,
        });
        self
    }
}

/// Named declaration of the initial inter-occasion covariance matrix.
///
/// κ is additive in transformed φ-space, independently for every occasion of
/// a subject. Its names refer to model parameters, not to IIV membership: a
/// parameter may have IOV with or without IIV.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Iov {
    omega: Omega,
}

impl Iov {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an estimated diagonal IOV declaration from variances.
    ///
    /// This compatibility constructor retains its original variance semantics.
    pub fn diagonal<N, I>(variances: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, f64)>,
    {
        Self::diagonal_variances(variances)
    }

    /// Builds an estimated diagonal IOV declaration from variances.
    pub fn diagonal_variances<N, I>(variances: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, f64)>,
    {
        Self {
            omega: Omega::diagonal_variances(variances),
        }
    }

    /// Builds an estimated diagonal IOV declaration from standard deviations.
    pub fn diagonal_standard_deviations<N, I>(standard_deviations: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, f64)>,
    {
        Self {
            omega: Omega::diagonal_standard_deviations(standard_deviations),
        }
    }

    pub fn variance(mut self, name: impl Into<String>, value: f64) -> Self {
        self.omega = self.omega.variance(name, value);
        self
    }

    pub fn fixed_variance(mut self, name: impl Into<String>, value: f64) -> Self {
        self.omega = self.omega.fixed_variance(name, value);
        self
    }

    pub fn covariance(
        mut self,
        left: impl Into<String>,
        right: impl Into<String>,
        value: f64,
    ) -> Self {
        self.omega = self.omega.covariance(left, right, value);
        self
    }

    pub fn fixed_covariance(
        mut self,
        left: impl Into<String>,
        right: impl Into<String>,
        value: f64,
    ) -> Self {
        self.omega = self.omega.fixed_covariance(left, right, value);
        self
    }
}

/// Initial population, IIV, and optional IOV distribution shared by all
/// parametric algorithms.
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricPrior {
    parameters: ParameterSpace<UnboundedParameter>,
    omega: ResolvedOmega,
    iov: Option<ResolvedIov>,
    covariates: Option<CovariateModel>,
}

impl ParametricPrior {
    #[cfg(test)]
    pub(crate) fn new(
        parameters: ParameterSpace<UnboundedParameter>,
        omega: Option<Omega>,
        iov: Option<Iov>,
    ) -> Result<Self> {
        Self::new_with_covariates(parameters, omega, iov, None)
    }

    pub(crate) fn new_with_covariates(
        parameters: ParameterSpace<UnboundedParameter>,
        omega: Option<Omega>,
        iov: Option<Iov>,
        covariates: Option<CovariateModel>,
    ) -> Result<Self> {
        let resolved = ResolvedOmega::resolve(&parameters, omega.as_ref())?;
        let iov = iov
            .as_ref()
            .map(|declaration| ResolvedIov::resolve(&parameters, declaration))
            .transpose()?;
        Ok(Self {
            parameters,
            omega: resolved,
            iov,
            covariates,
        })
    }

    pub fn parameters(&self) -> &ParameterSpace<UnboundedParameter> {
        &self.parameters
    }

    /// Random-effect names in η/Ω order.
    pub fn random_effect_names(&self) -> &[String] {
        &self.omega.names
    }

    /// Initial IIV covariance matrix.
    pub fn omega(&self) -> &Array2<f64> {
        &self.omega.initial
    }

    pub(crate) fn resolved_omega(&self) -> &ResolvedOmega {
        &self.omega
    }

    pub fn iov_effect_names(&self) -> Option<&[String]> {
        self.iov.as_ref().map(|iov| iov.omega.names.as_slice())
    }

    pub fn omega_iov(&self) -> Option<&Array2<f64>> {
        self.iov.as_ref().map(|iov| iov.omega.initial())
    }

    pub(crate) fn resolved_iov(&self) -> Option<&ResolvedIov> {
        self.iov.as_ref()
    }

    /// Fully validated subject-static covariate population model, when declared.
    pub fn covariates(&self) -> Option<&CovariateModel> {
        self.covariates.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedIov {
    parameter_indices: Vec<usize>,
    omega: ResolvedOmega,
}

impl ResolvedIov {
    fn resolve(parameters: &ParameterSpace<UnboundedParameter>, declaration: &Iov) -> Result<Self> {
        let mut names = Vec::new();
        for entry in &declaration.omega.entries {
            for name in [&entry.left, &entry.right] {
                if !names.iter().any(|existing| existing == name) {
                    names.push(name.clone());
                }
            }
        }
        if names.is_empty() {
            bail!("IOV declaration must contain at least one variance");
        }
        let parameter_indices = names
            .iter()
            .map(|name| {
                parameters
                    .iter()
                    .position(|parameter| parameter.name == *name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("IOV entry references unknown model parameter '{name}'")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let omega = ResolvedOmega::resolve_names(&names, &declaration.omega, "IOV")?;
        Ok(Self {
            parameter_indices,
            omega,
        })
    }

    pub(crate) fn parameter_indices(&self) -> &[usize] {
        &self.parameter_indices
    }

    pub(crate) fn omega(&self) -> &ResolvedOmega {
        &self.omega
    }
}

// ── Connected-component classification ────────────────────────────

/// Connected component of the declared potentially-nonzero graph.
#[derive(Debug, Clone)]
struct OmegaComponent {
    kind: OmegaComponentKind,
    indices: Vec<usize>,
    free_coords: Vec<(usize, usize)>,
    fixed_coords: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OmegaComponentKind {
    /// Every declared entry is fixed.
    AllFixed,
    /// Every declared entry is free and all cross-covariances are declared.
    DenseAllFree,
    /// Every declared entry is free but not all cross-covariances are
    /// declared (sparse structural mask).
    SparseAllFree,
    /// Both free and fixed entries coexist.
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CovarianceUpdateStatus {
    Accepted,
    NoOp,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VarianceFloorMode {
    LegacyAfterInterpolation,
    CappedSolvedTarget,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CovarianceUpdateResult {
    pub(crate) matrix: Array2<f64>,
    pub(crate) status: CovarianceUpdateStatus,
    pub(crate) solved_target: Option<Array2<f64>>,
    pub(crate) accepted_fraction: Option<f64>,
    pub(crate) attempted_fractions: Vec<f64>,
    pub(crate) trial_rejections: Vec<CovarianceTrialRejectionReason>,
    pub(crate) rejection_reason: Option<CovarianceUpdateRejectionReason>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedOmega {
    names: Vec<String>,
    initial: Array2<f64>,
    structural_mask: Array2<bool>,
    estimated_mask: Array2<bool>,
}

fn resolve_entry_value(entry: &OmegaEntry, domain: &str) -> Result<f64> {
    match entry.unit {
        OmegaEntryUnit::VarianceOrCovariance => {
            if !entry.value.is_finite() {
                bail!(
                    "{domain} entry for '{}' and '{}' must be finite",
                    entry.left,
                    entry.right
                );
            }
            Ok(entry.value)
        }
        OmegaEntryUnit::StandardDeviation => {
            if entry.left != entry.right {
                bail!("{domain} standard-deviation declarations must be diagonal");
            }
            if !entry.value.is_finite() || entry.value <= 0.0 {
                bail!(
                    "{domain} standard deviation for '{}' must be finite and strictly positive",
                    entry.left
                );
            }
            if entry.value > f64::MAX / entry.value {
                bail!(
                    "{domain} standard deviation for '{}' would overflow its variance representation",
                    entry.left
                );
            }
            let variance = entry.value * entry.value;
            if !variance.is_finite() || variance <= 0.0 {
                bail!(
                    "{domain} standard deviation for '{}' cannot be represented as a finite positive variance",
                    entry.left
                );
            }
            Ok(variance)
        }
    }
}

impl ResolvedOmega {
    fn resolve(
        parameters: &ParameterSpace<UnboundedParameter>,
        declaration: Option<&Omega>,
    ) -> Result<Self> {
        let names = parameters
            .iter()
            .filter(|parameter| parameter.random_effect)
            .map(|parameter| parameter.name.clone())
            .collect::<Vec<_>>();
        let indices = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect::<HashMap<_, _>>();
        let size = names.len();

        let Some(declaration) = declaration else {
            let initial = identity_matrix(size);
            let structural_mask = Array2::from_shape_fn((size, size), |(row, col)| row == col);
            return Ok(Self {
                names,
                initial,
                estimated_mask: structural_mask.clone(),
                structural_mask,
            });
        };

        let mut initial = Array2::zeros((size, size));
        let mut structural_mask = Array2::from_elem((size, size), false);
        let mut estimated_mask = Array2::from_elem((size, size), false);
        let mut declared = HashSet::new();

        for entry in &declaration.entries {
            let Some(&left) = indices.get(entry.left.as_str()) else {
                bail!(
                    "omega entry references '{}' which is not a declared IIV random effect",
                    entry.left
                );
            };
            let Some(&right) = indices.get(entry.right.as_str()) else {
                bail!(
                    "omega entry references '{}' which is not a declared IIV random effect",
                    entry.right
                );
            };
            let value = resolve_entry_value(entry, "omega")?;
            if left == right && value <= 0.0 {
                bail!("omega variance for '{}' must be positive", entry.left);
            }

            let key = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            if !declared.insert(key) {
                bail!(
                    "omega entry for '{}' and '{}' is declared more than once",
                    entry.left,
                    entry.right
                );
            }

            initial[[left, right]] = value;
            initial[[right, left]] = value;
            structural_mask[[left, right]] = true;
            structural_mask[[right, left]] = true;
            estimated_mask[[left, right]] = entry.estimated;
            estimated_mask[[right, left]] = entry.estimated;
        }

        for (index, name) in names.iter().enumerate() {
            if !structural_mask[[index, index]] {
                bail!("omega variance for random effect '{name}' is not declared");
            }
        }
        if size > 0 {
            cholesky_lower(&initial)?;
        }

        Ok(Self {
            names,
            initial,
            structural_mask,
            estimated_mask,
        })
    }

    fn resolve_names(names: &[String], declaration: &Omega, domain: &str) -> Result<Self> {
        let indices = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect::<HashMap<_, _>>();
        let size = names.len();
        let mut initial = Array2::zeros((size, size));
        let mut structural_mask = Array2::from_elem((size, size), false);
        let mut estimated_mask = Array2::from_elem((size, size), false);
        let mut declared = HashSet::new();

        for entry in &declaration.entries {
            let Some(&left) = indices.get(entry.left.as_str()) else {
                bail!("{domain} entry references unknown effect '{}'", entry.left);
            };
            let Some(&right) = indices.get(entry.right.as_str()) else {
                bail!("{domain} entry references unknown effect '{}'", entry.right);
            };
            let value = resolve_entry_value(entry, domain)?;
            if left == right && value <= 0.0 {
                bail!("{domain} variance for '{}' must be positive", entry.left);
            }
            let key = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            if !declared.insert(key) {
                bail!(
                    "{domain} entry for '{}' and '{}' is declared more than once",
                    entry.left,
                    entry.right
                );
            }
            initial[[left, right]] = value;
            initial[[right, left]] = value;
            structural_mask[[left, right]] = true;
            structural_mask[[right, left]] = true;
            estimated_mask[[left, right]] = entry.estimated;
            estimated_mask[[right, left]] = entry.estimated;
        }

        for (index, name) in names.iter().enumerate() {
            if !structural_mask[[index, index]] {
                bail!("{domain} variance for effect '{name}' is not declared");
            }
        }
        cholesky_lower(&initial)?;
        Ok(Self {
            names: names.to_vec(),
            initial,
            structural_mask,
            estimated_mask,
        })
    }

    pub(crate) fn names(&self) -> &[String] {
        &self.names
    }

    pub(crate) fn initial(&self) -> &Array2<f64> {
        &self.initial
    }

    pub(crate) fn update_with_status(
        &self,
        current: &Array2<f64>,
        candidate: &Array2<f64>,
        minimum_variance: f64,
    ) -> Result<CovarianceUpdateResult> {
        self.update_with_status_and_floor_mode(
            current,
            candidate,
            minimum_variance,
            1.0,
            VarianceFloorMode::LegacyAfterInterpolation,
        )
    }

    /// Perform a mask-aware covariance GEM update whose accepted displacement
    /// is at most `maximum_fraction` of the solved target displacement.
    ///
    /// This under-relaxes the accepted covariance iterate; it does not alter
    /// the stochastic-approximation history used to construct `candidate`.
    pub(crate) fn update_with_status_and_max_fraction(
        &self,
        current: &Array2<f64>,
        candidate: &Array2<f64>,
        minimum_variance: f64,
        maximum_fraction: f64,
    ) -> Result<CovarianceUpdateResult> {
        self.update_with_status_and_floor_mode(
            current,
            candidate,
            minimum_variance,
            maximum_fraction,
            VarianceFloorMode::CappedSolvedTarget,
        )
    }

    fn update_with_status_and_floor_mode(
        &self,
        current: &Array2<f64>,
        candidate: &Array2<f64>,
        minimum_variance: f64,
        maximum_fraction: f64,
        floor_mode: VarianceFloorMode,
    ) -> Result<CovarianceUpdateResult> {
        let dimensions = (self.names.len(), self.names.len());
        if current.dim() != dimensions || candidate.dim() != dimensions {
            bail!("omega update dimensions do not match the declared random effects");
        }
        if !minimum_variance.is_finite() || minimum_variance < 0.0 {
            bail!("omega minimum variance must be finite and non-negative");
        }
        if !maximum_fraction.is_finite() || maximum_fraction <= 0.0 || maximum_fraction > 1.0 {
            bail!("omega update maximum fraction must be finite and in (0, 1]");
        }
        if cholesky_lower(current).is_err() {
            bail!("omega update cannot retain a positive-definite current matrix");
        }
        if !self.has_estimated_entries() {
            return Ok(CovarianceUpdateResult {
                matrix: current.clone(),
                status: CovarianceUpdateStatus::NoOp,
                solved_target: None,
                accepted_fraction: None,
                attempted_fractions: Vec::new(),
                trial_rejections: Vec::new(),
                rejection_reason: None,
            });
        }

        let components = self.classify_components();
        let n = self.names.len();
        // Cross-component second moments cannot affect a block-diagonal
        // covariance objective. Preserve the original finite path exactly, but
        // sanitize malformed values that occur only in those irrelevant slots.
        let mut relevant_candidate = Array2::zeros((n, n));
        for component in &components {
            if component.free_coords.is_empty() {
                continue;
            }
            for &row in &component.indices {
                for &col in &component.indices {
                    relevant_candidate[[row, col]] = candidate[[row, col]];
                }
            }
        }
        let candidate = if validate_finite_symmetric(candidate, "omega second moment").is_ok() {
            candidate
        } else if validate_finite_symmetric(&relevant_candidate, "omega relevant second moment")
            .is_ok()
        {
            &relevant_candidate
        } else {
            return Ok(CovarianceUpdateResult {
                matrix: current.clone(),
                status: CovarianceUpdateStatus::Rejected,
                solved_target: None,
                accepted_fraction: None,
                attempted_fractions: Vec::new(),
                trial_rejections: Vec::new(),
                rejection_reason: Some(
                    CovarianceUpdateRejectionReason::CandidateNotFiniteSymmetric,
                ),
            });
        };
        let Ok(current_objective) = covariance_objective(current, candidate) else {
            return Ok(CovarianceUpdateResult {
                matrix: current.clone(),
                status: CovarianceUpdateStatus::Rejected,
                solved_target: None,
                accepted_fraction: None,
                attempted_fractions: Vec::new(),
                trial_rejections: Vec::new(),
                rejection_reason: Some(
                    CovarianceUpdateRejectionReason::CurrentObjectiveUnavailable,
                ),
            });
        };

        // Solve each connected declared component independently. Dense all-free
        // blocks use S as the exact unconstrained target before the outer
        // floor, interpolation, SPD, and objective checks. Sparse all-free and
        // mixed blocks use the deterministic constrained local GEM solver.
        // Structural zeros between components never enter either solve.
        let mut target = Array2::zeros((n, n));
        // Carry accepted component solutions only to remove irrelevant
        // cross-component floating-point work from later component solves.
        let mut gem_working = current.clone();
        for component in &components {
            match component.kind {
                OmegaComponentKind::AllFixed => {
                    copy_component_coordinates(&mut target, &self.initial, &component.fixed_coords);
                }
                OmegaComponentKind::DenseAllFree => {
                    copy_component_coordinates(&mut target, candidate, &component.free_coords);
                }
                OmegaComponentKind::SparseAllFree | OmegaComponentKind::Mixed => {
                    let Ok((gem_result, _)) =
                        self.local_gem_minimizer(&gem_working, candidate, &component.free_coords)
                    else {
                        return Ok(CovarianceUpdateResult {
                            matrix: current.clone(),
                            status: CovarianceUpdateStatus::Rejected,
                            solved_target: None,
                            accepted_fraction: None,
                            attempted_fractions: Vec::new(),
                            trial_rejections: Vec::new(),
                            rejection_reason: Some(
                                CovarianceUpdateRejectionReason::ConstrainedSolveFailed,
                            ),
                        });
                    };
                    copy_component_coordinates(&mut target, &gem_result, &component.free_coords);
                    copy_component_coordinates(&mut target, &self.initial, &component.fixed_coords);
                    copy_component_coordinates(
                        &mut gem_working,
                        &gem_result,
                        &component.free_coords,
                    );
                }
            }
        }

        // Capped covariate exploration applies the variance floor to the solved
        // target before interpolation, so the floor cannot bypass the requested
        // displacement fraction. The uncapped path retains the established
        // floor-after-interpolation order used by non-covariate IIV and IOV.
        // Every accepted trial must be strictly SPD and must not increase the
        // covariance objective beyond the explicit matrix-arithmetic roundoff
        // allowance in `objective_nonincrease`.
        if floor_mode == VarianceFloorMode::CappedSolvedTarget {
            for index in 0..n {
                if self.estimated_mask[[index, index]] {
                    target[[index, index]] = target[[index, index]].max(minimum_variance);
                }
            }
        }
        let mut attempted_fractions = Vec::with_capacity(16);
        let mut trial_rejections = Vec::with_capacity(16);
        for attempt in 0..16 {
            let fraction = maximum_fraction * 0.5_f64.powi(attempt);
            attempted_fractions.push(fraction);
            let mut updated = current.clone();
            for row in 0..n {
                for col in 0..n {
                    if !self.structural_mask[[row, col]] {
                        updated[[row, col]] = 0.0;
                    } else if self.estimated_mask[[row, col]] {
                        updated[[row, col]] = if floor_mode == VarianceFloorMode::CappedSolvedTarget
                            && fraction == 1.0
                        {
                            target[[row, col]]
                        } else {
                            current[[row, col]]
                                + fraction * (target[[row, col]] - current[[row, col]])
                        };
                    } else {
                        updated[[row, col]] = self.initial[[row, col]];
                    }
                }
            }
            match floor_mode {
                VarianceFloorMode::LegacyAfterInterpolation => {
                    for index in 0..n {
                        if self.estimated_mask[[index, index]] {
                            updated[[index, index]] = updated[[index, index]].max(minimum_variance);
                        }
                    }
                }
                VarianceFloorMode::CappedSolvedTarget
                    if (0..n).any(|index| {
                        self.estimated_mask[[index, index]]
                            && updated[[index, index]] < minimum_variance
                    }) =>
                {
                    trial_rejections.push(CovarianceTrialRejectionReason::VarianceFloorInfeasible);
                    continue;
                }
                VarianceFloorMode::CappedSolvedTarget => {}
            }
            if cholesky_lower(&updated).is_err() {
                trial_rejections.push(CovarianceTrialRejectionReason::NotPositiveDefinite);
                continue;
            }
            let Ok(updated_objective) = covariance_objective(&updated, candidate) else {
                trial_rejections.push(CovarianceTrialRejectionReason::ObjectiveUnavailable);
                continue;
            };
            if objective_nonincrease(updated_objective, current_objective, n) {
                let changed = (0..n).any(|row| {
                    (row..n).any(|col| {
                        self.estimated_mask[[row, col]]
                            && updated[[row, col]] != current[[row, col]]
                    })
                });
                return Ok(CovarianceUpdateResult {
                    matrix: updated,
                    status: if changed {
                        CovarianceUpdateStatus::Accepted
                    } else {
                        CovarianceUpdateStatus::NoOp
                    },
                    solved_target: Some(target),
                    accepted_fraction: Some(fraction),
                    attempted_fractions,
                    trial_rejections,
                    rejection_reason: None,
                });
            }
            trial_rejections.push(CovarianceTrialRejectionReason::ObjectiveIncrease);
        }

        Ok(CovarianceUpdateResult {
            matrix: current.clone(),
            status: CovarianceUpdateStatus::Rejected,
            solved_target: Some(target),
            accepted_fraction: None,
            attempted_fractions,
            trial_rejections,
            rejection_reason: Some(CovarianceUpdateRejectionReason::BacktrackingExhausted),
        })
    }

    #[cfg(test)]
    pub(crate) fn update(
        &self,
        current: &Array2<f64>,
        candidate: &Array2<f64>,
        minimum_variance: f64,
    ) -> Result<Array2<f64>> {
        Ok(self
            .update_with_status(current, candidate, minimum_variance)?
            .matrix)
    }

    fn classify_components(&self) -> Vec<OmegaComponent> {
        let n = self.names.len();
        let mut component_id = vec![usize::MAX; n];
        let mut next_id = 0;
        // BFS on structural_mask to find connected components.
        for start in 0..n {
            if component_id[start] != usize::MAX {
                continue;
            }
            component_id[start] = next_id;
            let mut pending = vec![start];
            while let Some(row) = pending.pop() {
                for (col, id) in component_id.iter_mut().enumerate() {
                    if row != col && self.structural_mask[[row, col]] && *id == usize::MAX {
                        *id = next_id;
                        pending.push(col);
                    }
                }
            }
            next_id += 1;
        }

        (0..next_id)
            .map(|id| {
                let indices: Vec<usize> = (0..n).filter(|&idx| component_id[idx] == id).collect();
                let mut free_coords = Vec::new();
                let mut fixed_coords = Vec::new();
                let mut has_free = false;
                let mut has_fixed = false;
                let mut declared_count = 0usize;
                for &ri in &indices {
                    for &ci in &indices {
                        if ri <= ci && self.structural_mask[[ri, ci]] {
                            declared_count += 1;
                            if self.estimated_mask[[ri, ci]] {
                                free_coords.push((ri, ci));
                                has_free = true;
                            } else {
                                fixed_coords.push((ri, ci));
                                has_fixed = true;
                            }
                        }
                    }
                }
                let kind = match (has_free, has_fixed) {
                    (false, true) => OmegaComponentKind::AllFixed,
                    (true, false) => {
                        // Dense iff every intra-component pair is declared.
                        let sz = indices.len();
                        let dense = declared_count == sz * (sz + 1) / 2;
                        if dense {
                            OmegaComponentKind::DenseAllFree
                        } else {
                            OmegaComponentKind::SparseAllFree
                        }
                    }
                    (true, true) => OmegaComponentKind::Mixed,
                    (false, false) => unreachable!("empty component"),
                };
                OmegaComponent {
                    kind,
                    indices,
                    free_coords,
                    fixed_coords,
                }
            })
            .collect()
    }

    // ── Deterministic local GEM minimizer ─────────────────────────────
    //
    /// Minimise `f(Ω)=logdet(Ω)+tr(Ω⁻¹S)` locally over one component's
    /// declared free symmetric coordinates. The negative gradient is
    /// `Ω⁻¹(S-Ω)Ω⁻¹`; off-diagonal coordinates use the full symmetric basis.
    /// Newton uses the analytic Hessian, with Fisher scoring only when that
    /// Hessian is not strictly PD or its strict solve fails. Armijo therefore
    /// uses `f(Ω+αp) <= f(Ω)-c α (-∇f)'p` with a positive directional decrease.
    fn local_gem_minimizer(
        &self,
        current: &Array2<f64>,
        second_moment: &Array2<f64>,
        free_coords: &[(usize, usize)],
    ) -> Result<(Array2<f64>, GemTrace)> {
        validate_finite_symmetric(second_moment, "local GEM second moment")?;
        if free_coords.is_empty() {
            return Ok((current.clone(), GemTrace::default()));
        }

        let mut omega = current.clone();
        self.restore_constraints(&mut omega);
        let start_objective = covariance_objective(&omega, second_moment)?;
        let mut objective = start_objective;
        let mut trace = GemTrace::new(start_objective);

        // Dimensionless terminal tolerance:
        // 4*ε^(2/3)*sqrt(max(1,m)). The factor four is a fixed dot/inverse
        // operation roundoff budget. The rule depends only on binary64 machine
        // precision and free-coordinate count and is tighter than frozen D0
        // relative accuracy at covariance-scale coordinates.
        let terminal_tolerance =
            4.0 * f64::EPSILON.powf(2.0 / 3.0) * (free_coords.len().max(1) as f64).sqrt();

        for _ in 0..512 {
            let inverse = inverse_spd(&omega)?;
            let score_matrix = inverse.dot(&(second_moment - &omega)).dot(&inverse);
            let raw_score = free_coords
                .iter()
                .map(|&(row, col)| {
                    if row == col {
                        score_matrix[[row, col]]
                    } else {
                        score_matrix[[row, col]] + score_matrix[[col, row]]
                    }
                })
                .collect::<Vec<_>>();
            if raw_score.iter().any(|value| !value.is_finite()) {
                bail!("local GEM produced a non-finite score");
            }

            // Overflow-safe coordinate scale d_k=sqrt(Ω_ii)*sqrt(Ω_jj).
            let scales = free_coords
                .iter()
                .map(|&(row, col)| omega[[row, row]].sqrt() * omega[[col, col]].sqrt())
                .collect::<Vec<_>>();
            if scales
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
            {
                bail!("local GEM produced a non-finite coordinate scale");
            }
            let scaled_score = raw_score
                .iter()
                .zip(&scales)
                .map(|(score, scale)| score * scale)
                .collect::<Vec<_>>();
            if scaled_score.iter().any(|value| !value.is_finite()) {
                bail!("local GEM produced a non-finite scaled score");
            }
            let scaled_score_norm = scaled_score
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();

            let hessian_scaled = scale_symmetric_matrix(
                covariance_hessian(&inverse, second_moment, free_coords),
                &scales,
                "local GEM Hessian",
            )?;
            let hessian_pd = dense_is_positive_definite(&hessian_scaled);

            if scaled_score_norm <= terminal_tolerance {
                certify_gem_terminal(hessian_pd, objective, start_objective, omega.nrows())?;
                return Ok((omega, trace));
            }

            let information_scaled = scale_symmetric_matrix(
                covariance_information(&inverse, free_coords),
                &scales,
                "local GEM Fisher information",
            )?;
            if !dense_is_positive_definite(&information_scaled) {
                bail!("local GEM Fisher information is not strictly positive definite");
            }

            let (scaled_direction, used_newton) = if hessian_pd {
                match solve_dense_strict(hessian_scaled.clone(), scaled_score.clone()) {
                    Ok(direction) => (direction, true),
                    Err(_) => (
                        solve_dense_strict(information_scaled, scaled_score.clone())?,
                        false,
                    ),
                }
            } else {
                (
                    solve_dense_strict(information_scaled, scaled_score.clone())?,
                    false,
                )
            };
            let directional_decrease = scaled_score
                .iter()
                .zip(&scaled_direction)
                .map(|(score, direction)| score * direction)
                .sum::<f64>();
            if !directional_decrease.is_finite() || directional_decrease <= 0.0 {
                bail!("local GEM produced a non-descent direction");
            }

            if directional_decrease.sqrt() <= terminal_tolerance {
                certify_gem_terminal(hessian_pd, objective, start_objective, omega.nrows())?;
                return Ok((omega, trace));
            }

            // x=Dz, so a scaled direction maps back as p=D p_s.
            let raw_direction = scaled_direction
                .iter()
                .zip(&scales)
                .map(|(direction, scale)| direction * scale)
                .collect::<Vec<_>>();
            if raw_direction.iter().any(|value| !value.is_finite()) {
                bail!("local GEM produced a non-finite raw direction");
            }

            let mut accepted = None;
            for attempt in 0..64 {
                let fraction = 0.5_f64.powi(attempt);
                let mut trial = omega.clone();
                for (&(row, col), delta) in free_coords.iter().zip(&raw_direction) {
                    let value = omega[[row, col]] + fraction * delta;
                    trial[[row, col]] = value;
                    trial[[col, row]] = value;
                }
                self.restore_constraints(&mut trial);
                let Ok(trial_objective) = covariance_objective(&trial, second_moment) else {
                    continue;
                };
                if trial_objective < objective
                    && trial_objective <= objective - 1e-4 * fraction * directional_decrease
                {
                    accepted = Some((trial, trial_objective));
                    break;
                }
            }
            let Some((trial, trial_objective)) = accepted else {
                if certify_roundoff_stalled_newton_iterate(
                    hessian_pd,
                    objective,
                    start_objective,
                    omega.nrows(),
                    used_newton,
                    directional_decrease,
                ) {
                    trace.newton_used = true;
                    return Ok((omega, trace));
                }
                bail!(
                    "local GEM line search did not find a strict SPD descent step (scaled score norm {scaled_score_norm}, directional decrease {directional_decrease}, terminal tolerance {terminal_tolerance})"
                );
            };
            trace.accept(trial_objective, used_newton);
            omega = trial;
            objective = trial_objective;
        }

        bail!("local GEM did not converge in 512 iterations")
    }

    fn restore_constraints(&self, matrix: &mut Array2<f64>) {
        for row in 0..self.names.len() {
            for col in 0..self.names.len() {
                if !self.structural_mask[[row, col]] {
                    matrix[[row, col]] = 0.0;
                } else if !self.estimated_mask[[row, col]] {
                    matrix[[row, col]] = self.initial[[row, col]];
                }
            }
        }
    }

    pub(crate) fn has_estimated_entries(&self) -> bool {
        self.estimated_mask.iter().any(|value| *value)
    }

    pub(crate) fn structural_mask(&self) -> &Array2<bool> {
        &self.structural_mask
    }

    pub(crate) fn estimated_mask(&self) -> &Array2<bool> {
        &self.estimated_mask
    }
}

#[derive(Debug, Clone, Default)]
struct GemTrace {
    newton_used: bool,
    fisher_used: bool,
    #[cfg(test)]
    objective_sequence: Vec<f64>,
}

impl GemTrace {
    #[allow(unused_variables)]
    fn new(start_objective: f64) -> Self {
        Self {
            #[cfg(test)]
            objective_sequence: vec![start_objective],
            ..Self::default()
        }
    }

    #[allow(unused_variables)]
    fn accept(&mut self, objective: f64, used_newton: bool) {
        self.newton_used |= used_newton;
        self.fisher_used |= !used_newton;
        #[cfg(test)]
        self.objective_sequence.push(objective);
    }
}

/// Post-hoc convergence diagnostics for the local GEM solver.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct CovarianceConvergenceMetrics {
    pub gradient_norm: f64,
    pub hessian_positive_definite: bool,
    pub objective: f64,
    pub convergence_threshold: f64,
    pub newton_used: bool,
    pub fisher_used: bool,
    /// Initial objective followed by every accepted inner objective.
    pub objective_sequence: Vec<f64>,
}

#[cfg(test)]
impl ResolvedOmega {
    pub(crate) fn local_gem_with_metrics(
        &self,
        current: &Array2<f64>,
        second_moment: &Array2<f64>,
    ) -> Result<(Array2<f64>, CovarianceConvergenceMetrics)> {
        let all_free_coords = (0..self.names.len())
            .flat_map(|row| (row..self.names.len()).map(move |col| (row, col)))
            .filter(|(row, col)| self.estimated_mask[[*row, *col]])
            .collect::<Vec<_>>();
        let gem_components = self
            .classify_components()
            .into_iter()
            .filter(|component| {
                matches!(
                    component.kind,
                    OmegaComponentKind::SparseAllFree | OmegaComponentKind::Mixed
                )
            })
            .collect::<Vec<_>>();

        let mut result = current.clone();
        let mut aggregate = GemTrace::new(covariance_objective(current, second_moment)?);
        if gem_components.is_empty() {
            result = self.update(current, second_moment, 0.0)?;
        } else {
            for component in gem_components {
                let (component_result, component_trace) =
                    self.local_gem_minimizer(&result, second_moment, &component.free_coords)?;
                result = component_result;
                aggregate.newton_used |= component_trace.newton_used;
                aggregate.fisher_used |= component_trace.fisher_used;
                aggregate
                    .objective_sequence
                    .extend(component_trace.objective_sequence.into_iter().skip(1));
            }
        }

        let inverse = inverse_spd(&result)
            .map_err(|error| anyhow::anyhow!("converged omega is not SPD: {error}"))?;
        let score_matrix = inverse.dot(&(second_moment - &result)).dot(&inverse);
        let scales = all_free_coords
            .iter()
            .map(|&(row, col)| result[[row, row]].sqrt() * result[[col, col]].sqrt())
            .collect::<Vec<_>>();
        let gradient_norm = all_free_coords
            .iter()
            .zip(&scales)
            .map(|(&(row, col), scale)| {
                let score = if row == col {
                    score_matrix[[row, col]]
                } else {
                    score_matrix[[row, col]] + score_matrix[[col, row]]
                };
                (score * scale).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        let reduced_hessian = scale_symmetric_matrix(
            covariance_hessian(&inverse, second_moment, &all_free_coords),
            &scales,
            "test local GEM Hessian",
        )?;
        let objective = covariance_objective(&result, second_moment)?;
        let convergence_threshold =
            4.0 * f64::EPSILON.powf(2.0 / 3.0) * (all_free_coords.len().max(1) as f64).sqrt();
        Ok((
            result,
            CovarianceConvergenceMetrics {
                gradient_norm,
                hessian_positive_definite: dense_is_positive_definite(&reduced_hessian),
                objective,
                convergence_threshold,
                newton_used: aggregate.newton_used,
                fisher_used: aggregate.fisher_used,
                objective_sequence: aggregate.objective_sequence,
            },
        ))
    }
}

fn copy_component_coordinates(
    destination: &mut Array2<f64>,
    source: &Array2<f64>,
    coordinates: &[(usize, usize)],
) {
    for &(row, col) in coordinates {
        destination[[row, col]] = source[[row, col]];
        destination[[col, row]] = source[[row, col]];
    }
}

fn objective_evaluation_roundoff_allowance(
    candidate: f64,
    current: f64,
    dimension: usize,
) -> Option<f64> {
    // A Cholesky/inverse/trace evaluation is O(n^3). The only equality slack
    // admitted is 64*n^3 binary64 roundoffs at the objective's finite scale.
    let relative_roundoffs = 64.0 * (dimension.max(1) as f64).powi(3) * f64::EPSILON;
    let allowance = candidate.abs().max(current.abs()).max(1.0) * relative_roundoffs;
    allowance.is_finite().then_some(allowance)
}

fn objective_nonincrease(candidate: f64, current: f64, dimension: usize) -> bool {
    if candidate <= current {
        return true;
    }
    objective_evaluation_roundoff_allowance(candidate, current, dimension).is_some_and(
        |allowance| (candidate - current).is_finite() && candidate - current <= allowance,
    )
}

fn certify_roundoff_stalled_newton_iterate(
    hessian_positive_definite: bool,
    objective: f64,
    start_objective: f64,
    dimension: usize,
    used_newton: bool,
    objective_scale_newton_decrement: f64,
) -> bool {
    hessian_positive_definite
        && used_newton
        && objective_nonincrease(objective, start_objective, dimension)
        && objective_scale_newton_decrement.is_finite()
        && objective_scale_newton_decrement >= 0.0
        && objective_evaluation_roundoff_allowance(objective, start_objective, dimension)
            .is_some_and(|allowance| objective_scale_newton_decrement <= allowance)
}

fn certify_gem_terminal(
    hessian_positive_definite: bool,
    objective: f64,
    start_objective: f64,
    dimension: usize,
) -> Result<()> {
    if !hessian_positive_definite {
        bail!(
            "local GEM converged to a non-minimum stationary point: analytic Hessian is not strictly positive definite"
        );
    }
    if !objective_nonincrease(objective, start_objective, dimension) {
        bail!(
            "local GEM terminal objective increased relative to its initial objective ({objective} > {start_objective})"
        );
    }
    Ok(())
}

fn validate_finite_symmetric(matrix: &Array2<f64>, label: &str) -> Result<()> {
    if matrix.nrows() != matrix.ncols() {
        bail!("{label} must be square");
    }
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            if !matrix[[row, col]].is_finite() {
                bail!("{label} must be finite");
            }
            if row > col && matrix[[row, col]] != matrix[[col, row]] {
                bail!("{label} must be symmetric");
            }
        }
    }
    Ok(())
}

fn inverse_spd(matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let lower = cholesky_lower(matrix)?;
    let n = matrix.nrows();
    let mut inverse = Array2::zeros((n, n));
    for column in 0..n {
        let mut forward = vec![0.0; n];
        for row in 0..n {
            let prior = (0..row)
                .map(|index| lower[row][index] * forward[index])
                .sum::<f64>();
            forward[row] = (f64::from(row == column) - prior) / lower[row][row];
        }
        let mut solution = vec![0.0; n];
        for row in (0..n).rev() {
            let prior = ((row + 1)..n)
                .map(|index| lower[index][row] * solution[index])
                .sum::<f64>();
            solution[row] = (forward[row] - prior) / lower[row][row];
        }
        for row in 0..n {
            inverse[[row, column]] = solution[row];
        }
    }
    if inverse.iter().any(|value| !value.is_finite()) {
        bail!("local GEM inverse is non-finite");
    }
    Ok(inverse)
}

fn covariance_objective(omega: &Array2<f64>, second_moment: &Array2<f64>) -> Result<f64> {
    let lower = cholesky_lower(omega)?;
    let inverse = inverse_spd(omega)?;
    let trace = (0..omega.nrows())
        .map(|row| {
            (0..omega.ncols())
                .map(|col| inverse[[row, col]] * second_moment[[col, row]])
                .sum::<f64>()
        })
        .sum::<f64>();
    let objective = cholesky_log_determinant(&lower) + trace;
    if !objective.is_finite() {
        bail!("local GEM objective is non-finite");
    }
    Ok(objective)
}

fn covariance_information(inverse: &Array2<f64>, coordinates: &[(usize, usize)]) -> Vec<Vec<f64>> {
    let n = inverse.nrows();
    let basis = |row: usize, col: usize| {
        Array2::from_shape_fn((n, n), |(left, right)| {
            if (left == row && right == col) || (row != col && left == col && right == row) {
                1.0
            } else {
                0.0
            }
        })
    };
    coordinates
        .iter()
        .map(|&(left_row, left_col)| {
            let left = inverse.dot(&basis(left_row, left_col));
            coordinates
                .iter()
                .map(|&(right_row, right_col)| {
                    let product = left.dot(inverse).dot(&basis(right_row, right_col));
                    (0..n).map(|index| product[[index, index]]).sum()
                })
                .collect()
        })
        .collect()
}

fn covariance_hessian(
    inverse: &Array2<f64>,
    second_moment: &Array2<f64>,
    coordinates: &[(usize, usize)],
) -> Vec<Vec<f64>> {
    let n = inverse.nrows();
    let bases = coordinates
        .iter()
        .map(|&(row, col)| {
            Array2::from_shape_fn((n, n), |(left, right)| {
                if (left == row && right == col) || (row != col && left == col && right == row) {
                    1.0
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    bases
        .iter()
        .map(|left| {
            bases
                .iter()
                .map(|right| {
                    let first = inverse.dot(right).dot(inverse).dot(left);
                    let second = first.dot(inverse).dot(second_moment);
                    let third = inverse
                        .dot(left)
                        .dot(inverse)
                        .dot(right)
                        .dot(inverse)
                        .dot(second_moment);
                    (0..n)
                        .map(|index| {
                            -first[[index, index]] + second[[index, index]] + third[[index, index]]
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn scale_symmetric_matrix(
    matrix: Vec<Vec<f64>>,
    scales: &[f64],
    label: &str,
) -> Result<Vec<Vec<f64>>> {
    let n = scales.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        bail!("{label} dimensions differ from the free-coordinate count");
    }
    let mut scaled = vec![vec![0.0; n]; n];
    for row in 0..n {
        for col in 0..n {
            scaled[row][col] = matrix[row][col] * scales[row] * scales[col];
            if !scaled[row][col].is_finite() {
                bail!("{label} is non-finite after coordinate scaling");
            }
        }
    }
    symmetrize_roundoff_equivalent(scaled, label)
}

fn symmetrize_roundoff_equivalent(mut matrix: Vec<Vec<f64>>, label: &str) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    if matrix.iter().any(|row| row.len() != n) {
        bail!("{label} must be square");
    }
    let scale = matrix.iter().flatten().try_fold(0.0_f64, |scale, value| {
        if value.is_finite() {
            Ok(scale.max(value.abs()))
        } else {
            Err(anyhow::anyhow!("{label} must be finite"))
        }
    })?;
    let asymmetry_allowance = scale * 64.0 * (n.max(1) as f64) * f64::EPSILON;
    let mut row = 0;
    while row < n {
        let mut col = 0;
        while col < row {
            let left = matrix[row][col];
            let right = matrix[col][row];
            if (left - right).abs() > asymmetry_allowance {
                bail!("{label} has material floating-point asymmetry");
            }
            // Overflow-safe average; this changes only entries already proven
            // equal within the explicit matrix-roundoff allowance.
            let symmetric = 0.5 * left + 0.5 * right;
            if !symmetric.is_finite() {
                bail!("{label} symmetrization is non-finite");
            }
            matrix[row][col] = symmetric;
            matrix[col][row] = symmetric;
            col += 1;
        }
        row += 1;
    }
    Ok(matrix)
}

fn dense_is_positive_definite(matrix: &[Vec<f64>]) -> bool {
    let n = matrix.len();
    if n == 0 {
        return true;
    }
    if matrix.iter().any(|row| row.len() != n) {
        return false;
    }
    let array = Array2::from_shape_fn((n, n), |(row, col)| matrix[row][col]);
    cholesky_lower(&array).is_ok()
}

#[cfg(test)]
fn hessian_is_positive_definite(hessian: &[Vec<f64>]) -> bool {
    symmetrize_roundoff_equivalent(hessian.to_vec(), "local GEM Hessian")
        .is_ok_and(|matrix| dense_is_positive_definite(&matrix))
}

fn solve_dense_strict(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        bail!("local GEM linear system dimensions differ");
    }
    for pivot in 0..n {
        let pivot_row = (pivot..n)
            .max_by(|left, right| {
                matrix[*left][pivot]
                    .abs()
                    .total_cmp(&matrix[*right][pivot].abs())
            })
            .ok_or_else(|| anyhow::anyhow!("local GEM pivot range is empty"))?;
        matrix.swap(pivot, pivot_row);
        rhs.swap(pivot, pivot_row);
        let diagonal = matrix[pivot][pivot];
        if diagonal == 0.0 || !diagonal.is_finite() {
            bail!("local GEM linear system is singular or non-finite");
        }
        let pivot_values = matrix[pivot][pivot..].to_vec();
        for row in (pivot + 1)..n {
            let factor = matrix[row][pivot] / diagonal;
            if !factor.is_finite() {
                bail!("local GEM linear solve is non-finite");
            }
            for (entry, pivot_entry) in matrix[row][pivot..].iter_mut().zip(&pivot_values) {
                *entry -= factor * pivot_entry;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let prior = ((row + 1)..n)
            .map(|col| matrix[row][col] * solution[col])
            .sum::<f64>();
        solution[row] = (rhs[row] - prior) / matrix[row][row];
        if !solution[row].is_finite() {
            bail!("local GEM linear solve is non-finite");
        }
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Parameter;

    fn parameters() -> ParameterSpace<UnboundedParameter> {
        [Parameter::log("ke"), Parameter::log("v")]
            .into_iter()
            .collect()
    }

    #[test]
    fn default_omega_is_estimated_diagonal_identity() {
        let prior = ParametricPrior::new(parameters(), None, None).unwrap();

        assert_eq!(prior.random_effect_names(), &["ke", "v"]);
        assert_eq!(prior.omega(), &ndarray::array![[1.0, 0.0], [0.0, 1.0]]);
        assert!(!prior.resolved_omega().structural_mask()[[0, 1]]);
        assert!(!prior.resolved_omega().estimated_mask()[[0, 1]]);
    }

    #[test]
    fn diagonal_sd_overflow_is_rejected_before_multiplication() {
        let error = ParametricPrior::new(
            parameters(),
            Some(Omega::diagonal_standard_deviations([
                ("ke", f64::MAX),
                ("v", 1.0),
            ])),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("would overflow its variance representation"));
    }

    #[test]
    fn explicit_omega_tracks_structural_and_fixed_entries() {
        let omega = Omega::diagonal([("ke", 0.2)])
            .fixed_variance("v", 0.4)
            .covariance("ke", "v", 0.1);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();

        assert_eq!(prior.omega(), &ndarray::array![[0.2, 0.1], [0.1, 0.4]]);
        assert!(prior.resolved_omega().estimated_mask()[[0, 0]]);
        assert!(!prior.resolved_omega().estimated_mask()[[1, 1]]);
        assert!(prior.resolved_omega().structural_mask()[[0, 1]]);
    }

    #[test]
    fn omega_update_preserves_fixed_values_and_structural_zeros() {
        let omega = Omega::diagonal([("ke", 0.2)]).fixed_variance("v", 0.4);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let candidate = ndarray::array![[0.8, 0.3], [0.3, 2.0]];

        let updated = prior
            .resolved_omega()
            .update(prior.omega(), &candidate, 1e-6)
            .unwrap();

        assert!((updated[[0, 0]] - 0.8).abs() < 1e-12);
        assert!((updated[[1, 1]] - 0.4).abs() < 1e-12);
        assert_eq!(updated[[0, 1]], 0.0);
        assert_eq!(updated[[1, 0]], 0.0);
    }

    #[test]
    fn omega_update_preserves_fixed_covariance_and_positive_definiteness_jointly() {
        let omega = Omega::diagonal([("ke", 0.2)])
            .fixed_variance("v", 0.4)
            .fixed_covariance("ke", "v", 0.1);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let candidate = ndarray::array![[0.3, 0.1], [0.1, 0.4]];

        let updated = prior
            .resolved_omega()
            .update(prior.omega(), &candidate, 1e-6)
            .unwrap();

        assert_eq!(updated[[0, 1]], 0.1);
        assert_eq!(updated[[1, 0]], 0.1);
        assert_eq!(updated[[1, 1]], 0.4);
        assert!(cholesky_lower(&updated).is_ok());
    }

    #[test]
    fn mixed_mask_under_relaxation_caps_only_the_free_profile_coordinate() {
        let omega = Omega::new()
            .variance("ke", 0.02)
            .fixed_variance("v", 0.04)
            .fixed_covariance("ke", "v", 0.012);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let current = prior.omega().clone();
        let second_moment = ndarray::array![[0.002, 0.0], [0.0, 0.04]];

        let update = prior
            .resolved_omega()
            .update_with_status_and_max_fraction(&current, &second_moment, 0.0, 0.1)
            .unwrap();

        // The unconstrained profile target is 0.0092, so a 0.1 accepted
        // displacement from 0.02 is 0.01892. Fixed coordinates remain exact.
        assert_eq!(update.status, CovarianceUpdateStatus::Accepted);
        assert!((update.matrix[[0, 0]] - 0.01892).abs() <= 1e-10);
        assert_eq!(update.matrix[[0, 1]], 0.012);
        assert_eq!(update.matrix[[1, 0]], 0.012);
        assert_eq!(update.matrix[[1, 1]], 0.04);
        assert!(cholesky_lower(&update.matrix).is_ok());
        assert!(
            covariance_objective(&update.matrix, &second_moment).unwrap()
                <= covariance_objective(&current, &second_moment).unwrap()
        );
    }

    #[test]
    fn mixed_mask_update_reports_bit_exact_capped_result() {
        let omega = Omega::new()
            .variance("ke", 0.02)
            .fixed_variance("v", 0.04)
            .fixed_covariance("ke", "v", 0.012);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let current = prior.omega().clone();
        let proposal = ndarray::array![[0.002, 0.0], [0.0, 0.04]];
        let update = prior
            .resolved_omega()
            .update_with_status_and_max_fraction(&current, &proposal, 0.0, 0.1)
            .unwrap();

        let expected_matrix = ndarray::array![[0.01892_f64, 0.012], [0.012, 0.04]];
        for (actual, expected) in update.matrix.iter().zip(expected_matrix.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(update.status, CovarianceUpdateStatus::Accepted);
        assert_eq!(update.accepted_fraction, Some(0.1));
        assert_eq!(update.attempted_fractions, vec![0.1]);
        assert!(update.trial_rejections.is_empty());
        assert!(update.rejection_reason.is_none());
        assert_eq!(current, prior.omega().clone());
    }

    #[test]
    fn mixed_mask_two_by_two_matches_exact_profile_score_and_objective() {
        let omega = Omega::new()
            .variance("ke", 0.02)
            .fixed_variance("v", 0.04)
            .fixed_covariance("ke", "v", 0.012);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let second_moment = ndarray::array![[0.002, 0.0], [0.0, 0.04]];

        let updated = prior
            .resolved_omega()
            .update(prior.omega(), &second_moment, 0.0)
            .unwrap();
        let expected = 0.0092;
        assert!((updated[[0, 0]] - expected).abs() <= 1e-10);
        assert!((updated[[0, 0]] / expected - 1.0).abs() <= 1e-9);
        assert_eq!(updated[[0, 1]], 0.012);
        assert_eq!(updated[[1, 1]], 0.04);

        let inverse = inverse_spd(&updated).unwrap();
        let score = inverse.dot(&(&second_moment - &updated)).dot(&inverse);
        assert!(score[[0, 0]].abs() <= 1e-10);
        let objective = covariance_objective(&updated, &second_moment).unwrap();
        for delta in [-1e-5, 1e-5] {
            let mut profile = updated.clone();
            profile[[0, 0]] += delta;
            assert!(covariance_objective(&profile, &second_moment).unwrap() > objective);
        }
    }

    #[test]
    fn mixed_mask_three_by_three_matches_exact_score_and_finite_difference_oracle() {
        let parameters = [
            Parameter::log("a"),
            Parameter::log("b"),
            Parameter::log("c"),
        ]
        .into_iter()
        .collect();
        let omega = Omega::new()
            .variance("a", 0.2)
            .variance("b", 0.4)
            .fixed_variance("c", 0.4)
            .fixed_covariance("a", "b", 0.04)
            .fixed_covariance("a", "c", -0.02)
            .covariance("b", "c", 0.03);
        let prior = ParametricPrior::new(parameters, Some(omega), None).unwrap();
        let expected = ndarray::array![[0.3, 0.04, -0.02], [0.04, 0.5, 0.06], [-0.02, 0.06, 0.4]];
        // Construct S=Ω+ΩGΩ with G exactly zero on every free symmetric
        // coordinate. Thus `expected` is an independent analytic stationary
        // oracle while fixed-coordinate scores remain nonzero.
        let fixed_score = ndarray::array![[0.0, 0.04, -0.03], [0.04, 0.0, 0.0], [-0.03, 0.0, 0.05]];
        let second_moment = &expected + expected.dot(&fixed_score).dot(&expected);
        let updated = prior
            .resolved_omega()
            .update(prior.omega(), &second_moment, 0.0)
            .unwrap();
        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    (updated[[row, col]] - expected[[row, col]]).abs() <= 1e-10,
                    "updated={updated:?}, expected={expected:?}"
                );
            }
        }

        let inverse = inverse_spd(&updated).unwrap();
        let score = inverse.dot(&(&second_moment - &updated)).dot(&inverse);
        for &(row, col) in &[(0, 0), (1, 1), (1, 2)] {
            assert!(score[[row, col]].abs() <= 1e-10);
            let step = 1e-6;
            let mut plus = updated.clone();
            let mut minus = updated.clone();
            plus[[row, col]] += step;
            minus[[row, col]] -= step;
            if row != col {
                plus[[col, row]] += step;
                minus[[col, row]] -= step;
            }
            let finite_difference = (covariance_objective(&plus, &second_moment).unwrap()
                - covariance_objective(&minus, &second_moment).unwrap())
                / (2.0 * step);
            assert!(finite_difference.abs() <= 1e-9);
        }
    }

    #[test]
    fn omega_update_accepts_finite_high_scale_positive_definite_candidate() {
        let prior = ParametricPrior::new(
            parameters(),
            Some(Omega::diagonal([("ke", 0.2), ("v", 0.4)]).covariance("ke", "v", 0.1)),
            None,
        )
        .unwrap();
        let candidate = ndarray::array![[1e200, 5e199], [5e199, 1e200]];

        let updated = prior
            .resolved_omega()
            .update(prior.omega(), &candidate, 1e-6)
            .unwrap();

        assert_eq!(updated, candidate);
        assert!(cholesky_lower(&updated).is_ok());
    }

    #[test]
    fn omega_update_rejects_non_finite_candidate_by_retaining_current_matrix() {
        let prior = ParametricPrior::new(parameters(), None, None).unwrap();
        let current = prior.omega().clone();
        let candidate = ndarray::array![[f64::NAN, 0.0], [0.0, f64::INFINITY]];

        let updated = prior
            .resolved_omega()
            .update(&current, &candidate, 1e-6)
            .unwrap();

        assert_eq!(updated, current);
    }

    #[test]
    fn covariance_update_status_reports_accepted_change() {
        let prior = ParametricPrior::new(parameters(), None, None).unwrap();
        let update = prior
            .resolved_omega()
            .update_with_status(prior.omega(), &ndarray::array![[0.8, 0.0], [0.0, 1.5]], 0.0)
            .unwrap();

        assert_eq!(update.status, CovarianceUpdateStatus::Accepted);
        assert_eq!(update.matrix, ndarray::array![[0.8, 0.0], [0.0, 1.5]]);
        assert_eq!(
            update.solved_target,
            Some(ndarray::array![[0.8, 0.0], [0.0, 1.5]])
        );
        assert_eq!(update.accepted_fraction, Some(1.0));
        assert_eq!(update.attempted_fractions, vec![1.0]);
        assert!(update.trial_rejections.is_empty());
        assert!(update.rejection_reason.is_none());
    }

    #[test]
    fn covariance_update_status_reports_all_fixed_no_op() {
        let prior = ParametricPrior::new(
            parameters(),
            Some(
                Omega::new()
                    .fixed_variance("ke", 0.2)
                    .fixed_variance("v", 0.4),
            ),
            None,
        )
        .unwrap();
        let update = prior
            .resolved_omega()
            .update_with_status(prior.omega(), &ndarray::array![[0.8, 0.3], [0.3, 2.0]], 0.0)
            .unwrap();

        assert_eq!(update.status, CovarianceUpdateStatus::NoOp);
        assert_eq!(&update.matrix, prior.omega());
        assert!(update.solved_target.is_none());
        assert_eq!(update.accepted_fraction, None);
        assert!(update.attempted_fractions.is_empty());
        assert!(update.trial_rejections.is_empty());
        assert!(update.rejection_reason.is_none());
    }

    #[test]
    fn covariance_update_status_ignores_irrelevant_cross_component_candidate() {
        let prior = ParametricPrior::new(parameters(), None, None).unwrap();
        let update = prior
            .resolved_omega()
            .update_with_status(
                prior.omega(),
                &ndarray::array![[1.0, 0.75], [0.75, 1.0]],
                0.0,
            )
            .unwrap();

        assert_eq!(update.status, CovarianceUpdateStatus::NoOp);
        assert_eq!(&update.matrix, prior.omega());
    }

    #[test]
    fn explicit_omega_rejects_missing_or_unknown_variances() {
        let missing =
            ParametricPrior::new(parameters(), Some(Omega::diagonal([("ke", 0.2)])), None)
                .unwrap_err();
        assert!(missing
            .to_string()
            .contains("omega variance for random effect 'v' is not declared"));

        let unknown = ParametricPrior::new(
            parameters(),
            Some(Omega::diagonal([("ke", 0.2), ("v", 0.4)]).covariance("ke", "ka", 0.1)),
            None,
        )
        .unwrap_err();
        assert!(unknown
            .to_string()
            .contains("not a declared IIV random effect"));
    }

    #[test]
    fn iov_resolves_against_model_parameters_independently_of_iiv() {
        let parameters = [
            Parameter::log("ke").without_random_effect(),
            Parameter::log("v"),
        ]
        .into_iter()
        .collect();
        let prior =
            ParametricPrior::new(parameters, None, Some(Iov::diagonal([("ke", 0.3)]))).unwrap();

        assert_eq!(
            prior.iov_effect_names(),
            Some(["ke".to_string()].as_slice())
        );
        assert_eq!(prior.omega_iov(), Some(&ndarray::array![[0.3]]));
        assert_eq!(prior.resolved_iov().unwrap().parameter_indices(), &[0]);
    }

    #[test]
    fn explicit_omega_must_be_positive_definite() {
        let error = ParametricPrior::new(
            parameters(),
            Some(Omega::diagonal([("ke", 0.2), ("v", 0.4)]).covariance("ke", "v", 1.0)),
            None,
        )
        .unwrap_err();

        assert_eq!(error.to_string(), "omega must be positive definite");
    }

    // ── PD Hessian and stationary-point certification ──────────────────

    #[test]
    fn local_gem_certifies_pd_hessian_at_convergence() {
        // Two free coordinates (ke, v) + fixed covariance. The exact solution
        // has a PD 2×2 Hessian by construction; verify the certification passes.
        let omega = Omega::new()
            .variance("ke", 0.2)
            .variance("v", 0.4)
            .fixed_covariance("ke", "v", 0.05);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        // S = Ω at the solution: score vanishes, Ω = S simplifies the Hessian.
        let second_moment = ndarray::array![[0.2, 0.05], [0.05, 0.4]];
        let current = prior.omega().clone();
        let (result, metrics) = prior
            .resolved_omega()
            .local_gem_with_metrics(&current, &second_moment)
            .unwrap();
        assert!(metrics.hessian_positive_definite);
        assert!(metrics.gradient_norm <= 1e-8);
        assert!(metrics.gradient_norm <= metrics.convergence_threshold);
        // The result must match the exact solution.
        assert!((result[[0, 0]] - 0.2).abs() <= 1e-10);
        assert!((result[[1, 1]] - 0.4).abs() <= 1e-10);
        assert!((result[[0, 1]] - 0.05).abs() <= 1e-10);
    }

    #[test]
    fn roundoff_stalled_mixed_mask_newton_iterate_is_certified() {
        let parameters = [
            Parameter::log("ke"),
            Parameter::log("v"),
            Parameter::log("bio"),
        ]
        .into_iter()
        .collect();
        let current_a = f64::from_bits(0x3fb3_ae45_0de5_60fe);
        let prior = ParametricPrior::new(
            parameters,
            Some(
                Omega::new()
                    .variance("ke", current_a)
                    .fixed_variance("v", 0.04)
                    .fixed_variance("bio", 0.03)
                    .fixed_covariance("ke", "v", 0.012),
            ),
            None,
        )
        .unwrap();
        let proposal = ndarray::array![
            [
                f64::from_bits(0x3fb3_cdaf_fb74_e0a1),
                f64::from_bits(0x3f89_6735_4993_5f80),
                f64::from_bits(0xbf80_be7b_bd4e_b876),
            ],
            [
                f64::from_bits(0x3f89_6735_4993_5f80),
                f64::from_bits(0x3fa1_40d8_5314_f738),
                f64::from_bits(0xbf62_706b_63ca_bde0),
            ],
            [
                f64::from_bits(0xbf80_be7b_bd4e_b876),
                f64::from_bits(0xbf62_706b_63ca_bde0),
                f64::from_bits(0x3f9d_9f0e_1372_9e0b),
            ],
        ];
        let fixed_ratio = 0.012 / 0.04;
        let analytic_target = 0.0036 + proposal[[0, 0]] - 2.0 * fixed_ratio * proposal[[0, 1]]
            + fixed_ratio * fixed_ratio * proposal[[1, 1]];
        assert_eq!(analytic_target.to_bits(), 0x3fb3_98a2_6afb_8fc1);

        let current = prior.omega().clone();
        let (local_target, metrics) = prior
            .resolved_omega()
            .local_gem_with_metrics(&current, &proposal)
            .unwrap();
        assert_eq!(local_target[[0, 0]].to_bits(), 0x3fb3_98a2_6a73_c4bb);
        assert!(metrics.gradient_norm > metrics.convergence_threshold);
        assert!(metrics.hessian_positive_definite);
        assert!(metrics.newton_used);

        let update = prior
            .resolved_omega()
            .update_with_status_and_max_fraction(&current, &proposal, 1e-6, 0.1)
            .unwrap();
        assert_eq!(update.status, CovarianceUpdateStatus::Accepted);
        assert_eq!(update.accepted_fraction, Some(0.1));
        assert_eq!(
            update.solved_target.as_ref().unwrap()[[0, 0]].to_bits(),
            0x3fb3_98a2_6a73_c4bb
        );
        assert_eq!(update.matrix[[0, 0]].to_bits(), 0x3fb3_ac1b_30c0_6af7);
    }

    #[test]
    fn roundoff_stall_certification_rejects_a_nonstationary_or_unsafe_iterate() {
        let objective = -6.5;
        let start_objective = -6.4;
        let allowance =
            objective_evaluation_roundoff_allowance(objective, start_objective, 3).unwrap();
        assert!(certify_roundoff_stalled_newton_iterate(
            true,
            objective,
            start_objective,
            3,
            true,
            allowance,
        ));
        assert!(!certify_roundoff_stalled_newton_iterate(
            true,
            objective,
            start_objective,
            3,
            true,
            2.0 * allowance,
        ));
        assert!(!certify_roundoff_stalled_newton_iterate(
            false,
            objective,
            start_objective,
            3,
            true,
            allowance,
        ));
        assert!(!certify_roundoff_stalled_newton_iterate(
            true,
            objective,
            start_objective,
            3,
            false,
            allowance,
        ));
        assert!(!certify_roundoff_stalled_newton_iterate(
            true,
            start_objective + 2.0 * allowance,
            start_objective,
            3,
            true,
            allowance,
        ));
    }

    #[test]
    fn three_by_three_mixed_mask_has_pd_hessian_at_exact_solution() {
        let parameters = [
            Parameter::log("a"),
            Parameter::log("b"),
            Parameter::log("c"),
        ]
        .into_iter()
        .collect();
        let omega = Omega::new()
            .variance("a", 0.3)
            .variance("b", 0.5)
            .fixed_variance("c", 0.4)
            .fixed_covariance("a", "b", 0.04)
            .fixed_covariance("a", "c", -0.02)
            .covariance("b", "c", 0.06);
        let prior = ParametricPrior::new(parameters, Some(omega), None).unwrap();
        // S = Ω at the solution.
        let second_moment =
            ndarray::array![[0.3, 0.04, -0.02], [0.04, 0.5, 0.06], [-0.02, 0.06, 0.4]];
        let current = prior.omega().clone();
        let (_result, metrics) = prior
            .resolved_omega()
            .local_gem_with_metrics(&current, &second_moment)
            .unwrap();
        // Gradient must vanish and Hessian must be PD.
        assert!(metrics.gradient_norm <= 1e-8);
        assert!(metrics.hessian_positive_definite);
    }

    #[test]
    fn stationary_nonminimum_zero_score_rejected() {
        // With fixed unit variances and only the covariance free, c=0 is
        // stationary for S=0.2I, but f''(0)=-2+2tr(S)=-1.2 < 0.
        let omega = Omega::new()
            .fixed_variance("ke", 1.0)
            .fixed_variance("v", 1.0)
            .covariance("ke", "v", 0.0);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let current = prior.omega().clone();
        let second_moment = ndarray::array![[0.2, 0.0], [0.0, 0.2]];
        let error = prior
            .resolved_omega()
            .local_gem_with_metrics(&current, &second_moment)
            .unwrap_err();
        assert!(error.to_string().contains("non-minimum stationary point"));

        // Production update follows the explicit reject-and-retain-current path.
        let retained = prior
            .resolved_omega()
            .update(&current, &second_moment, 0.0)
            .unwrap();
        assert_eq!(retained, current);
    }

    // ── Scale-invariant convergence ────────────────────────────────────

    #[test]
    fn scaled_problem_converges_with_same_iteration_budget() {
        // The covariance M-step target is scale-equivariant: scaling S and
        // current by the same factor preserves the GEM iteration behaviour.
        // For the DenseAllFree component, the target = S directly; for mixed
        // components, the GEM converges to the same relative optimum.
        // Verify that the convergence threshold (epsilon-only) is independent
        // of the objective magnitude.
        let omega = Omega::new()
            .variance("ke", 1.0)
            .variance("v", 1.0)
            .covariance("ke", "v", 0.3);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let base = ndarray::array![[0.8, 0.24], [0.24, 1.2]];
        for scale in [1.0, 100.0, 0.01_f64] {
            let second_moment = &base * scale;
            let current = &prior.omega().clone() * scale;
            let (_result, metrics) = prior
                .resolved_omega()
                .local_gem_with_metrics(&current, &second_moment)
                .unwrap();
            // Convergence must be reached. The threshold is epsilon-only.
            assert!(
                metrics.gradient_norm <= metrics.convergence_threshold,
                "scale={scale}: gradient_norm={} > threshold={}",
                metrics.gradient_norm,
                metrics.convergence_threshold
            );
            assert!(metrics.hessian_positive_definite);
        }
    }

    // ── Component dispatch preserving legacy all-free behavior ─────────

    #[test]
    fn all_free_diagonal_bypasses_gem_and_preserves_candidate_exactly() {
        // All-free Omega with no fixed entries must return the candidate
        // unchanged — no GEM optimisation runs.
        let omega = Omega::diagonal([("ke", 0.2), ("v", 0.4)]);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let candidate = ndarray::array![[0.8, 0.0], [0.0, 1.5]];
        let result = prior
            .resolved_omega()
            .update(prior.omega(), &candidate, 1e-6)
            .unwrap();
        // All-free dispatch returns candidate unchanged (no GEM, no mixed mask).
        assert_eq!(result, candidate);
    }

    #[test]
    fn all_free_correlated_bypasses_gem_and_returns_candidate() {
        let omega = Omega::diagonal([("ke", 0.2), ("v", 0.4)]).covariance("ke", "v", 0.1);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let candidate = ndarray::array![[0.5, 0.15], [0.15, 0.7]];
        let result = prior
            .resolved_omega()
            .update(prior.omega(), &candidate, 1e-6)
            .unwrap();
        assert_eq!(result, candidate);
    }

    #[test]
    fn mixed_mask_dispatch_uses_constrained_gem_not_legacy_path() {
        let omega = Omega::new().variance("ke", 0.2).fixed_variance("v", 0.4);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let candidate = ndarray::array![[0.8, 0.0], [0.0, 0.4]];
        let result = prior
            .resolved_omega()
            .update(prior.omega(), &candidate, 1e-6)
            .unwrap();
        // Free variance (ke) is updated; fixed variance (v) stays at 0.4.
        assert!((result[[0, 0]] - 0.8).abs() <= 1e-10);
        assert!((result[[1, 1]] - 0.4).abs() <= 1e-10);
        assert_eq!(result[[0, 1]], 0.0);
    }

    // ── Objective nonincrease ──────────────────────────────────────────

    #[test]
    fn gem_objective_does_not_increase_at_any_step() {
        // This distant mixed-mask start has an indefinite analytic Hessian
        // initially (Fisher fallback) and a PD Hessian later (Newton).
        let omega = Omega::new()
            .fixed_variance("ke", 1.0)
            .fixed_variance("v", 1.0)
            .covariance("ke", "v", 0.0);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let second_moment = ndarray::array![[0.2, 0.1], [0.1, 0.2]];
        let current = prior.omega().clone();
        let (result, metrics) = prior
            .resolved_omega()
            .local_gem_with_metrics(&current, &second_moment)
            .unwrap();
        assert!(
            metrics.fisher_used,
            "fixture did not exercise Fisher fallback"
        );
        assert!(
            metrics.newton_used,
            "fixture did not exercise Newton scoring"
        );
        assert!(metrics
            .objective_sequence
            .windows(2)
            .all(|pair| pair[1] <= pair[0]));
        assert!(metrics.objective < metrics.objective_sequence[0]);
        assert!(cholesky_lower(&result).is_ok());
    }

    // ── Derivative analytic correctness ────────────────────────────────

    #[test]
    fn analytic_score_matches_finite_difference_to_machine_precision() {
        // For the one-free-coordinate case, the analytic score formula
        // must match central finite differences within D0 tolerance.
        let omega = Omega::new()
            .variance("ke", 0.3)
            .fixed_variance("v", 0.4)
            .fixed_covariance("ke", "v", 0.06);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let second_moment = ndarray::array![[0.25, 0.06], [0.06, 0.4]];
        let current = prior.omega().clone();
        let inverse = inverse_spd(&current).unwrap();
        let score_matrix = inverse.dot(&(&second_moment - &current)).dot(&inverse);
        let analytic = score_matrix[[0, 0]];
        let step = 1e-6;
        let mut plus = current.clone();
        let mut minus = current.clone();
        plus[[0, 0]] += step;
        minus[[0, 0]] -= step;
        let finite = (covariance_objective(&plus, &second_moment).unwrap()
            - covariance_objective(&minus, &second_moment).unwrap())
            / (2.0 * step);
        assert!((analytic + finite).abs() <= 1e-9);
    }

    #[test]
    fn analytic_gradient_and_hessian_match_independent_central_differences() {
        let current = ndarray::array![[0.7, 0.08, -0.04], [0.08, 0.9, 0.11], [-0.04, 0.11, 0.6]];
        let second_moment =
            ndarray::array![[0.5, -0.03, 0.07], [-0.03, 1.1, 0.02], [0.07, 0.02, 0.8]];
        let coordinates = (0..3)
            .flat_map(|row| (row..3).map(move |col| (row, col)))
            .collect::<Vec<_>>();
        let inverse = inverse_spd(&current).unwrap();
        let score_matrix = inverse.dot(&(&second_moment - &current)).dot(&inverse);
        let negative_gradient = coordinates
            .iter()
            .map(|&(row, col)| {
                if row == col {
                    score_matrix[[row, col]]
                } else {
                    score_matrix[[row, col]] + score_matrix[[col, row]]
                }
            })
            .collect::<Vec<_>>();
        let hessian = covariance_hessian(&inverse, &second_moment, &coordinates);
        let objective_step = 2e-6;
        let score_step = 2e-6;

        for (column, &(row, col)) in coordinates.iter().enumerate() {
            let mut plus = current.clone();
            let mut minus = current.clone();
            plus[[row, col]] += objective_step;
            minus[[row, col]] -= objective_step;
            if row != col {
                plus[[col, row]] += objective_step;
                minus[[col, row]] -= objective_step;
            }
            let objective_derivative = (covariance_objective(&plus, &second_moment).unwrap()
                - covariance_objective(&minus, &second_moment).unwrap())
                / (2.0 * objective_step);
            assert!(
                (objective_derivative + negative_gradient[column]).abs() <= 2e-8,
                "gradient coordinate {column}: analytic={}, finite={objective_derivative}",
                -negative_gradient[column]
            );

            let score = |delta: f64| {
                let mut perturbed = current.clone();
                perturbed[[row, col]] += delta;
                if row != col {
                    perturbed[[col, row]] += delta;
                }
                let inverse = inverse_spd(&perturbed).unwrap();
                let matrix = inverse.dot(&(&second_moment - &perturbed)).dot(&inverse);
                coordinates
                    .iter()
                    .map(|&(score_row, score_col)| {
                        if score_row == score_col {
                            matrix[[score_row, score_col]]
                        } else {
                            matrix[[score_row, score_col]] + matrix[[score_col, score_row]]
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let plus_score = score(score_step);
            let minus_score = score(-score_step);
            for row_index in 0..coordinates.len() {
                let finite = -(plus_score[row_index] - minus_score[row_index]) / (2.0 * score_step);
                assert!(
                    (hessian[row_index][column] - finite).abs() <= 2e-7,
                    "H[{row_index}][{column}] analytic={} finite={finite}",
                    hessian[row_index][column]
                );
            }
        }
    }

    #[test]
    fn convergence_metrics_instrumentation_is_consistent() {
        // The test-only instrumentation wrapper must report metrics consistent
        // with independently computed values at the returned point.
        let omega = Omega::new()
            .variance("ke", 0.5)
            .fixed_variance("v", 0.5)
            .covariance("ke", "v", 0.2);
        let prior = ParametricPrior::new(parameters(), Some(omega), None).unwrap();
        let second_moment = ndarray::array![[0.3, 0.2], [0.2, 0.5]];
        let current = prior.omega().clone();
        let (result, metrics) = prior
            .resolved_omega()
            .local_gem_with_metrics(&current, &second_moment)
            .unwrap();
        // Independently compute the objective.
        let independent_objective = covariance_objective(&result, &second_moment).unwrap();
        assert!((metrics.objective - independent_objective).abs() <= 1e-12);
        // Hessian PD flag must match direct check.
        let inverse = inverse_spd(&result).unwrap();
        let coords: Vec<_> = (0..2)
            .flat_map(|r| (r..2).map(move |c| (r, c)))
            .filter(|(r, c)| prior.resolved_omega().estimated_mask()[[*r, *c]])
            .collect();
        let direct_hessian = covariance_hessian(&inverse, &second_moment, &coords);
        assert_eq!(
            metrics.hessian_positive_definite,
            hessian_is_positive_definite(&direct_hessian)
        );
    }

    #[test]
    fn sparse_and_mixed_gem_are_unit_rescaling_equivariant() {
        let sparse_parameters = || {
            [
                Parameter::log("a"),
                Parameter::log("b"),
                Parameter::log("c"),
            ]
            .into_iter()
            .collect()
        };
        let sparse_second = ndarray::array![[0.7, 0.05, 0.3], [0.05, 1.1, 0.08], [0.3, 0.08, 0.9]];
        let mixed_second = ndarray::array![[0.18, 0.025], [0.025, 0.5]];
        let mut sparse_reference: Option<Array2<f64>> = None;
        let mut mixed_reference: Option<Array2<f64>> = None;
        for scale in [0.01, 1.0, 100.0] {
            let sparse = ParametricPrior::new(
                sparse_parameters(),
                Some(
                    Omega::diagonal([("a", 1.0 * scale), ("b", 0.8 * scale), ("c", 1.2 * scale)])
                        .covariance("a", "b", 0.15 * scale)
                        .covariance("b", "c", 0.10 * scale),
                ),
                None,
            )
            .unwrap();
            assert_eq!(
                sparse.resolved_omega().classify_components()[0].kind,
                OmegaComponentKind::SparseAllFree
            );
            let (sparse_result, sparse_metrics) = sparse
                .resolved_omega()
                .local_gem_with_metrics(sparse.omega(), &(&sparse_second * scale))
                .unwrap();
            assert!(sparse_metrics.newton_used || sparse_metrics.fisher_used);
            let normalized_sparse = sparse_result / scale;
            if let Some(reference) = &sparse_reference {
                assert!((&normalized_sparse - reference)
                    .iter()
                    .all(|difference| difference.abs() <= 2e-9));
            } else {
                sparse_reference = Some(normalized_sparse);
            }

            let mixed = ParametricPrior::new(
                parameters(),
                Some(
                    Omega::new()
                        .variance("ke", 0.3 * scale)
                        .fixed_variance("v", 0.5 * scale)
                        .fixed_covariance("ke", "v", 0.04 * scale),
                ),
                None,
            )
            .unwrap();
            assert_eq!(
                mixed.resolved_omega().classify_components()[0].kind,
                OmegaComponentKind::Mixed
            );
            let (mixed_result, mixed_metrics) = mixed
                .resolved_omega()
                .local_gem_with_metrics(mixed.omega(), &(&mixed_second * scale))
                .unwrap();
            assert!(mixed_metrics.newton_used || mixed_metrics.fisher_used);
            let normalized_mixed = mixed_result / scale;
            if let Some(reference) = &mixed_reference {
                assert!((&normalized_mixed - reference)
                    .iter()
                    .all(|difference| difference.abs() <= 2e-9));
            } else {
                mixed_reference = Some(normalized_mixed);
            }
        }
    }

    #[test]
    fn connected_sparse_gem_beats_legacy_masked_second_moment_and_satisfies_score() {
        let parameters = [
            Parameter::log("a"),
            Parameter::log("b"),
            Parameter::log("c"),
        ]
        .into_iter()
        .collect();
        let prior = ParametricPrior::new(
            parameters,
            Some(
                Omega::diagonal([("a", 1.0), ("b", 0.8), ("c", 1.2)])
                    .covariance("a", "b", 0.15)
                    .covariance("b", "c", 0.10),
            ),
            None,
        )
        .unwrap();
        let second_moment = ndarray::array![[0.7, 0.05, 0.3], [0.05, 1.1, 0.08], [0.3, 0.08, 0.9]];
        let (result, metrics) = prior
            .resolved_omega()
            .local_gem_with_metrics(prior.omega(), &second_moment)
            .unwrap();
        let mut legacy_masked = second_moment.clone();
        legacy_masked[[0, 2]] = 0.0;
        legacy_masked[[2, 0]] = 0.0;
        assert!(covariance_objective(&legacy_masked, &second_moment).unwrap() > metrics.objective);
        assert!(metrics.gradient_norm <= metrics.convergence_threshold);
        assert!(result.diag().iter().all(|variance| *variance > 1e-8));
        assert!(metrics
            .objective_sequence
            .windows(2)
            .all(|pair| pair[1] <= pair[0]));
    }

    #[test]
    fn disconnected_mixed_components_ignore_cross_component_second_moments() {
        let four_parameters = [
            Parameter::log("a"),
            Parameter::log("b"),
            Parameter::log("c"),
            Parameter::log("d"),
        ]
        .into_iter()
        .collect();
        let declaration = Omega::new()
            .variance("a", 0.3)
            .fixed_variance("b", 0.5)
            .fixed_covariance("a", "b", 0.04)
            .variance("c", 0.6)
            .fixed_variance("d", 0.9)
            .fixed_covariance("c", "d", -0.05);
        let prior = ParametricPrior::new(four_parameters, Some(declaration), None).unwrap();
        let second_moment = ndarray::array![
            [0.18, 0.025, 0.30, -0.20],
            [0.025, 0.5, 0.10, 0.15],
            [0.30, 0.10, 0.45, -0.03],
            [-0.20, 0.15, -0.03, 0.9]
        ];
        let (direct_result, direct_metrics) = prior
            .resolved_omega()
            .local_gem_with_metrics(prior.omega(), &second_moment)
            .unwrap();
        let initial_objective = covariance_objective(prior.omega(), &second_moment).unwrap();
        assert!(
            direct_metrics.objective <= initial_objective,
            "direct={} initial={initial_objective}, result={direct_result:?}",
            direct_metrics.objective
        );
        let result = prior
            .resolved_omega()
            .update(prior.omega(), &second_moment, 0.0)
            .unwrap();
        for row in 0..2 {
            for col in 2..4 {
                assert_eq!(result[[row, col]], 0.0);
                assert_eq!(result[[col, row]], 0.0);
            }
        }
        let first = ParametricPrior::new(
            parameters(),
            Some(
                Omega::new()
                    .variance("ke", 0.3)
                    .fixed_variance("v", 0.5)
                    .fixed_covariance("ke", "v", 0.04),
            ),
            None,
        )
        .unwrap();
        let expected_first = first
            .resolved_omega()
            .update(
                first.omega(),
                &second_moment.slice(ndarray::s![0..2, 0..2]).to_owned(),
                0.0,
            )
            .unwrap();
        let difference = (result[[0, 0]] - expected_first[[0, 0]]).abs();
        let relative = difference / expected_first[[0, 0]].abs().max(f64::MIN_POSITIVE);
        assert!(
            difference <= 1e-10 || relative <= 1e-9,
            "disconnected={}, independent={}, abs={difference}, rel={relative}",
            result[[0, 0]],
            expected_first[[0, 0]]
        );
        assert_eq!(result[[0, 1]], expected_first[[0, 1]]);
    }

    #[test]
    fn covariance_update_status_reports_mixed_nonfinite_rejection() {
        let prior = ParametricPrior::new(
            parameters(),
            Some(
                Omega::new()
                    .variance("ke", 0.3)
                    .fixed_variance("v", 0.5)
                    .fixed_covariance("ke", "v", 0.04),
            ),
            None,
        )
        .unwrap();
        let current = prior.omega().clone();
        let candidate = ndarray::array![[f64::NAN, 0.04], [0.04, 0.5]];
        let update = prior
            .resolved_omega()
            .update_with_status(&current, &candidate, 0.0)
            .unwrap();
        assert_eq!(update.status, CovarianceUpdateStatus::Rejected);
        assert_eq!(update.matrix, current);
        assert_eq!(
            update.rejection_reason,
            Some(CovarianceUpdateRejectionReason::CandidateNotFiniteSymmetric)
        );
        assert!(update.solved_target.is_none());
        assert!(update.attempted_fractions.is_empty());
        assert!(update.trial_rejections.is_empty());
    }

    #[test]
    fn covariance_update_fraction_under_relaxes_the_accepted_iterate() {
        let one_parameter: ParameterSpace<UnboundedParameter> =
            [Parameter::log("x")].into_iter().collect();
        let prior =
            ParametricPrior::new(one_parameter, Some(Omega::diagonal([("x", 1.0)])), None).unwrap();
        let current = ndarray::array![[1.0]];
        let candidate = ndarray::array![[4.0]];

        let full = prior
            .resolved_omega()
            .update_with_status(&current, &candidate, 0.0)
            .unwrap();
        let limited = prior
            .resolved_omega()
            .update_with_status_and_max_fraction(&current, &candidate, 0.0, 0.1)
            .unwrap();

        assert_eq!(full.status, CovarianceUpdateStatus::Accepted);
        assert_eq!(full.matrix, candidate);
        assert_eq!(limited.status, CovarianceUpdateStatus::Accepted);
        assert!((limited.matrix[[0, 0]] - 1.3).abs() <= 1e-12);
        assert!(
            covariance_objective(&limited.matrix, &candidate).unwrap()
                <= covariance_objective(&current, &candidate).unwrap()
        );
    }

    #[test]
    fn uncapped_update_preserves_legacy_floor_after_backtracking() {
        let three_parameters: ParameterSpace<UnboundedParameter> = [
            Parameter::log("x"),
            Parameter::log("y"),
            Parameter::log("z"),
        ]
        .into_iter()
        .collect();
        let declaration = Omega::new()
            .variance("x", 1.0)
            .variance("y", 1.0)
            .variance("z", 1.0)
            .covariance("x", "y", 0.0);
        let prior = ParametricPrior::new(three_parameters, Some(declaration), None).unwrap();
        let current = ndarray::Array2::eye(3);
        let candidate = ndarray::array![[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.01]];

        let legacy = prior
            .resolved_omega()
            .update_with_status(&current, &candidate, 0.1)
            .unwrap();
        let capped = prior
            .resolved_omega()
            .update_with_status_and_max_fraction(&current, &candidate, 0.1, 1.0)
            .unwrap();

        assert_eq!(legacy.status, CovarianceUpdateStatus::Accepted);
        assert_eq!(capped.status, CovarianceUpdateStatus::Accepted);
        assert!((legacy.matrix[[0, 1]] - 0.5).abs() <= 1e-12);
        assert!((legacy.matrix[[2, 2]] - 0.505).abs() <= 1e-12);
        assert!((capped.matrix[[2, 2]] - 0.55).abs() <= 1e-12);
    }

    #[test]
    fn covariance_floor_cannot_bypass_the_displacement_fraction() {
        let one_parameter: ParameterSpace<UnboundedParameter> =
            [Parameter::log("x")].into_iter().collect();
        let prior = ParametricPrior::new(one_parameter, Some(Omega::diagonal([("x", 0.01)])), None)
            .unwrap();
        let current = ndarray::array![[0.01]];
        let candidate = ndarray::array![[0.2]];

        let update = prior
            .resolved_omega()
            .update_with_status_and_max_fraction(&current, &candidate, 0.1, 0.1)
            .unwrap();

        // Reaching the 0.1 floor would require a displacement larger than 0.1
        // of the solved-target displacement. Reject rather than bypass the cap.
        assert_eq!(update.status, CovarianceUpdateStatus::Rejected);
        assert_eq!(update.matrix, current);
    }

    #[test]
    fn covariance_update_status_reports_floor_rejection_when_candidate_equals_current() {
        let one_parameter: ParameterSpace<UnboundedParameter> =
            [Parameter::log("x")].into_iter().collect();
        let prior =
            ParametricPrior::new(one_parameter, Some(Omega::diagonal([("x", 1.0)])), None).unwrap();
        let improving = prior
            .resolved_omega()
            .update_with_status(&ndarray::array![[1.0]], &ndarray::array![[0.01]], 0.1)
            .unwrap();
        assert_eq!(improving.status, CovarianceUpdateStatus::Accepted);
        assert_eq!(improving.matrix, ndarray::array![[0.1]]);

        let current = ndarray::array![[0.01]];
        let rejected = prior
            .resolved_omega()
            .update_with_status(&current, &current, 0.1)
            .unwrap();

        assert_eq!(rejected.status, CovarianceUpdateStatus::Rejected);
        assert_eq!(rejected.matrix, current);
        assert_eq!(
            rejected.rejection_reason,
            Some(CovarianceUpdateRejectionReason::BacktrackingExhausted)
        );
        assert_eq!(rejected.attempted_fractions.len(), 16);
        assert_eq!(rejected.trial_rejections.len(), 16);
        assert!(rejected
            .trial_rejections
            .iter()
            .all(|reason| *reason == CovarianceTrialRejectionReason::ObjectiveIncrease));
    }
}
