use faer::{Col, Mat};
use serde::{Deserialize, Serialize};

use crate::compile::OccasionDesign;
use crate::estimation::parametric::transforms::ParameterTransform;
use crate::estimation::parametric::ResidualErrorEstimates;
use crate::estimation::parametric::{IndividualEstimates, Population};
use crate::model::{
    CovariateModel, ParameterTransform as ModelParameterTransform, VariabilityModel,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhiVector(pub Vec<f64>);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PsiVector(pub Vec<f64>);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtaVector(pub Vec<f64>);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KappaVector(pub Vec<f64>);

impl PhiVector {
    pub fn to_col(&self) -> Col<f64> {
        Col::from_fn(self.0.len(), |index| self.0[index])
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

impl PsiVector {
    pub fn to_col(&self) -> Col<f64> {
        Col::from_fn(self.0.len(), |index| self.0[index])
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

impl EtaVector {
    pub fn to_col(&self) -> Col<f64> {
        Col::from_fn(self.0.len(), |index| self.0[index])
    }
}

impl KappaVector {
    pub fn to_col(&self) -> Col<f64> {
        Col::from_fn(self.0.len(), |index| self.0[index])
    }
}

impl From<&Col<f64>> for PhiVector {
    fn from(value: &Col<f64>) -> Self {
        Self(col_to_vec(value))
    }
}

impl From<&Col<f64>> for PsiVector {
    fn from(value: &Col<f64>) -> Self {
        Self(col_to_vec(value))
    }
}

impl From<&Col<f64>> for EtaVector {
    fn from(value: &Col<f64>) -> Self {
        Self(col_to_vec(value))
    }
}

impl From<&Col<f64>> for KappaVector {
    fn from(value: &Col<f64>) -> Self {
        Self(col_to_vec(value))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhiTable(pub Vec<Vec<f64>>);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PsiTable(pub Vec<Vec<f64>>);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtaTable(pub Vec<Vec<f64>>);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OccasionKappa {
    pub subject_index: usize,
    pub occasion_index: usize,
    pub values: KappaVector,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OccasionKappaTable(pub Vec<OccasionKappa>);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffects {
    pub parameter_names: Vec<String>,
    pub population_mean: PsiVector,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomEffects {
    pub covariance: Vec<Vec<f64>>,
    pub standard_deviations: Vec<f64>,
    pub correlation: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualState {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformSet {
    pub transforms: Vec<ParametricTransformKind>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParametricTransformKind {
    Identity,
    LogNormal,
    Logit,
    Probit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovariateState {
    pub subject_effects: Option<CovariateEffectsSnapshot>,
    pub occasion_effects: Option<CovariateEffectsSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovariateEffectsSnapshot {
    pub parameter_names: Vec<String>,
    pub column_names: Vec<String>,
    pub covariate_mask: Vec<Vec<bool>>,
    pub coefficients: Vec<f64>,
    pub estimate_coefficients: Vec<bool>,
    pub values: Vec<Vec<Option<f64>>>,
}

impl CovariateEffectsSnapshot {
    pub fn from_model(model: &CovariateModel, values: Vec<Vec<Option<f64>>>) -> Self {
        Self {
            parameter_names: model.param_names().to_vec(),
            column_names: model.covariate_names().to_vec(),
            covariate_mask: model.covariate_mask().to_vec(),
            coefficients: (0..model.beta().nrows())
                .map(|index| model.beta()[index])
                .collect(),
            estimate_coefficients: model.estimate_beta().to_vec(),
            values,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndividualEffectsState {
    pub subject_eta: EtaTable,
    pub subject_psi: PsiTable,
    pub occasion_kappa: Option<OccasionKappaTable>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricModelState {
    pub fixed_effects: FixedEffects,
    pub random_effects: RandomEffects,
    pub residual: ResidualState,
    pub transforms: TransformSet,
    pub covariates: CovariateState,
    pub variability: VariabilityModel,
}

impl ParametricModelState {
    pub fn from_population_and_sigma(
        population: &Population,
        sigma: &ResidualErrorEstimates,
    ) -> Self {
        let parameter_names = population.param_names();
        let n_parameters = parameter_names.len();

        Self {
            fixed_effects: FixedEffects {
                parameter_names,
                population_mean: PsiVector(col_to_vec(population.mu())),
            },
            random_effects: RandomEffects {
                covariance: mat_to_nested_vec(population.omega()),
                standard_deviations: col_to_vec(&population.standard_deviations()),
                correlation: mat_to_nested_vec(&population.correlation_matrix()),
            },
            residual: ResidualState {
                values: residual_values(sigma),
            },
            transforms: TransformSet {
                transforms: vec![ParametricTransformKind::Identity; n_parameters],
            },
            covariates: CovariateState {
                subject_effects: None,
                occasion_effects: None,
            },
            variability: VariabilityModel::default(),
        }
    }

    pub fn merged(self, fitted: Self) -> Self {
        let covariates = CovariateState {
            subject_effects: fitted
                .covariates
                .subject_effects
                .or(self.covariates.subject_effects),
            occasion_effects: fitted
                .covariates
                .occasion_effects
                .or(self.covariates.occasion_effects),
        };

        Self {
            fixed_effects: fitted.fixed_effects,
            random_effects: fitted.random_effects,
            residual: fitted.residual,
            transforms: self.transforms,
            covariates,
            variability: self.variability,
        }
    }
}

impl IndividualEffectsState {
    pub fn from_individual_estimates(individual_estimates: &IndividualEstimates) -> Self {
        Self::from_individual_estimates_with_occasion_design(
            individual_estimates,
            &[],
            &VariabilityModel::default(),
        )
    }

    pub fn from_individual_estimates_with_occasion_design(
        individual_estimates: &IndividualEstimates,
        occasions: &[OccasionDesign],
        variability: &VariabilityModel,
    ) -> Self {
        let subject_eta = individual_estimates
            .iter()
            .map(|individual| col_to_vec(individual.eta()))
            .collect();
        let subject_psi = individual_estimates
            .iter()
            .map(|individual| col_to_vec(individual.psi()))
            .collect();
        let n_parameters = individual_estimates
            .get(0)
            .map(|individual| individual.npar())
            .unwrap_or_else(|| variability.subject.enabled_for.len());

        Self {
            subject_eta: EtaTable(subject_eta),
            subject_psi: PsiTable(subject_psi),
            occasion_kappa: occasion_kappa_table(occasions, variability, n_parameters),
        }
    }

    pub fn with_occasion_design(
        mut self,
        occasions: &[OccasionDesign],
        variability: &VariabilityModel,
        n_parameters: usize,
    ) -> Self {
        self.occasion_kappa = occasion_kappa_table(occasions, variability, n_parameters);
        self
    }
}

impl From<&ParameterTransform> for ParametricTransformKind {
    fn from(transform: &ParameterTransform) -> Self {
        match transform {
            ParameterTransform::None => Self::Identity,
            ParameterTransform::LogNormal => Self::LogNormal,
            ParameterTransform::Logit { .. } => Self::Logit,
            ParameterTransform::Probit { .. } => Self::Probit,
        }
    }
}

impl From<&ModelParameterTransform> for ParametricTransformKind {
    fn from(transform: &ModelParameterTransform) -> Self {
        match transform {
            ModelParameterTransform::Identity => Self::Identity,
            ModelParameterTransform::LogNormal => Self::LogNormal,
            ModelParameterTransform::Logit => Self::Logit,
            ModelParameterTransform::Probit => Self::Probit,
        }
    }
}

fn col_to_vec(col: &Col<f64>) -> Vec<f64> {
    (0..col.nrows()).map(|index| col[index]).collect()
}

fn mat_to_nested_vec(mat: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..mat.nrows())
        .map(|row| (0..mat.ncols()).map(|col| mat[(row, col)]).collect())
        .collect()
}

fn residual_values(residual: &ResidualErrorEstimates) -> Vec<f64> {
    residual.as_vec()
}

fn occasion_kappa_table(
    occasions: &[OccasionDesign],
    variability: &VariabilityModel,
    n_parameters: usize,
) -> Option<OccasionKappaTable> {
    let occasion = variability.occasion.as_ref()?;
    if occasions.is_empty() || !occasion.enabled_for.iter().any(|enabled| *enabled) {
        return None;
    }

    Some(OccasionKappaTable(
        occasions
            .iter()
            .map(|occasion_design| OccasionKappa {
                subject_index: occasion_design.subject_index,
                occasion_index: occasion_design.occasion_index,
                values: KappaVector(vec![0.0; n_parameters]),
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{PhiVector, PsiVector};
    use faer::Col;

    #[test]
    fn typed_vectors_roundtrip_through_col() {
        let values = Col::from_fn(3, |index| match index {
            0 => 1.0,
            1 => 2.0,
            _ => 3.0,
        });

        let phi = PhiVector::from(&values);
        let psi = PsiVector::from(&values);

        assert_eq!(phi.to_col(), values);
        assert_eq!(psi.to_col(), values);
        assert_eq!(phi.as_slice(), &[1.0, 2.0, 3.0]);
    }
}
