use anyhow::Result;
use argmin::{
    core::{CostFunction, Error, Executor, State, TerminationReason},
    solver::neldermead::NelderMead,
};
use pharmsol::prelude::simulator::Prediction;
use pharmsol::{Equation, Predictions, Subject};

use crate::estimation::{ParametricErrorModels, ResidualErrorModel, ResidualErrorModels};

/// One output's SAEM residual-error sufficient statistic (`statrese`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct ResidualOutputStatistic {
    pub(crate) weighted_sum: f64,
    pub(crate) observation_count: usize,
    pub(crate) proportional_floor_count: usize,
    pub(crate) non_finite_prediction_count: usize,
    pub(crate) exponential_domain_violation_count: usize,
}

impl ResidualOutputStatistic {
    pub(crate) fn sigma(self) -> Option<f64> {
        if self.observation_count == 0 || !self.weighted_sum.is_finite() {
            return None;
        }
        Some((self.weighted_sum / self.observation_count as f64).sqrt())
    }
}

/// Output-indexed residual sufficient statistics.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResidualSufficientStatistics {
    outputs: Vec<ResidualOutputStatistic>,
    observations: Vec<Vec<ResidualObservation>>,
}

impl ResidualSufficientStatistics {
    pub(crate) fn zero(output_count: usize) -> Self {
        Self {
            outputs: vec![ResidualOutputStatistic::default(); output_count],
            observations: vec![Vec::new(); output_count],
        }
    }

    pub(crate) fn from_predictions<P>(predictions: &P, error_models: &ResidualErrorModels) -> Self
    where
        P: Predictions,
    {
        let mut statistics = Self::zero(error_models.len());
        predictions.for_each_prediction(|prediction| {
            statistics.accumulate_prediction(prediction, error_models);
        });
        statistics
    }

    fn accumulate_prediction(
        &mut self,
        prediction: &Prediction,
        error_models: &ResidualErrorModels,
    ) {
        let Some(observation) = prediction.observation() else {
            return;
        };
        let outeq = prediction.outeq();
        let Some(model) = error_models.get(outeq) else {
            return;
        };
        self.accumulate_values(outeq, model, observation, prediction.prediction());
    }

    fn accumulate_values(
        &mut self,
        outeq: usize,
        model: &ResidualErrorModel,
        observation: f64,
        prediction: f64,
    ) {
        let Some(statistic) = self.outputs.get_mut(outeq) else {
            return;
        };
        statistic.weighted_sum += weighted_squared_residual(model, observation, prediction);
        statistic.observation_count += 1;
        if let Some(observations) = self.observations.get_mut(outeq) {
            observations.push(ResidualObservation {
                observation,
                prediction,
            });
        }
        if !prediction.is_finite() {
            statistic.non_finite_prediction_count += 1;
        } else if matches!(model, ResidualErrorModel::Proportional { .. })
            && prediction.powi(2) <= PROPORTIONAL_PREDICTION_SQUARED_FLOOR
        {
            statistic.proportional_floor_count += 1;
        }
        if matches!(model, ResidualErrorModel::Exponential { .. })
            && (!observation.is_finite()
                || observation <= 0.0
                || !prediction.is_finite()
                || prediction <= 0.0)
        {
            statistic.exponential_domain_violation_count += 1;
        }
    }

    pub(crate) fn add_assign(&mut self, observed: &Self) {
        if self.outputs.len() < observed.outputs.len() {
            self.outputs
                .resize(observed.outputs.len(), ResidualOutputStatistic::default());
            self.observations
                .resize_with(observed.outputs.len(), Vec::new);
        }
        for (output_index, (total, value)) in
            self.outputs.iter_mut().zip(&observed.outputs).enumerate()
        {
            total.weighted_sum += value.weighted_sum;
            total.observation_count += value.observation_count;
            total.proportional_floor_count += value.proportional_floor_count;
            total.non_finite_prediction_count += value.non_finite_prediction_count;
            total.exponential_domain_violation_count += value.exponential_domain_violation_count;
            self.observations[output_index].extend_from_slice(&observed.observations[output_index]);
        }
    }

    pub(crate) fn stochastic_update(&self, observed: Self, step_size: f64) -> Self {
        let mut updated = self.clone();
        if updated.outputs.len() < observed.outputs.len() {
            updated
                .outputs
                .resize(observed.outputs.len(), ResidualOutputStatistic::default());
            updated
                .observations
                .resize_with(observed.outputs.len(), Vec::new);
        }
        updated.observations = observed.observations;
        for (current, value) in updated.outputs.iter_mut().zip(observed.outputs) {
            current.weighted_sum += step_size * (value.weighted_sum - current.weighted_sum);
            current.observation_count = value.observation_count.max(current.observation_count);
            current.proportional_floor_count = value.proportional_floor_count;
            current.non_finite_prediction_count = value.non_finite_prediction_count;
            current.exponential_domain_violation_count = value.exponential_domain_violation_count;
        }
        updated
    }

    pub(crate) fn output(&self, outeq: usize) -> Option<ResidualOutputStatistic> {
        self.outputs.get(outeq).copied()
    }

    pub(crate) fn observations(&self, outeq: usize) -> Option<&[ResidualObservation]> {
        self.observations.get(outeq).map(Vec::as_slice)
    }
}

pub(crate) fn residual_statistics_for_subject<E: Equation>(
    equation: &E,
    subject: &Subject,
    parameters: &[f64],
    error_models: &ResidualErrorModels,
) -> Result<ResidualSufficientStatistics> {
    let predictions = equation.estimate_predictions_dense(subject, parameters)?;
    Ok(ResidualSufficientStatistics::from_predictions(
        &predictions,
        error_models,
    ))
}

/// Denominator floor used by the proportional residual-statistic update.
///
/// This applies to squared predictions, not to the proportional coefficient.
/// It keeps an exactly zero prediction in the statistic with a very large
/// penalty instead of silently dropping the observation.
const PROPORTIONAL_PREDICTION_SQUARED_FLOOR: f64 = f64::EPSILON;

pub(crate) fn weighted_squared_residual(
    model: &ResidualErrorModel,
    observation: f64,
    prediction: f64,
) -> f64 {
    match model {
        ResidualErrorModel::Exponential { .. } => {
            if observation.is_finite()
                && observation > 0.0
                && prediction.is_finite()
                && prediction > 0.0
            {
                (observation.ln() - prediction.ln()).powi(2)
            } else {
                f64::NAN
            }
        }
        ResidualErrorModel::Constant { .. } => (observation - prediction).powi(2),
        ResidualErrorModel::Proportional { .. } => {
            let residual_sq = (observation - prediction).powi(2);
            residual_sq
                / prediction
                    .powi(2)
                    .max(PROPORTIONAL_PREDICTION_SQUARED_FLOOR)
        }
        ResidualErrorModel::Combined { a, b } => {
            let residual_sq = (observation - prediction).powi(2);
            let variance = (a.powi(2) + b.powi(2) * prediction.powi(2)).max(f64::EPSILON);
            residual_sq / variance
        }
        ResidualErrorModel::CorrelatedCombined { a, b, rho } => {
            let residual_sq = (observation - prediction).powi(2);
            let variance =
                (a.powi(2) + 2.0 * rho * a * b * prediction + b.powi(2) * prediction.powi(2))
                    .max(f64::EPSILON);
            residual_sq / variance
        }
    }
}

pub(crate) fn primary_sigma_parameter(model: &ResidualErrorModel) -> f64 {
    match model {
        ResidualErrorModel::Constant { a } => *a,
        ResidualErrorModel::Proportional { b } => *b,
        ResidualErrorModel::Combined { a, .. }
        | ResidualErrorModel::CorrelatedCombined { a, .. } => *a,
        ResidualErrorModel::Exponential { sigma } => *sigma,
    }
}

pub(crate) fn primary_sigma_parameters(error_models: &ResidualErrorModels) -> Vec<f64> {
    error_models
        .iter()
        .map(|(_, model)| primary_sigma_parameter(model))
        .collect()
}

pub(crate) fn update_estimated_combined_residual_model(
    error_models: &mut ParametricErrorModels,
    outeq: usize,
    additive_sd: f64,
    proportional_sd: f64,
) {
    if !error_models.is_estimated(outeq) {
        return;
    }
    if let Some(slot) = error_models.models_mut().get_mut(outeq) {
        *slot = ResidualErrorModel::combined(additive_sd, proportional_sd);
    }
}

pub(crate) fn update_estimated_correlated_combined_residual_model(
    error_models: &mut ParametricErrorModels,
    outeq: usize,
    additive_sd: f64,
    proportional_sd: f64,
    correlation: f64,
) {
    if !error_models.is_estimated(outeq) {
        return;
    }
    if let Some(slot) = error_models.models_mut().get_mut(outeq) {
        *slot = ResidualErrorModel::correlated_combined(additive_sd, proportional_sd, correlation);
    }
}

pub(crate) fn update_estimated_simple_residual_model_with_sigma(
    error_models: &mut ParametricErrorModels,
    outeq: usize,
    sigma: f64,
) {
    if !error_models.is_estimated(outeq) {
        return;
    }
    let Some(model) = error_models.models().get(outeq).cloned() else {
        return;
    };
    let Some(updated) = simple_model_with_sigma(&model, sigma) else {
        return;
    };
    if let Some(slot) = error_models.models_mut().get_mut(outeq) {
        *slot = updated;
    }
}

fn simple_model_with_sigma(model: &ResidualErrorModel, sigma: f64) -> Option<ResidualErrorModel> {
    match model {
        ResidualErrorModel::Constant { .. } => Some(ResidualErrorModel::constant(sigma)),
        ResidualErrorModel::Proportional { .. } => Some(ResidualErrorModel::proportional(sigma)),
        ResidualErrorModel::Exponential { .. } => Some(ResidualErrorModel::exponential(sigma)),
        ResidualErrorModel::Combined { .. } | ResidualErrorModel::CorrelatedCombined { .. } => None,
    }
}

/// Observation/prediction pair retained for residual-NLL optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResidualObservation {
    pub(crate) observation: f64,
    pub(crate) prediction: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CombinedResidualSolution {
    pub(crate) additive_sd: f64,
    pub(crate) proportional_sd: f64,
    pub(crate) objective: f64,
    pub(crate) converged: bool,
    pub(crate) iterations: u64,
    pub(crate) termination: String,
}

const RESIDUAL_OPTIMIZER_NON_FINITE_PENALTY: f64 = 1e100;
const COMBINED_ADDITIVE_COLLAPSE_THRESHOLD: f64 = 1e-3;
// Upper log-sigma bound of 5.
pub(crate) const RESIDUAL_OPTIMIZER_MAX_SIGMA: f64 = 148.413_159_102_576_6;

/// Conditional Gaussian residual NLL for the quadrature combined model.
///
/// This is the active `sqrt(a^2 + b^2 f^2)` model, equivalently the
/// `sigma_add^2 + f^2 sigma_prop^2` variance, including the log-variance term.
pub(crate) fn combined_additive_sigma_collapsed(additive_sd: f64, estimated: bool) -> bool {
    estimated && additive_sd.is_finite() && additive_sd <= COMBINED_ADDITIVE_COLLAPSE_THRESHOLD
}

pub(crate) fn combined_residual_nll(
    observations: &[ResidualObservation],
    additive_sd: f64,
    proportional_sd: f64,
) -> f64 {
    if observations.is_empty()
        || !additive_sd.is_finite()
        || !proportional_sd.is_finite()
        || additive_sd < 0.0
        || proportional_sd < 0.0
        || (additive_sd == 0.0 && proportional_sd == 0.0)
    {
        return f64::INFINITY;
    }

    observations.iter().fold(0.0, |total, value| {
        if !value.observation.is_finite() || !value.prediction.is_finite() {
            return f64::INFINITY;
        }
        let variance = (additive_sd.powi(2) + proportional_sd.powi(2) * value.prediction.powi(2))
            .max(f64::EPSILON);
        total + 0.5 * (variance.ln() + (value.observation - value.prediction).powi(2) / variance)
    })
}

struct CombinedResidualCost<'a> {
    observations: &'a [ResidualObservation],
    minimum_sigma: f64,
    initial_sigmas: [f64; 2],
    estimated: [bool; 2],
}

impl CombinedResidualCost<'_> {
    fn unpack(&self, log_sigmas: &[f64]) -> Option<[f64; 2]> {
        if log_sigmas.len() != self.estimated.iter().filter(|value| **value).count() {
            return None;
        }
        let mut sigmas = self.initial_sigmas;
        let mut coordinate = 0;
        for (sigma, estimated) in sigmas.iter_mut().zip(self.estimated) {
            if estimated {
                *sigma = log_sigmas[coordinate].exp();
                coordinate += 1;
            }
        }
        Some(sigmas)
    }
}

impl CostFunction for CombinedResidualCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, log_sigmas: &Self::Param) -> std::result::Result<Self::Output, Error> {
        let Some([additive_sd, proportional_sd]) = self.unpack(log_sigmas) else {
            return Ok(RESIDUAL_OPTIMIZER_NON_FINITE_PENALTY);
        };
        for (component, sigma) in [additive_sd, proportional_sd].into_iter().enumerate() {
            if self.estimated[component]
                && (sigma < self.minimum_sigma || sigma > RESIDUAL_OPTIMIZER_MAX_SIGMA)
            {
                return Ok(RESIDUAL_OPTIMIZER_NON_FINITE_PENALTY);
            }
        }
        let objective = combined_residual_nll(self.observations, additive_sd, proportional_sd);
        Ok(if objective.is_finite() {
            objective
        } else {
            RESIDUAL_OPTIMIZER_NON_FINITE_PENALTY
        })
    }
}

pub(crate) fn optimize_combined_residual(
    observations: &[ResidualObservation],
    initial_additive_sd: f64,
    initial_proportional_sd: f64,
    estimated: [bool; 2],
    minimum_sigma: f64,
    max_iterations: u64,
) -> Result<CombinedResidualSolution> {
    anyhow::ensure!(
        !observations.is_empty(),
        "combined residual optimization requires observations"
    );
    anyhow::ensure!(
        estimated.iter().any(|value| *value),
        "combined residual optimization requires an estimated component"
    );
    anyhow::ensure!(
        minimum_sigma.is_finite()
            && minimum_sigma > 0.0
            && minimum_sigma < RESIDUAL_OPTIMIZER_MAX_SIGMA,
        "combined residual minimum sigma must be finite, positive, and below the maximum"
    );
    let initial_sigmas = [initial_additive_sd, initial_proportional_sd];
    for (component, sigma) in initial_sigmas.into_iter().enumerate() {
        anyhow::ensure!(
            sigma.is_finite()
                && if estimated[component] {
                    sigma > 0.0
                } else {
                    sigma >= 0.0
                },
            "combined residual SD components must be finite and estimated components must be positive"
        );
    }

    let initial = initial_sigmas
        .into_iter()
        .zip(estimated)
        .filter(|&(_sigma, estimate)| estimate)
        .map(|(sigma, _estimate)| {
            sigma
                .clamp(minimum_sigma, RESIDUAL_OPTIMIZER_MAX_SIGMA)
                .ln()
        })
        .collect::<Vec<_>>();
    let mut simplex = Vec::with_capacity(initial.len() + 1);
    simplex.push(initial.clone());
    for component in 0..initial.len() {
        let mut point = initial.clone();
        point[component] += 0.2;
        simplex.push(point);
    }
    let cost = CombinedResidualCost {
        observations,
        minimum_sigma,
        initial_sigmas,
        estimated,
    };
    let solver = NelderMead::new(simplex).with_sd_tolerance(1e-8)?;
    let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(max_iterations))
        .run()?;
    let state = result.state;
    let best_is_valid =
        state.best_cost.is_finite() && state.best_cost < RESIDUAL_OPTIMIZER_NON_FINITE_PENALTY;
    let best = state
        .best_param
        .clone()
        .filter(|_| best_is_valid)
        .unwrap_or(initial);
    let cost = CombinedResidualCost {
        observations,
        minimum_sigma,
        initial_sigmas,
        estimated,
    };
    let [additive_sd, proportional_sd] = cost.unpack(&best).ok_or_else(|| {
        anyhow::anyhow!("combined residual optimizer returned invalid coordinates")
    })?;
    let objective = combined_residual_nll(observations, additive_sd, proportional_sd);
    let termination_reason = state.get_termination_reason();

    Ok(CombinedResidualSolution {
        additive_sd,
        proportional_sd,
        objective,
        converged: best_is_valid
            && matches!(termination_reason, Some(TerminationReason::SolverConverged)),
        iterations: state.iter,
        termination: termination_reason
            .map(ToString::to_string)
            .unwrap_or_else(|| "unknown termination".to_owned()),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CorrelatedCombinedResidualSolution {
    pub(crate) additive_sd: f64,
    pub(crate) proportional_sd: f64,
    pub(crate) correlation: f64,
    pub(crate) objective: f64,
    pub(crate) converged: bool,
    pub(crate) iterations: u64,
    pub(crate) termination: String,
}

/// Conditional Gaussian residual NLL for the within-observation correlated
/// additive/proportional family, including the log-variance term.
pub(crate) fn correlated_combined_residual_nll(
    observations: &[ResidualObservation],
    additive_sd: f64,
    proportional_sd: f64,
    correlation: f64,
) -> f64 {
    if observations.is_empty()
        || !additive_sd.is_finite()
        || additive_sd <= 0.0
        || !proportional_sd.is_finite()
        || proportional_sd <= 0.0
        || !correlation.is_finite()
        || correlation <= -1.0
        || correlation >= 1.0
    {
        return f64::INFINITY;
    }

    observations.iter().fold(0.0, |total, value| {
        if !total.is_finite() || !value.observation.is_finite() || !value.prediction.is_finite() {
            return f64::INFINITY;
        }
        let variance = additive_sd.powi(2)
            + 2.0 * correlation * additive_sd * proportional_sd * value.prediction
            + proportional_sd.powi(2) * value.prediction.powi(2);
        if !variance.is_finite() || variance <= 0.0 {
            return f64::INFINITY;
        }
        total + 0.5 * (variance.ln() + (value.observation - value.prediction).powi(2) / variance)
    })
}

struct CorrelatedCombinedResidualCost<'a> {
    observations: &'a [ResidualObservation],
    minimum_sigma: f64,
    initial: [f64; 3],
    estimated: [bool; 3],
}

impl CorrelatedCombinedResidualCost<'_> {
    fn unpack(&self, coordinates: &[f64]) -> Option<[f64; 3]> {
        if coordinates.len() != self.estimated.iter().filter(|value| **value).count() {
            return None;
        }
        let mut values = self.initial;
        let mut coordinate = 0;
        for (component, estimated) in self.estimated.into_iter().enumerate() {
            if estimated {
                values[component] = if component < 2 {
                    coordinates[coordinate].exp()
                } else {
                    coordinates[coordinate].tanh()
                };
                coordinate += 1;
            }
        }
        Some(values)
    }

    fn valid(&self, values: [f64; 3]) -> bool {
        values.into_iter().all(f64::is_finite)
            && values[0] > 0.0
            && values[1] > 0.0
            && values[2] > -1.0
            && values[2] < 1.0
            && (!self.estimated[0]
                || (values[0] >= self.minimum_sigma && values[0] <= RESIDUAL_OPTIMIZER_MAX_SIGMA))
            && (!self.estimated[1]
                || (values[1] >= self.minimum_sigma && values[1] <= RESIDUAL_OPTIMIZER_MAX_SIGMA))
    }
}

impl CostFunction for CorrelatedCombinedResidualCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, coordinates: &Self::Param) -> std::result::Result<Self::Output, Error> {
        let Some(values) = self.unpack(coordinates) else {
            return Ok(RESIDUAL_OPTIMIZER_NON_FINITE_PENALTY);
        };
        if !self.valid(values) {
            return Ok(RESIDUAL_OPTIMIZER_NON_FINITE_PENALTY);
        }
        let objective =
            correlated_combined_residual_nll(self.observations, values[0], values[1], values[2]);
        Ok(if objective.is_finite() {
            objective
        } else {
            RESIDUAL_OPTIMIZER_NON_FINITE_PENALTY
        })
    }
}

pub(crate) fn optimize_correlated_combined_residual(
    observations: &[ResidualObservation],
    initial_additive_sd: f64,
    initial_proportional_sd: f64,
    initial_correlation: f64,
    estimated: [bool; 3],
    minimum_sigma: f64,
    max_iterations: u64,
) -> Result<CorrelatedCombinedResidualSolution> {
    anyhow::ensure!(
        !observations.is_empty(),
        "correlated-combined residual optimization requires observations"
    );
    anyhow::ensure!(
        estimated.iter().any(|value| *value),
        "correlated-combined residual optimization requires an estimated component"
    );
    anyhow::ensure!(
        minimum_sigma.is_finite()
            && minimum_sigma > 0.0
            && minimum_sigma < RESIDUAL_OPTIMIZER_MAX_SIGMA,
        "correlated-combined residual minimum sigma must be finite, positive, and below the maximum"
    );
    let initial_values = [
        initial_additive_sd,
        initial_proportional_sd,
        initial_correlation,
    ];
    anyhow::ensure!(
        initial_values.into_iter().all(f64::is_finite)
            && initial_additive_sd > 0.0
            && initial_proportional_sd > 0.0
            && initial_correlation > -1.0
            && initial_correlation < 1.0,
        "correlated-combined residual components require positive SDs and correlation strictly inside (-1, 1)"
    );
    for component in 0..2 {
        if estimated[component] {
            anyhow::ensure!(
                initial_values[component] >= minimum_sigma
                    && initial_values[component] <= RESIDUAL_OPTIMIZER_MAX_SIGMA,
                "estimated correlated-combined residual SD is outside optimizer bounds"
            );
        }
    }

    let initial = initial_values
        .into_iter()
        .enumerate()
        .filter(|(component, _)| estimated[*component])
        .map(|(component, value)| {
            if component < 2 {
                value.ln()
            } else {
                value.atanh()
            }
        })
        .collect::<Vec<_>>();
    let mut simplex = Vec::with_capacity(initial.len() + 1);
    simplex.push(initial.clone());
    for component in 0..initial.len() {
        let mut point = initial.clone();
        point[component] += 0.2;
        simplex.push(point);
    }
    let cost = CorrelatedCombinedResidualCost {
        observations,
        minimum_sigma,
        initial: initial_values,
        estimated,
    };
    let solver = NelderMead::new(simplex).with_sd_tolerance(1e-8)?;
    let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(max_iterations))
        .run()?;
    let state = result.state;
    let best_is_valid =
        state.best_cost.is_finite() && state.best_cost < RESIDUAL_OPTIMIZER_NON_FINITE_PENALTY;
    anyhow::ensure!(
        best_is_valid,
        "correlated-combined residual optimizer did not return a finite valid objective"
    );
    let best = state.best_param.clone().ok_or_else(|| {
        anyhow::anyhow!("correlated-combined residual optimizer returned no candidate")
    })?;
    let cost = CorrelatedCombinedResidualCost {
        observations,
        minimum_sigma,
        initial: initial_values,
        estimated,
    };
    let [additive_sd, proportional_sd, correlation] = cost.unpack(&best).ok_or_else(|| {
        anyhow::anyhow!("correlated-combined residual optimizer returned invalid coordinates")
    })?;
    anyhow::ensure!(
        cost.valid([additive_sd, proportional_sd, correlation]),
        "correlated-combined residual optimizer returned an invalid candidate"
    );
    let objective =
        correlated_combined_residual_nll(observations, additive_sd, proportional_sd, correlation);
    anyhow::ensure!(
        objective.is_finite(),
        "correlated-combined residual optimizer returned a non-finite objective"
    );
    let termination_reason = state.get_termination_reason();

    Ok(CorrelatedCombinedResidualSolution {
        additive_sd,
        proportional_sd,
        correlation,
        objective,
        converged: best_is_valid
            && matches!(termination_reason, Some(TerminationReason::SolverConverged)),
        iterations: state.iter,
        termination: termination_reason
            .map(ToString::to_string)
            .unwrap_or_else(|| "unknown termination".to_owned()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_residual_matches_constant_and_proportional_forms() {
        assert_eq!(
            weighted_squared_residual(&ResidualErrorModel::constant(99.0), 5.0, 3.0),
            4.0
        );
        assert!(
            (weighted_squared_residual(&ResidualErrorModel::proportional(99.0), 12.0, 10.0) - 0.04)
                .abs()
                < 1e-12
        );
        assert!(
            (weighted_squared_residual(&ResidualErrorModel::proportional(99.0), -8.0, -10.0)
                - 0.04)
                .abs()
                < 1e-12
        );
        assert_eq!(
            weighted_squared_residual(&ResidualErrorModel::proportional(99.0), 1.0, 0.0),
            1.0 / PROPORTIONAL_PREDICTION_SQUARED_FLOOR
        );
    }

    #[test]
    fn exponential_statistic_is_squared_log_residual_and_rejects_nonpositive_domain() {
        let model = ResidualErrorModel::exponential(0.25);
        let expected = (12.0_f64.ln() - 10.0_f64.ln()).powi(2);

        assert!((weighted_squared_residual(&model, 12.0, 10.0) - expected).abs() < 1e-12);
        assert!(weighted_squared_residual(&model, 0.0, 10.0).is_nan());
        assert!(weighted_squared_residual(&model, 12.0, 0.0).is_nan());

        let mut statistics = ResidualSufficientStatistics::zero(1);
        statistics.accumulate_values(0, &model, 12.0, 10.0);
        statistics.accumulate_values(0, &model, 0.0, 10.0);
        statistics.accumulate_values(0, &model, 12.0, -1.0);
        statistics.accumulate_values(0, &model, 12.0, f64::NAN);
        let statistic = statistics.output(0).unwrap();
        assert_eq!(statistic.exponential_domain_violation_count, 3);
        assert_eq!(statistic.non_finite_prediction_count, 1);
    }

    #[test]
    fn exponential_fixed_trace_statistic_matches_reference_checkpoint() {
        let model = ResidualErrorModel::exponential(0.25);
        let observations = [12.0, 8.0, 4.5, 1.2];
        let predictions = [10.0, 7.5, 5.0, 1.5];
        let statrese = observations
            .into_iter()
            .zip(predictions)
            .map(|(observation, prediction)| {
                weighted_squared_residual(&model, observation, prediction)
            })
            .sum::<f64>();
        let sigma = (statrese / observations.len() as f64).sqrt();

        assert!((statrese - 0.098_300_253_535_196_28).abs() < 1e-15);
        assert!((sigma - 0.156_764_356_228_701_02).abs() < 1e-15);
    }

    #[test]
    fn residual_statistics_remain_separate_by_output() {
        let model = ResidualErrorModel::constant(1.0);
        let mut statistics = ResidualSufficientStatistics::zero(2);
        statistics.accumulate_values(0, &model, 10.0, 8.0);
        statistics.accumulate_values(1, &model, 10.0, 5.0);

        assert_eq!(statistics.output(0).unwrap().weighted_sum, 4.0);
        assert_eq!(statistics.output(0).unwrap().observation_count, 1);
        assert_eq!(statistics.output(1).unwrap().weighted_sum, 25.0);
        assert_eq!(statistics.output(1).unwrap().observation_count, 1);
    }

    #[test]
    fn proportional_statistics_count_floor_and_non_finite_predictions() {
        let model = ResidualErrorModel::proportional(0.1);
        let mut statistics = ResidualSufficientStatistics::zero(1);
        statistics.accumulate_values(0, &model, 1.0, 0.0);
        statistics.accumulate_values(0, &model, 1.0, f64::NAN);

        let output = statistics.output(0).unwrap();
        assert_eq!(output.observation_count, 2);
        assert_eq!(output.proportional_floor_count, 1);
        assert_eq!(output.non_finite_prediction_count, 1);
        assert!(!output.weighted_sum.is_finite());
        assert_eq!(output.sigma(), None);
    }

    #[test]
    fn residual_statistic_stochastic_update_matches_reference_shape() {
        let current = ResidualSufficientStatistics {
            outputs: vec![
                ResidualOutputStatistic {
                    weighted_sum: 10.0,
                    observation_count: 5,
                    ..Default::default()
                },
                ResidualOutputStatistic {
                    weighted_sum: 8.0,
                    observation_count: 4,
                    ..Default::default()
                },
            ],
            observations: vec![Vec::new(), Vec::new()],
        };
        let observed = ResidualSufficientStatistics {
            outputs: vec![
                ResidualOutputStatistic {
                    weighted_sum: 20.0,
                    observation_count: 5,
                    ..Default::default()
                },
                ResidualOutputStatistic {
                    weighted_sum: 4.0,
                    observation_count: 4,
                    ..Default::default()
                },
            ],
            observations: vec![Vec::new(), Vec::new()],
        };
        let updated = current.stochastic_update(observed, 0.25);
        assert_eq!(updated.output(0).unwrap().weighted_sum, 12.5);
        assert_eq!(
            updated.output(0).and_then(ResidualOutputStatistic::sigma),
            Some((12.5_f64 / 5.0).sqrt())
        );
        assert_eq!(updated.output(1).unwrap().weighted_sum, 7.0);
        assert_eq!(
            updated.output(1).and_then(ResidualOutputStatistic::sigma),
            Some((7.0_f64 / 4.0).sqrt())
        );
    }

    #[test]
    fn combined_additive_collapse_warning_ignores_fixed_components() {
        assert!(combined_additive_sigma_collapsed(5e-4, true));
        assert!(!combined_additive_sigma_collapsed(5e-4, false));
        assert!(!combined_additive_sigma_collapsed(0.5, true));
    }

    #[test]
    fn combined_residual_nll_matches_quadrature_gaussian_formula() {
        let observations = [ResidualObservation {
            observation: 3.0,
            prediction: 2.0,
        }];
        let variance = 0.5_f64.powi(2) + 0.1_f64.powi(2) * 2.0_f64.powi(2);
        let expected = 0.5 * (variance.ln() + 1.0 / variance);

        assert!((combined_residual_nll(&observations, 0.5, 0.1) - expected).abs() < 1e-12);
    }

    #[test]
    fn combined_residual_optimizer_recovers_known_additive_and_proportional_scales() {
        let true_additive = 0.3;
        let true_proportional = 0.1;
        let mut observations = Vec::new();
        for prediction in [1.0_f64, 2.0, 4.0, 8.0] {
            let sd = (true_additive * true_additive
                + true_proportional * true_proportional * prediction * prediction)
                .sqrt();
            observations.push(ResidualObservation {
                observation: prediction - sd,
                prediction,
            });
            observations.push(ResidualObservation {
                observation: prediction + sd,
                prediction,
            });
        }

        let initial_objective = combined_residual_nll(&observations, 0.6, 0.2);
        let solution =
            optimize_combined_residual(&observations, 0.6, 0.2, [true, true], 1e-6, 500).unwrap();

        assert!(solution.objective < initial_objective);
        assert!(solution.converged);
        assert!((solution.additive_sd - true_additive).abs() / true_additive < 0.05);
        assert!((solution.proportional_sd - true_proportional).abs() / true_proportional < 0.05);
        assert!(solution.iterations > 0);
        assert!(!solution.termination.is_empty());

        let fixed_additive =
            optimize_combined_residual(&observations, true_additive, 0.2, [false, true], 1e-6, 500)
                .unwrap();
        assert_eq!(fixed_additive.additive_sd, true_additive);
        assert!(
            (fixed_additive.proportional_sd - true_proportional).abs() / true_proportional < 0.05
        );

        let fixed_proportional = optimize_combined_residual(
            &observations,
            0.6,
            true_proportional,
            [true, false],
            1e-6,
            500,
        )
        .unwrap();
        assert_eq!(fixed_proportional.proportional_sd, true_proportional);
        assert!((fixed_proportional.additive_sd - true_additive).abs() / true_additive < 0.05);
    }

    fn correlated_truth_observations() -> Vec<ResidualObservation> {
        let (a, b, rho) = (0.7_f64, 0.25_f64, -0.3_f64);
        let mut observations = Vec::new();
        for prediction in [-2.0_f64, 0.5, 3.0] {
            let variance = a * a + 2.0 * rho * a * b * prediction + b * b * prediction * prediction;
            let sd = variance.sqrt();
            observations.push(ResidualObservation {
                observation: prediction - sd,
                prediction,
            });
            observations.push(ResidualObservation {
                observation: prediction + sd,
                prediction,
            });
        }
        observations
    }

    #[test]
    fn correlated_combined_nll_matches_signed_formula_and_rho_zero_combined() {
        for prediction in [-2.0_f64, 0.0, 3.0] {
            let observation = prediction + 0.4;
            let observations = [ResidualObservation {
                observation,
                prediction,
            }];
            let variance = 0.7_f64.powi(2)
                + 2.0 * -0.3 * 0.7 * 0.25 * prediction
                + 0.25_f64.powi(2) * prediction.powi(2);
            let expected = 0.5 * (variance.ln() + 0.4_f64.powi(2) / variance);
            assert!(
                (correlated_combined_residual_nll(&observations, 0.7, 0.25, -0.3) - expected).abs()
                    < 1e-12
            );
            assert_eq!(
                correlated_combined_residual_nll(&observations, 0.7, 0.25, 0.0),
                combined_residual_nll(&observations, 0.7, 0.25)
            );
        }
    }

    #[test]
    fn correlated_combined_optimizer_recovers_all_components_and_preserves_fixed_values() {
        let observations = correlated_truth_observations();
        let truth = [0.7, 0.25, -0.3];
        let all_free = optimize_correlated_combined_residual(
            &observations,
            0.45,
            0.4,
            0.25,
            [true, true, true],
            1e-6,
            1_000,
        )
        .unwrap();
        assert!(all_free.converged);
        assert!((all_free.additive_sd - truth[0]).abs() < 2e-3);
        assert!((all_free.proportional_sd - truth[1]).abs() < 2e-3);
        assert!((all_free.correlation - truth[2]).abs() < 2e-3);
        assert!(all_free.objective.is_finite());
        assert!(all_free.iterations > 0);

        for estimated in [
            [false, true, true],
            [true, false, true],
            [true, true, false],
            [false, false, true],
            [false, true, false],
            [true, false, false],
        ] {
            let solution = optimize_correlated_combined_residual(
                &observations,
                truth[0],
                truth[1],
                truth[2],
                estimated,
                1e-6,
                1_000,
            )
            .unwrap();
            let values = [
                solution.additive_sd,
                solution.proportional_sd,
                solution.correlation,
            ];
            for component in 0..3 {
                assert!(values[component].is_finite());
                if !estimated[component] {
                    assert_eq!(values[component], truth[component]);
                } else {
                    assert!((values[component] - truth[component]).abs() < 2e-3);
                }
            }
            assert!(solution.correlation > -1.0 && solution.correlation < 1.0);
        }
    }

    #[test]
    fn correlated_combined_optimizer_rejects_when_no_valid_candidate_exists() {
        let invalid = [ResidualObservation {
            observation: f64::NAN,
            prediction: 1.0,
        }];
        let result = optimize_correlated_combined_residual(
            &invalid,
            0.7,
            0.25,
            -0.3,
            [true, true, true],
            1e-6,
            50,
        );
        assert!(result.is_err());
    }

    #[test]
    fn sigma_update_mutates_only_estimated_simple_models() {
        use crate::estimation::ParametricErrorModel;

        let mut models = ParametricErrorModels::new()
            .add(0, "first", ResidualErrorModel::constant(0.5).into())
            .add(
                1,
                "second",
                ParametricErrorModel::from(ResidualErrorModel::proportional(0.1)).fixed(),
            )
            .add(2, "third", ResidualErrorModel::combined(0.5, 0.1).into())
            .add(3, "fourth", ResidualErrorModel::exponential(0.2).into());

        update_estimated_simple_residual_model_with_sigma(&mut models, 0, 2.0);
        update_estimated_simple_residual_model_with_sigma(&mut models, 1, 3.0);
        update_estimated_simple_residual_model_with_sigma(&mut models, 2, 4.0);
        update_estimated_simple_residual_model_with_sigma(&mut models, 3, 0.3);

        assert_eq!(
            primary_sigma_parameters(models.models()),
            vec![2.0, 0.1, 0.5, 0.3]
        );
        assert_eq!(models.output_name(0), Some("first"));
        assert_eq!(models.output_name(1), Some("second"));
        assert_eq!(models.output_name(2), Some("third"));
        assert_eq!(models.output_name(3), Some("fourth"));
        assert_eq!(
            models.models().get(0),
            Some(&ResidualErrorModel::constant(2.0))
        );
        assert_eq!(
            models.models().get(1),
            Some(&ResidualErrorModel::proportional(0.1))
        );
        assert_eq!(
            models.models().get(2),
            Some(&ResidualErrorModel::combined(0.5, 0.1))
        );
        assert_eq!(
            models.models().get(3),
            Some(&ResidualErrorModel::exponential(0.3))
        );
    }
}
