//! Prediction-dependent residual error models for parametric estimation.
//!
//! Constant, proportional, combined, correlated-combined, and exponential
//! models provide residual scale calculations and simulation. Estimation uses
//! separate canonical
//! likelihood scoring routines.

use serde::{Deserialize, Serialize};

/// Residual standard deviation as a function of the model prediction.
///
/// # Examples
///
/// ```rust
/// use pmcore::ResidualErrorModel;
///
/// // Constant (additive) error: σ = 0.5
/// let constant = ResidualErrorModel::Constant { a: 0.5 };
/// assert!((constant.sigma(100.0) - 0.5).abs() < 1e-10);
///
/// // Proportional error: σ = 0.1 * |f|
/// let proportional = ResidualErrorModel::Proportional { b: 0.1 };
/// assert!((proportional.sigma(100.0) - 10.0).abs() < 1e-10);
///
/// // Combined error: σ = sqrt(0.5² + 0.1² * f²)
/// let combined = ResidualErrorModel::Combined { a: 0.5, b: 0.1 };
/// // For f=100: σ = sqrt(0.25 + 100) = sqrt(100.25) ≈ 10.01
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ResidualErrorModel {
    /// Constant (additive) error model
    ///
    /// σ = a
    ///
    /// Error is independent of the predicted value.
    /// Appropriate when measurement error is constant regardless of concentration.
    Constant {
        /// Additive error standard deviation
        a: f64,
    },

    /// Proportional error model
    ///
    /// σ = b * |f|
    ///
    /// Error scales linearly with the prediction.
    /// Appropriate when measurement error is a constant percentage of the value.
    ///
    /// Note: Uses |f| to handle negative predictions gracefully.
    Proportional {
        /// Proportional coefficient (e.g., 0.1 = 10% CV)
        b: f64,
    },

    /// Combined (additive + proportional) error model
    ///
    /// σ = sqrt(a² + b² * f²)
    ///
    /// This is the standard quadrature combined-error model:
    /// ```R
    /// g <- cutoff(sqrt(ab[1]^2 + ab[2]^2 * f^2))
    /// ```
    ///
    /// The combined model:
    /// - Dominates at low concentrations (a term)
    /// - Scales proportionally at high concentrations (b term)
    Combined {
        /// Additive component (a)
        a: f64,
        /// Proportional component (b)
        b: f64,
    },

    /// Within-observation correlated additive/proportional error model.
    ///
    /// `Y = f + epsilon_a + f * epsilon_p`, where the component standard
    /// deviations are `a` and `b` and their correlation is `rho`. Therefore
    /// `Var(Y | f) = a² + 2 rho a b f + b² f²`. Observations remain
    /// conditionally independent: this does not model serial or cross-output
    /// residual correlation.
    CorrelatedCombined {
        /// Additive component standard deviation.
        a: f64,
        /// Proportional component standard deviation.
        b: f64,
        /// Within-observation additive/proportional correlation.
        rho: f64,
    },

    /// Exponential error model (for log-transformed data)
    ///
    /// σ = σ_exp (constant on log scale)
    ///
    /// When data is analyzed on the log scale:
    /// ```text
    /// log(Y) = log(f) + ε, where ε ~ N(0, σ²)
    /// ```
    ///
    /// This corresponds to multiplicative error on the original scale.
    Exponential {
        /// Error standard deviation on log scale
        sigma: f64,
    },
}

impl Default for ResidualErrorModel {
    fn default() -> Self {
        // Default to constant error with σ = 1.0
        ResidualErrorModel::Constant { a: 1.0 }
    }
}

impl ResidualErrorModel {
    /// Create a constant (additive) error model
    ///
    /// # Arguments
    /// * `a` - Standard deviation (must be positive)
    pub fn constant(a: f64) -> Self {
        ResidualErrorModel::Constant { a }
    }

    /// Create a proportional error model
    ///
    /// # Arguments
    /// * `b` - Proportional coefficient (e.g., 0.1 for 10% CV)
    pub fn proportional(b: f64) -> Self {
        ResidualErrorModel::Proportional { b }
    }

    /// Create a combined (additive + proportional) error model
    ///
    /// # Arguments
    /// * `a` - Additive component
    /// * `b` - Proportional component
    pub fn combined(a: f64, b: f64) -> Self {
        ResidualErrorModel::Combined { a, b }
    }

    /// Create a within-observation correlated additive/proportional model.
    ///
    /// Valid parametric declarations require finite `a, b > 0` and finite
    /// `rho` strictly inside `(-1, 1)`.
    pub fn correlated_combined(a: f64, b: f64, rho: f64) -> Self {
        ResidualErrorModel::CorrelatedCombined { a, b, rho }
    }

    /// Create an exponential error model
    ///
    /// # Arguments
    /// * `sigma` - Standard deviation on log scale
    pub fn exponential(sigma: f64) -> Self {
        ResidualErrorModel::Exponential { sigma }
    }

    /// Compute sigma (standard deviation) for a given prediction
    ///
    /// # Arguments
    /// * `prediction` - The model prediction (f)
    ///
    /// # Returns
    /// The standard deviation σ at this prediction value.
    /// Returns a cutoff minimum to avoid numerical issues with very small σ.
    pub fn sigma(&self, prediction: f64) -> f64 {
        let raw_sigma = match self {
            ResidualErrorModel::Constant { a } => *a,
            ResidualErrorModel::Proportional { b } => b * prediction.abs(),
            ResidualErrorModel::Combined { a, b } => {
                (a.powi(2) + b.powi(2) * prediction.powi(2)).sqrt()
            }
            ResidualErrorModel::CorrelatedCombined { a, b, rho } => {
                let proportional = b * prediction;
                (a + rho * proportional).hypot((1.0 - rho * rho).sqrt() * proportional)
            }
            ResidualErrorModel::Exponential { sigma } => *sigma,
        };

        // Apply a machine-precision cutoff to prevent division by zero.
        raw_sigma.max(f64::EPSILON.sqrt())
    }

    /// Simulate one observation from a supplied standard-normal draw.
    ///
    /// Exponential residual error is lognormal on the observation scale and
    /// therefore requires a finite, strictly positive prediction. The other
    /// models are additive normal errors with their prediction-dependent sigma.
    pub fn simulate_with_standard_normal(
        &self,
        prediction: f64,
        standard_normal: f64,
    ) -> Option<f64> {
        if !prediction.is_finite() || !standard_normal.is_finite() {
            return None;
        }

        let observation = match self {
            Self::Constant { .. }
            | Self::Proportional { .. }
            | Self::Combined { .. }
            | Self::CorrelatedCombined { .. } => {
                prediction + self.sigma(prediction) * standard_normal
            }
            Self::Exponential { sigma } => {
                if prediction <= 0.0 {
                    return None;
                }
                prediction * (sigma * standard_normal).exp()
            }
        };

        observation.is_finite().then_some(observation)
    }

    /// Compute the residual variance for a prediction.
    pub fn variance(&self, prediction: f64) -> f64 {
        self.sigma(prediction).powi(2)
    }

    /// Return the model's primary scale parameter.
    pub fn primary_parameter(&self) -> f64 {
        match self {
            Self::Constant { a } => *a,
            Self::Proportional { b } => *b,
            Self::Combined { a, .. } | Self::CorrelatedCombined { a, .. } => *a,
            Self::Exponential { sigma } => *sigma,
        }
    }

    /// Return whether this is a proportional model.
    pub fn is_proportional(&self) -> bool {
        matches!(self, Self::Proportional { .. })
    }

    /// Return whether this is a constant model.
    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Constant { .. })
    }

    /// Return whether this is a combined model.
    pub fn is_combined(&self) -> bool {
        matches!(self, Self::Combined { .. })
    }

    /// Return whether this is a correlated additive/proportional model.
    pub fn is_correlated_combined(&self) -> bool {
        matches!(self, Self::CorrelatedCombined { .. })
    }

    /// Return whether this is an exponential model.
    pub fn is_exponential(&self) -> bool {
        matches!(self, Self::Exponential { .. })
    }
}

/// Residual error models indexed by output equation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResidualErrorModels {
    models: Vec<Option<ResidualErrorModel>>,
}

impl ResidualErrorModels {
    /// Create an empty collection
    pub fn new() -> Self {
        Self { models: vec![] }
    }

    /// Add an error model for a specific output equation
    pub fn add(mut self, outeq: usize, model: ResidualErrorModel) -> Self {
        if outeq >= self.models.len() {
            self.models.resize(outeq + 1, None);
        }
        self.models[outeq] = Some(model);
        self
    }

    /// Get the error model for a specific output equation
    pub fn get(&self, outeq: usize) -> Option<&ResidualErrorModel> {
        self.models.get(outeq).and_then(Option::as_ref)
    }

    /// Get a mutable reference to the error model for a specific output equation
    pub fn get_mut(&mut self, outeq: usize) -> Option<&mut ResidualErrorModel> {
        self.models.get_mut(outeq).and_then(Option::as_mut)
    }

    /// Compute sigma for an output equation and prediction.
    pub fn sigma(&self, outeq: usize, prediction: f64) -> Option<f64> {
        self.get(outeq).map(|model| model.sigma(prediction))
    }

    /// Number of error models
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if collection has no declared models.
    pub fn is_empty(&self) -> bool {
        self.models.iter().all(Option::is_none)
    }

    /// Iterate over declared output indices and models.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &ResidualErrorModel)> {
        self.models
            .iter()
            .enumerate()
            .filter_map(|(index, model)| model.as_ref().map(|model| (index, model)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_error() {
        let model = ResidualErrorModel::constant(0.5);
        assert!((model.sigma(0.0) - 0.5).abs() < 1e-10);
        assert!((model.sigma(100.0) - 0.5).abs() < 1e-10);
        assert!((model.sigma(-50.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_proportional_error() {
        let model = ResidualErrorModel::proportional(0.1);
        assert!((model.sigma(100.0) - 10.0).abs() < 1e-10);
        assert!((model.sigma(50.0) - 5.0).abs() < 1e-10);
        // Uses absolute value, so negative predictions work
        assert!((model.sigma(-100.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_combined_error() {
        let model = ResidualErrorModel::combined(0.5, 0.1);
        // At f=0: sigma = sqrt(0.25 + 0) = 0.5
        assert!((model.sigma(0.0) - 0.5).abs() < 1e-10);
        // At f=100: sigma = sqrt(0.25 + 100) = sqrt(100.25)
        assert!((model.sigma(100.0) - 100.25_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn correlated_combined_matches_loading_formula_for_signed_predictions() {
        let model = ResidualErrorModel::correlated_combined(0.7, 0.2, -0.35);
        for prediction in [-3.0, 0.0, 2.5] {
            let direct = 0.7_f64.powi(2)
                + 2.0 * -0.35 * 0.7 * 0.2 * prediction
                + 0.2_f64.powi(2) * prediction.powi(2);
            let loading = (0.7 + -0.35 * 0.2 * prediction).powi(2)
                + (1.0 - (-0.35_f64).powi(2)) * (0.2 * prediction).powi(2);
            assert!((direct - loading).abs() < 1e-14);
            assert!((model.variance(prediction) - direct).abs() < 1e-14);

            let draw = model
                .simulate_with_standard_normal(prediction, -1.25)
                .unwrap();
            assert!(((draw - prediction).powi(2) / 1.25_f64.powi(2) - direct).abs() < 1e-13);
        }

        let ordinary = ResidualErrorModel::combined(0.7, 0.2);
        let independent = ResidualErrorModel::correlated_combined(0.7, 0.2, 0.0);
        for prediction in [-3.0, 0.0, 2.5] {
            assert_eq!(
                ordinary.variance(prediction),
                independent.variance(prediction)
            );
        }
    }

    #[test]
    fn test_sigma_cutoff() {
        let model = ResidualErrorModel::proportional(0.1);
        // At prediction = 0, raw sigma would be 0, but cutoff prevents this
        let sigma = model.sigma(0.0);
        assert!(sigma > 0.0);
        assert!(sigma >= f64::EPSILON.sqrt());
    }

    #[test]
    fn test_residual_error_models_collection() {
        let models = ResidualErrorModels::new()
            .add(0, ResidualErrorModel::constant(0.5))
            .add(1, ResidualErrorModel::proportional(0.1));

        assert_eq!(models.len(), 2);
        assert_eq!(models.get(0), Some(&ResidualErrorModel::constant(0.5)));
        assert_eq!(models.get(1), Some(&ResidualErrorModel::proportional(0.1)));
        assert!(models.get(2).is_none());
        assert!((models.get(0).unwrap().sigma(100.0) - 0.5).abs() < 1e-10);
        assert!((models.get(1).unwrap().sigma(100.0) - 10.0).abs() < 1e-10);

        let sparse = ResidualErrorModels::new().add(1, ResidualErrorModel::constant(0.25));
        assert_eq!(sparse.len(), 2);
        assert!(!sparse.is_empty());
        assert_eq!(sparse.get(0), None);
        assert_eq!(sparse.sigma(0, 1.0), None);
        assert_eq!(
            sparse.iter().collect::<Vec<_>>(),
            vec![(1, sparse.get(1).unwrap())]
        );
    }
}
