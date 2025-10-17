//! Core data types for the BestDose algorithm

use crate::prelude::*;
use crate::routines::output::predictions::NPPredictions;
use crate::routines::settings::Settings;
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;
use pharmsol::prelude::*;

/// Target type for dose optimization
///
/// Determines whether targets in the "future" file represent concentrations or AUCs.
/// This matches Fortran's ITARGET parameter (1=concentration, 2=AUC).
#[derive(Debug, Clone, Copy)]
pub enum Target {
    /// Target concentrations at observation times (ITARGET=1)
    Concentration,
    /// Target cumulative AUC values from time 0 (ITARGET=2)
    /// AUC is calculated using trapezoidal rule with dense time grid
    AUC,
}

/// Allowable dose range constraints
#[derive(Debug, Clone)]
pub struct DoseRange {
    pub(crate) min: f64,
    pub(crate) max: f64,
}

impl DoseRange {
    pub fn new(min: f64, max: f64) -> Self {
        DoseRange { min, max }
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Default for DoseRange {
    fn default() -> Self {
        DoseRange {
            min: 0.0,
            max: f64::MAX,
        }
    }
}

/// The BestDose optimization problem
///
/// Contains all data needed for the three-stage BestDose algorithm:
/// 1. Posterior density calculation (NPAGFULL11 + NPAGFULL)
/// 2. Dual optimization (posterior weights vs uniform weights)
/// 3. Predictions and cost evaluation
#[derive(Debug, Clone)]
pub struct BestDoseProblem {
    // Input data
    pub past_data: Subject,
    pub target: Subject,
    pub target_type: Target,

    // Population prior
    pub prior_theta: Theta,
    pub prior_weights: Weights,

    // Patient-specific posterior (from NPAGFULL11 + NPAGFULL)
    pub theta: Theta,
    pub posterior: Weights,

    // Model and settings
    pub eq: ODE,
    pub error_models: ErrorModels,
    pub settings: Settings,

    // Optimization parameters
    pub doserange: DoseRange,
    pub bias_weight: f64, // λ: 0=personalized, 1=population
}

/// Result from BestDose optimization
#[derive(Debug)]
pub struct BestDoseResult {
    /// Optimal dose amount(s)
    pub dose: Vec<f64>,

    /// Final cost function value
    pub objf: f64,

    /// Optimization status
    pub status: String,

    /// Concentration-time predictions for optimal doses
    pub preds: NPPredictions,

    /// AUC values at observation times (only populated when target_type is AUC)
    pub auc_predictions: Option<Vec<(f64, f64)>>, // (time, auc) pairs

    /// Which optimization method produced the best result: "posterior" or "uniform"
    /// Matches Fortran's dual-optimization approach (BESTDOS113+)
    pub optimization_method: String,
}
