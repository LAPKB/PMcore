//! PMcore is a framework for developing and running population pharmacokinetic algorithms.
//!
//! # Algorithm Types
//!
//! ## Non-Parametric Algorithms
//! Represent the population distribution as a discrete set of support points with associated weights.
//! - NPAG (Non-Parametric Adaptive Grid)
//! - NPOD (Non-Parametric Optimal Design)
//! - NPMAP (Maximum a posteriori reweighting)
//!
//! ## Parametric Algorithms (planned)
//! Represent the population distribution with a parametric form (e.g. a normal distribution) and
//! estimate the parameters of that distribution. This family is not yet implemented; the API is
//! present but calling it will panic until a solver (SAEM) is available.
//!
//! # Public Interface
//!
//! PMcore centers on the estimation interface in [estimation]. Models are defined in
//! [model], configured with [estimation::EstimationProblem], and then executed with the
//! selected algorithm.
//!
//! # Data format
//!
//! PMcore is heavily linked to [pharmsol], which provides the data structures and routines for handling
//! pharmacokinetic data. The data is stored in a [pharmsol::Data] structure, and can either be read
//! from a CSV file, using `pharmsol::data::parse_pmetrics::read_pmetrics`, or created dynamically
//! using the [pharmsol::data::builder::SubjectBuilder].
//!

/// Provides the various algorithms used within the framework
pub mod algorithms;

/// Estimation problems and the non-parametric and parametric fitting families.
pub mod estimation;

/// Model-domain types: parameters, parameter spaces, and metadata.
pub mod model;

/// Result and summary types shared across algorithms.
pub mod results;

/// Logging utilities.
pub mod logs;

// Re-export commonly used items
pub use anyhow::Result;
pub use std::collections::HashMap;

/// Dose optimization and forecasting (BestDose).
pub mod bestdose;

/// SDE-based Inter-Occasion Variability optimization.
pub mod iov;

/// A collection of commonly used items to simplify imports.
pub mod prelude {
    pub use super::logs::Logger;
    pub use super::HashMap;
    pub use super::Result;
    pub use crate::algorithms;
    pub use crate::algorithms::Algorithm;

    pub use crate::estimation::NonParametric;
    pub use crate::estimation::Parametric;
    pub use crate::estimation::{
        ErrorModels, EstimationProblem, FitProgress, NcnpagConfig, NonParametricAlgorithm,
        NonparametricCycleProgress, NpagConfig, NpmapConfig, NpodConfig, ParametricAlgorithm,
        SaemConfig,
    };

    pub use crate::model::parameter_space::{
        BoundedParameter, Parameter, ParameterScale, ParameterSpace, UnboundedParameter,
    };

    pub use crate::algorithms::nonparametric::{CycleFlow, FitController, FitObserver};
    pub use crate::estimation::nonparametric::{
        CycleLog, NPCycle, NPPredictions, NonParametricResult, Posterior, Psi, Theta, Weights,
    };
    pub use crate::iov::{DiffusionConfig, DiffusionOptimize, DiffusionResult};
    pub use crate::model::{EquationMetadataSource, ModelMetadata};
    pub use crate::results::{
        FitResult, FitSummary, IndividualSummary, ParameterSummary, PopulationSummary,
    };

    // pharmsol: re-export the crate itself and its curated prelude.
    pub use pharmsol;
    pub use pharmsol::prelude::*;

    // Items required by downstream code that are not part of `pharmsol::prelude`.
    pub use pharmsol::equation::{EquationTypes, Predictions};
    pub use pharmsol::optimize::effect::get_e2;
    pub use pharmsol::{ODE, SDE};

    // Organized submodules mirroring pharmsol's grouping.
    pub mod simulator {
        pub use pharmsol::prelude::simulator::*;
    }
    pub mod data {
        pub use pharmsol::prelude::data::*;
    }
    pub mod models {
        pub use pharmsol::prelude::models::*;
    }

    // macros
    pub use pharmsol::{fa, fetch_cov, fetch_params, lag};
}
