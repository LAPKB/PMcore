//! PMcore is a framework for developing and running population pharmacokinetic algorithms.
//!
//! The structure branch keeps the refactored platform surface together with the baseline
//! non-parametric workflows that existed on `main`.
//!
//! # Algorithm Types
//!
//! ## Non-Parametric Algorithms
//! Represent the population distribution as a discrete set of support points with associated weights.
//! - NPAG (Non-Parametric Adaptive Grid)
//! - NPOD (Non-Parametric Optimal Design)
//! - POSTPROB (Posterior probability reweighting)
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
//! from a CSV file, using [pharmsol::data::parse_pmetrics::read_pmetrics], or created dynamically
//! using the [pharmsol::data::builder::SubjectBuilder].
//!

/// Provides the various algorithms used within the framework
pub mod algorithms;

/// Estimation family boundaries for the new architecture.
pub mod estimation;

/// Public model-domain types used by the new API.
pub mod model;

/// Shared result and summary types for the new API.
pub mod results;

/// Logs
pub mod logs;

// Re-export commonly used items
pub use anyhow::Result;
pub use std::collections::HashMap;

// BestDose
pub mod bestdose;

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
        ErrorModels, EstimationProblem, FitProgress, NonParametricAlgorithm,
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
