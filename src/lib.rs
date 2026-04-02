//! PMcore is a framework for developing and running population pharmacokinetic algorithms
//!
//! The framework exposes a unified modeling platform for nonparametric PMcore algorithms.
//!
//! # Algorithm Types
//!
//! ## Non-Parametric Algorithms
//! Represent the population distribution as a discrete set of support points with associated weights.
//! - NPAG (Non-Parametric Adaptive Grid)
//! - NPOD (Non-Parametric Optimal Design)
//! - And others...
//!
//! # Public API
//!
//! PMcore centers on the model/problem API in [api]. Models are defined with
//! [api::ModelDefinition], configured with [api::EstimationProblem], and executed through
//! [api::fit].
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

/// New public modeling and execution API.
pub mod api;

/// Shared preprocessing and compilation layer.
pub mod compile;

/// Estimation family boundaries for the new architecture.
pub mod estimation;

/// Public model-domain types used by the new API.
pub mod model;

/// Shared result and summary types for the new API.
pub mod results;

/// Shared output writers for the new API.
pub mod output;

// Re-export commonly used items
pub use anyhow::Result;
pub use std::collections::HashMap;

// BestDose
pub mod bestdose;

/// A collection of commonly used items to simplify imports.
pub mod prelude {
    pub use super::HashMap;
    pub use super::Result;
    pub use crate::algorithms;
    pub use crate::algorithms::Algorithm;
    pub use crate::api::fit;
    pub use crate::api::{
        AlgorithmTuning, ConvergenceOptions, EstimationMethod, EstimationProblem, LoggingLevel,
        LoggingOptions, ModelDefinition, NexusOptions, NonparametricMethod, NpagOptions,
        NpboOptions, NpcatOptions, NpcmaOptions, NpodOptions, NpoptOptions, NppsoOptions,
        Npsah2Options, NpsahOptions, NpxoOptions, OutputPlan, PostProbOptions,
        RuntimeOptions,
    };
    pub use crate::compile::{CompiledProblem, DesignContext, ObservationIndex};
    pub use crate::estimation::nonparametric::{
        CycleLog, NPCycle, NPPredictions, NonparametricEngine, NonparametricWorkspace, Posterior,
        Psi, Theta, Weights,
    };
    pub use crate::model::{
        ContinuousObservationSpec, CovariateEffectsSpec, CovariateModel, CovariateSpec,
        ModelMetadata, ObservationChannel, ObservationLikelihood, ObservationSpec, ParameterDomain,
        ParameterSpace, ParameterSpec, ParameterTransform as ModelParameterTransform,
        ParameterVariability, RandomEffectsSpec, VariabilityModel,
    };
    pub use crate::results::{
        ArtifactIndex, DiagnosticsBundle, FitResult, FitSummary, IndividualSummary,
        ParameterSummary, PopulationSummary, PredictionsBundle,
    };
    pub use pharmsol::optimize::effect::get_e2;

    pub use pharmsol;

    pub use crate::estimation::nonparametric::{read_prior, Prior};

    pub mod simulator {
        pub use pharmsol::prelude::simulator::*;
    }
    pub mod data {
        pub use pharmsol::prelude::data::*;
    }
    pub mod models {
        pub use pharmsol::prelude::models::*;
    }

    //traits
    pub use pharmsol::data::*;
    pub use pharmsol::equation::Equation;
    pub use pharmsol::equation::EquationTypes;
    pub use pharmsol::equation::Predictions;
    pub use pharmsol::equation::*;
    pub use pharmsol::prelude::*;
    pub use pharmsol::simulator::*;
    pub use pharmsol::ODE;
    pub use pharmsol::SDE;

    //macros
    pub use pharmsol::fa;
    pub use pharmsol::fetch_cov;
    pub use pharmsol::fetch_params;
    pub use pharmsol::lag;
}
