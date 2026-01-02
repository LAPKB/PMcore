//! PMcore is a framework for developing and running population pharmacokinetic algorithms
//!
//! The framework supports both **non-parametric** and **parametric** approaches to population modeling,
//! allowing for flexible estimation of population parameter distributions.
//!
//! # Algorithm Types
//!
//! ## Non-Parametric Algorithms
//! Represent the population distribution as a discrete set of support points with associated weights.
//! - NPAG (Non-Parametric Adaptive Grid)
//! - NPOD (Non-Parametric Optimal Design)
//! - And others...
//!
//! ## Parametric Algorithms
//! Represent the population distribution as a continuous distribution (typically multivariate normal).
//! - SAEM (Stochastic Approximation Expectation-Maximization)
//! - FOCE/FOCEI (First-Order Conditional Estimation)
//! - IT2B (Iterative Two-Stage Bayesian)
//! - And others...
//!
//! # Configuration
//!
//! PMcore is configured using [routines::settings::Settings], which specifies the settings for the program.
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

/// Routines for data processing, optimization, and output
pub mod routines;

/// Data structures for population modeling
pub mod structs;

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
    pub use crate::algorithms::dispatch_algorithm;
    pub use crate::algorithms::Algorithm;
    pub use crate::routines;
    pub use crate::routines::logger;
    pub use pharmsol::optimize::effect::get_e2;

    pub use pharmsol;

    pub use crate::routines::initialization::Prior;

    pub use crate::routines::settings::*;
    pub use crate::structs::*;

    // Non-parametric specific (explicit imports for clarity)
    pub use crate::structs::nonparametric::{Psi, Theta, Weights};

    // Parametric specific
    pub use crate::structs::parametric::{
        CovarianceStructure, CovariateModel, Individual, IndividualEstimates, ParameterTransform,
        Population, SufficientStats,
    };

    // Output types
    pub use crate::routines::output::{NPResult, ParametricIterationLog, ParametricResult};

    // Sampling utilities (for custom parametric algorithms)
    pub use crate::routines::sampling::{
        MetropolisHastings, GaussianProposal, ParameterTransforms, ProposalDistribution,
    };

    // Settings
    pub use crate::routines::settings::SaemSettings;

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
