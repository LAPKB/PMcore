//! PMcore is a framework for developing and running non-parametric algorithms for population pharmacokinetic modelling
//!
//! The framework is designed to be modular and flexible, allowing for easy integration of new algorithms and methods. It is heavily designed around the specifications for Pmetrics, a package for R, and is designed to be used in conjunction with it. However, as a general rust library, it can be used for a wide variety of applications, not limited to pharmacometrics.
//!
//! # Configuration
//!
//! PMcore is configured using [routines::settings::Settings], which specifies the settings for the program.
//!
//! # Data format
//!
//! PMcore is heavily linked to [pharmsol], which provides the data structures and routines for handling pharmacokinetic data. The data is stored in a [pharmsol::Data] structure, and can either be read from a CSV file, using [pharmsol::data::parse_pmetrics::read_pmetrics], or created dynamically using the [pharmsol::data::builder::SubjectBuilder].
//!

/// Provides the various algorithms used within the framework
// pub mod algorithms;
pub mod algorithms;

/// Routines
pub mod routines;

// Structures
pub mod structs;

// Re-export commonly used items
pub use anyhow::Result;
pub use std::collections::HashMap;

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
