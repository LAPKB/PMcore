//! PMcore is a framework for developing and running non-parametric algorithms for population pharmacokinetic modelling
//!
//! The framework is designed to be modular and flexible, allowing for easy integration of new algorithms and methods. It is heavily designed around the specifications for Pmetrics, a package for R, and is designed to be used in conjunction with it. However, as a general rust library, it can be used for a wide variety of applications, not limited to pharmacometrics.
//!
//! # Configuration
//!
//! PMcore is configured using a TOML file, which specifies the settings for the algorithm. The settings file is divided into sections, each of which specifies a different aspect of the algorithm. They are further described in the [routines::settings] module.
//!
//! # Model definition
//!
//! As PMcore is provided as a library, the user must define the model to be used. Some algebraic models are provided, and more will be added, but the user is free to define their own model. The model must implement the [routines::simulation::predict] trait, which specifies the methods that the model must implement. For more information on how to define models with ordinary differential equations, please look at at the examples.
//!
//! # Data format
//!
//! Data is provided in a CSV file, and the format is described in the table below. For each subject, there must be at least one dose event.
//!
//! | Column | Description                                                         | Conditions                       |
//! |--------|---------------------------------------------------------------------|----------------------------------|
//! | `ID`   | Unique subject ID                                                   |                                  |
//! | `EVID` | Event type; 0 = observation, 1 = dose, 4 = reset                    |                                  |
//! | `TIME` | Time of event                                                       |                                  |
//! | `DUR`  | Duration of an infusion                                             | Must be provided if EVID = 1     |
//! | `DOSE` | The dose amount                                                     | Must be provided if EVID = 1     |
//! | `ADDL` | The number of additional doses to be given at the interval `II`     |                                  |
//! | `II`   | Interval for additional doses                                       |                                  |
//! | `INPUT`| The compartment the dose should be delivered to                     |                                  |
//! | `OUT`  | The observed value                                                  | Must be provided if EVID = 0     |
//! | `OUTEQ`| Denotes the output equation for which `OUT` is provided             |                                  |
//! | `C0`   | Optional override of the error polynomial coefficient               |                                  |
//! | `C1`   | Optional override of the error polynomial coefficient               |                                  |
//! | `C2`   | Optional override of the error polynomial coefficient               |                                  |
//! | `C3`   | Optional override of the error polynomial coefficient               |                                  |
//! | `COV...`| Any additional columns are assumed to be covariates, one per column | Must be present for the first event for each subject |
//!
//! # Examples
//!
//! A couple of examples are provided in the `examples` directory. The `settings.toml` file contains the settings for the algorithm, and the `data.csv` file contains the data.
//!
//! They can be run using the following command
//!
//! ```sh
//! cargo run --release --example `example_name`
//! ```
//!
//! where `example_name` is the name of the example to run. Currently, the following examples are available:
//!
//! - `bimodal_ke`: A simple, one-compartmental model following an intravenous infusion. The example is named by the bimodal distribution of one of two parameters, `Ke`, the elimination rate constant. The example is designed to demonstrate the ability of the algorithm to handle multimodal distributions, and detect outliers.
//! - `simple_covariates`: A simple example with a single subject and a single dose event, with covariates.

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

    pub use crate::routines::initialization::Prior;

    pub use crate::routines::settings::ErrorModel;
    pub use crate::routines::settings::*;
    pub use crate::structs::*;

    //Alma re-exports
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
