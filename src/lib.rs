//! PMcore is a framework for developing and running non-parametric algorithms for population pharmacokinetic modelling

/// Provides the various algorithms used within the framework
pub mod algorithms;

/// Routines for the crate
pub mod routines {
    /// Handles datafile operations
    pub mod datafile;
    /// Routines for initializing the grid
    pub mod initialization;
    pub mod optimization {
        pub mod d_optimizer;
        pub mod optim;
    }

    /// Routines for writing results to file, such as predicted values
    pub mod output;
    /// Routines for condensing grids
    pub mod condensation {
        pub mod prune;
    }
    /// Routines for expanding grids
    pub mod expansion {
        pub mod adaptative_grid;
    }

    /// Provides routines for reading and parsing settings
    pub mod settings;
    pub mod evaluation {

        /// Interior point method for solving the optimization problem
        pub mod ipm;
        pub mod ipm_faer;
        pub mod prob;
        pub mod qr;
        pub mod sigma;
    }
    pub mod simulation {
        pub mod predict;
    }
}

/// Entry points for external use of the framework.
pub mod entrypoints;
/// Logger functionality for the framework using [tracing]
pub mod logger;
/// Terminal-based user interface components.
pub mod tui;

/// A collection of commonly used items to simplify imports.
pub mod prelude {
    pub use crate::algorithms;
    pub use crate::entrypoints::simulate;
    pub use crate::entrypoints::start;
    pub use crate::entrypoints::start_internal;
    pub use crate::logger;
    pub use crate::prelude::evaluation::{prob, sigma, *};
    pub use crate::routines::condensation;
    pub use crate::routines::expansion::*;
    pub use crate::routines::initialization::*;
    pub use crate::routines::optimization;
    pub use crate::routines::simulation::*;
    pub use crate::routines::*;
    pub use crate::tui::ui::*;
}

//Tests
mod tests;
