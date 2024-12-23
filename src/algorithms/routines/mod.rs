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
    pub mod adaptive_grid;
}

/// Provides routines for reading and parsing settings
pub mod settings;
pub mod evaluation {

    /// Interior point method for solving the optimization problem
    pub mod ipm;
    pub mod ipm_faer;
    pub mod qr;
}
