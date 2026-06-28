//! Methods for generating the initial set of support points ([`Theta`]).

pub mod latin;
pub mod sobol;

/// Default seed used for quasi-random sampling when none is given.
pub const DEFAULT_SEED: usize = 22;

/// Default number of support points sampled for a quasi-random starting grid.
pub const DEFAULT_POINTS: usize = 2028;
