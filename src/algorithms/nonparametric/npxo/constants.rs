//! Constants for NPXO algorithm

/// Weight threshold for condensation
pub const THETA_G: f64 = 1e-4;

/// Distance threshold for new points
pub const THETA_D: f64 = 1e-4;

/// Number of offspring to generate each cycle
pub const CROSSOVER_COUNT: usize = 30;

/// BLX-α extension parameter (0 = no extension, 0.5 = 50% extension)
pub const BLX_ALPHA: f64 = 0.25;

/// SBX distribution index (higher = closer to parents)
pub const SBX_ETA: f64 = 20.0;

/// Minimum cycles before convergence check
pub const MIN_CYCLES: usize = 5;

/// Objective function tolerance for convergence
pub const OBJF_TOLERANCE: f64 = 1e-3;

/// Number of stable cycles for convergence
pub const STABLE_CYCLES: usize = 3;

/// Boundary margin
pub const BOUNDARY_MARGIN: f64 = 0.001;

/// Mutation probability (small random perturbation)
pub const MUTATION_PROB: f64 = 0.1;

/// Mutation scale as fraction of range
pub const MUTATION_SCALE: f64 = 0.05;
