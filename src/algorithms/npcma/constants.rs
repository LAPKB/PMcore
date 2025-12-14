//! NPCMA Constants
//!
//! Configuration constants for the CMA-ES algorithm.

// ============================================================================
// CONVERGENCE CONSTANTS (matching NPAG/NPSAH)
// ============================================================================

/// Grid spacing convergence threshold
pub const THETA_E: f64 = 1e-4;

/// Objective function convergence threshold
pub const THETA_G: f64 = 1e-4;

/// P(Y|L) convergence criterion
pub const THETA_F: f64 = 1e-2;

/// Distance threshold for new support points
pub const THETA_D: f64 = 1e-4;

// ============================================================================
// CMA-ES PARAMETERS
// ============================================================================

/// Population size (lambda) - number of samples per generation
pub const POPULATION_SIZE: usize = 30;

/// Number of parents (mu) - typically lambda/2
pub const N_PARENTS: usize = 15;

/// Initial step size (sigma)
pub const INITIAL_SIGMA: f64 = 0.3;

/// Minimum step size before restart
pub const MIN_SIGMA: f64 = 1e-8;

/// Maximum step size
pub const MAX_SIGMA: f64 = 2.0;

/// Cumulation factor for step size control (c_sigma)
pub const C_SIGMA: f64 = 0.3;

/// Damping factor for step size (d_sigma)
pub const D_SIGMA: f64 = 1.0;

/// Cumulation factor for covariance matrix (c_c)
pub const C_C: f64 = 0.4;

/// Learning rate for rank-1 update (c_1)
pub const C_1: f64 = 0.2;

/// Learning rate for rank-mu update (c_mu)  
pub const C_MU: f64 = 0.3;

/// Maximum stagnation cycles before restart
pub const MAX_STAGNATION: usize = 15;

/// Eigenvalue floor for numerical stability
pub const EIGENVALUE_FLOOR: f64 = 1e-10;

/// Condition number threshold for restart
pub const CONDITION_THRESHOLD: f64 = 1e14;

// ============================================================================
// ALGORITHM PHASES
// ============================================================================

/// Number of warm-up cycles using NPAG-style grid expansion
pub const WARMUP_CYCLES: usize = 3;

/// Fraction of max D-criterion to use as threshold for adding points
pub const D_THRESHOLD_FRACTION: f64 = 0.5;

// ============================================================================
// GLOBAL OPTIMALITY CHECK
// ============================================================================

/// Number of samples for global optimality check
pub const GLOBAL_CHECK_SAMPLES: usize = 200;

/// D-criterion threshold for global optimality
pub const GLOBAL_D_THRESHOLD: f64 = 0.01;
