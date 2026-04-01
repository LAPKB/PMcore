//! Constants for NPPSO algorithm

// ============================================================================
// CONVERGENCE CONSTANTS (matching NPAG/NPSAH)
// ============================================================================

/// Grid spacing convergence threshold
pub const THETA_E: f64 = 1e-4;

/// Objective function convergence threshold
pub const THETA_G: f64 = 1e-4;

/// P(Y|L) convergence criterion
pub const THETA_F: f64 = 1e-2;

/// Distance threshold for new points
pub const THETA_D: f64 = 1e-4;

// ============================================================================
// PSO PARAMETERS
// ============================================================================

/// Number of particles in swarm
pub const SWARM_SIZE: usize = 40;

/// Inertia weight bounds (adaptive)
pub const INERTIA_MAX: f64 = 0.9;
pub const INERTIA_MIN: f64 = 0.4;

/// Cognitive weight (attraction to personal best)
pub const COGNITIVE_WEIGHT: f64 = 2.0;

/// Social weight (attraction to global best)
pub const SOCIAL_WEIGHT: f64 = 2.0;

/// Max velocity as fraction of range
pub const MAX_VELOCITY_FRACTION: f64 = 0.15;

/// Boundary margin (fraction of range)
pub const BOUNDARY_MARGIN: f64 = 0.001;

// ============================================================================
// ALGORITHM PHASES
// ============================================================================

/// Number of warm-up cycles using NPAG-style grid expansion
pub const WARMUP_CYCLES: usize = 3;

/// Fraction of max D-criterion to use as threshold for adding points
pub const D_THRESHOLD_FRACTION: f64 = 0.5;

/// Convergence threshold for swarm clustering
pub const CONVERGENCE_THRESHOLD: f64 = 0.8;

/// Fraction of particles to reinject when converging
pub const REINJECT_FRACTION: f64 = 0.25;

// ============================================================================
// GLOBAL OPTIMALITY CHECK
// ============================================================================

/// Number of random samples for global optimality check
pub const GLOBAL_CHECK_SAMPLES: usize = 500;

/// D-criterion threshold for global optimality (should be near 0 when optimal)
pub const GLOBAL_D_THRESHOLD: f64 = 0.01;

// ============================================================================
// SUBJECT MAP & D-OPTIMAL REFINEMENT
// ============================================================================

/// Number of worst-fit subjects to target with MAP injection
pub const RESIDUAL_SUBJECTS: usize = 2;

/// Max evaluations for subject MAP optimization (COBYLA)
pub const SUBJECT_MAP_EVALS: usize = 100;

/// Max evaluations for D-optimal refinement (COBYLA)
pub const DOPT_REFINE_EVALS: usize = 50;

/// Perform D-optimal refinement every N cycles
pub const DOPT_REFINE_INTERVAL: usize = 10;

/// Weight threshold for D-optimal refinement (only refine important points)
pub const DOPT_WEIGHT_THRESHOLD: f64 = 0.05;

// ============================================================================
// SIMULATED ANNEALING (for escaping local optima)
// ============================================================================

/// Initial SA temperature (high for exploration)
pub const SA_INITIAL_TEMP: f64 = 3.0;

/// Cooling rate per cycle
pub const SA_COOLING_RATE: f64 = 0.95;

/// Minimum temperature
pub const SA_MIN_TEMP: f64 = 0.05;

/// Number of SA injection attempts per cycle
pub const SA_INJECT_COUNT: usize = 40;

/// Reheat when temperature drops and objf stagnates
pub const SA_REHEAT_FACTOR: f64 = 2.0;

// ============================================================================
// ELITE PRESERVATION
// ============================================================================

/// Number of elite points to preserve
pub const ELITE_COUNT: usize = 5;

/// Maximum age (cycles) before elite point is removed
pub const ELITE_MAX_AGE: usize = 20;
