//! Constants for NPOPT algorithm

// ============================================================================
// CONVERGENCE THRESHOLDS
// ============================================================================

/// Objective function convergence threshold
pub const THETA_G: f64 = 1e-4;
/// P(Y|L) convergence criterion
pub const THETA_F: f64 = 1e-2;
/// Minimum distance between support points
pub const THETA_D: f64 = 1e-4;
/// Weight stability threshold
pub const THETA_W: f64 = 1e-3;
/// Global optimality D-criterion threshold
pub const GLOBAL_D_THRESHOLD: f64 = 0.008;

// ============================================================================
// GRID EXPANSION
// ============================================================================

/// Initial grid spacing
pub const INITIAL_EPS: f64 = 0.2;
/// Minimum grid spacing
pub const MIN_EPS: f64 = 1e-4;

// ============================================================================
// PHASE CONTROL
// ============================================================================

/// Number of exploration cycles
pub const EXPLORATION_CYCLES: usize = 3;
/// Number of Sobol samples for initial coverage
pub const SOBOL_INIT_SAMPLES: usize = 50;
/// Cycles between global checks
pub const GLOBAL_CHECK_INTERVAL: usize = 3;
/// Number of Sobol samples for global check
pub const SOBOL_GLOBAL_SAMPLES: usize = 256;
/// Consecutive passes needed for convergence
pub const CONVERGENCE_PASSES: usize = 2;
/// Convergence window for objf stability
pub const CONVERGENCE_WINDOW: usize = 3;

// ============================================================================
// ADAPTIVE SIMULATED ANNEALING
// ============================================================================

/// Initial SA temperature
pub const INITIAL_TEMPERATURE: f64 = 2.0;
/// Base cooling rate
pub const BASE_COOLING_RATE: f64 = 0.90;
/// Minimum temperature
pub const MIN_TEMPERATURE: f64 = 0.01;
/// Target acceptance ratio
pub const TARGET_ACCEPTANCE: f64 = 0.23;
/// Trigger reheat when acceptance below this
pub const REHEAT_TRIGGER: f64 = 0.08;
/// Reheat factor
pub const REHEAT_FACTOR: f64 = 1.5;
/// Number of SA points to inject per cycle
pub const SA_INJECT_COUNT: usize = 30;
/// History window for acceptance ratio
pub const SA_HISTORY_WINDOW: usize = 5;

// ============================================================================
// FISHER-GUIDED EXPANSION
// ============================================================================

/// Fraction of candidates from Fisher directions
pub const FISHER_RATIO: f64 = 0.70;
/// Fraction of candidates from D-optimal gradient
pub const DOPT_RATIO: f64 = 0.30;
/// Number of Fisher-guided candidates
pub const FISHER_CANDIDATES: usize = 20;

// ============================================================================
// D-OPTIMAL REFINEMENT
// ============================================================================

/// High weight threshold (fraction of max)
pub const HIGH_WEIGHT_THRESHOLD: f64 = 0.10;
/// Medium weight threshold
pub const MED_WEIGHT_THRESHOLD: f64 = 0.01;
/// Low weight threshold (skip below this)
pub const LOW_WEIGHT_THRESHOLD: f64 = 0.001;
/// Max iterations for high-weight points
pub const DOPT_HIGH_ITERS: u64 = 80;
/// Max iterations for medium-weight points
pub const DOPT_MED_ITERS: u64 = 30;
/// Max iterations for low-weight points
pub const DOPT_LOW_ITERS: u64 = 10;

// ============================================================================
// SUBJECT RESIDUAL INJECTION
// ============================================================================

/// Number of worst-fit subjects to process
pub const RESIDUAL_SUBJECTS: usize = 3;
/// Max iterations for subject MAP
pub const SUBJECT_MAP_ITERS: u64 = 30;

// ============================================================================
// ELITE PRESERVATION
// ============================================================================

/// Number of elite points to preserve
pub const ELITE_COUNT: usize = 5;
/// Max age of elite point (cycles)
pub const ELITE_MAX_AGE: usize = 15;

// ============================================================================
// CONDENSATION
// ============================================================================

/// Lambda filter divisor (keep if > max_lambda / divisor)
pub const LAMBDA_FILTER_DIVISOR: f64 = 10000.0;

// ============================================================================
// BOUNDARY MARGIN
// ============================================================================

/// Margin from boundaries (fraction of range)
pub const BOUNDARY_MARGIN: f64 = 0.005;
