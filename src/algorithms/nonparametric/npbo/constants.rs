//! NPBO Constants
//!
//! Configuration constants for the Bayesian Optimization algorithm.

#![allow(dead_code)] // Many constants reserved for future tuning/experimentation

/// Number of warmup cycles using grid expansion before BO
pub const WARMUP_CYCLES: usize = 3;

/// Number of initial points for GP training (Sobol sampling)
pub const INITIAL_SAMPLES: usize = 50;

/// Maximum GP training points to prevent O(n³) scaling issues
pub const MAX_GP_POINTS: usize = 500;

/// Minimum GP training points before optimization
pub const MIN_GP_POINTS: usize = 20;

/// Number of acquisition optimization restarts
pub const ACQUISITION_RESTARTS: usize = 10;

/// Batch size for parallel acquisition
pub const BATCH_SIZE: usize = 5;

/// GP kernel length scale initial value
pub const INITIAL_LENGTH_SCALE: f64 = 0.3;

/// GP kernel signal variance initial value  
pub const INITIAL_SIGNAL_VAR: f64 = 1.0;

/// GP noise variance (jitter for numerical stability)
pub const NOISE_VAR: f64 = 1e-6;

/// Exploration-exploitation tradeoff (higher = more exploration)
pub const EXPLORATION_WEIGHT: f64 = 2.0;

/// Convergence threshold for EI improvement
pub const EI_CONVERGENCE_THRESHOLD: f64 = 1e-6;

/// Maximum cycles without improvement before termination
pub const MAX_STAGNATION_CYCLES: usize = 10;

/// Weight threshold for condensation
pub const WEIGHT_THRESHOLD: f64 = 1e-8;

/// D-optimal refinement iterations per cycle
pub const DOPT_ITERATIONS: usize = 3;

/// Adaptive length scale bounds
pub const LENGTH_SCALE_MIN: f64 = 0.01;
pub const LENGTH_SCALE_MAX: f64 = 1.0;

/// Whether to use ARD (Automatic Relevance Determination)
pub const USE_ARD: bool = true;
