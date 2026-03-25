use crate::algorithms::Algorithm;
use crate::routines::initialization::Prior;
use crate::routines::output::OutputFile;
use anyhow::{bail, Result};
use pharmsol::prelude::data::{AssayErrorModels, ResidualErrorModels};

use serde::{Deserialize, Serialize};
use serde_json;
use std::fmt::Display;
use std::path::PathBuf;

/// Contains all settings for PMcore
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// General configuration settings
    pub(crate) config: Config,
    /// Parameters to be estimated
    pub(crate) parameters: Parameters,
    /// Error models for non-parametric algorithms (assay/measurement error)
    ///
    /// These use observation-based sigma calculation.
    /// Required for: NPAG, NPOD, NPSAH, NPBO, etc.
    pub(crate) errormodels: AssayErrorModels,
    /// Residual error models for parametric algorithms
    ///
    /// These use prediction-based sigma calculation (matching R saemix).
    /// Required for: SAEM, FOCE, etc.
    /// Parameters (a, b) are estimated during the algorithm.
    #[serde(default)]
    pub(crate) residual_error: Option<ResidualErrorModels>,
    /// Configuration for predictions
    pub(crate) predictions: Predictions,
    /// Configuration for logging
    pub(crate) log: Log,
    /// Configuration for (optional) prior
    pub(crate) prior: Prior,
    /// Configuration for the output files
    pub(crate) output: Output,
    /// Configuration for the convergence criteria
    pub(crate) convergence: Convergence,
    /// Advanced options, mostly hyperparameters, for the algorithm(s)
    pub(crate) advanced: Advanced,
}

impl Settings {
    /// Create a new [SettingsBuilder]
    pub fn builder() -> SettingsBuilder<InitialState> {
        SettingsBuilder::new()
    }

    /* Getters */
    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    pub fn errormodels(&self) -> &AssayErrorModels {
        &self.errormodels
    }

    /// Get the residual error models for parametric algorithms
    ///
    /// Returns `None` for non-parametric algorithms.
    pub fn residual_error(&self) -> Option<&ResidualErrorModels> {
        self.residual_error.as_ref()
    }

    pub fn predictions(&self) -> &Predictions {
        &self.predictions
    }

    pub fn log(&self) -> &Log {
        &self.log
    }

    pub fn prior(&self) -> &Prior {
        &self.prior
    }

    pub fn output(&self) -> &Output {
        &self.output
    }
    pub fn convergence(&self) -> &Convergence {
        &self.convergence
    }

    pub fn advanced(&self) -> &Advanced {
        &self.advanced
    }

    /* Setters */
    pub fn set_cycles(&mut self, cycles: usize) {
        self.config.cycles = cycles;
    }

    pub fn set_algorithm(&mut self, algorithm: Algorithm) {
        self.config.algorithm = algorithm;
    }

    pub fn set_cache(&mut self, cache: bool) {
        self.config.cache = cache;
    }

    pub fn set_idelta(&mut self, idelta: f64) {
        self.predictions.idelta = idelta;
    }

    pub fn set_tad(&mut self, tad: f64) {
        self.predictions.tad = tad;
    }

    pub fn set_prior(&mut self, prior: Prior) {
        self.prior = prior;
    }

    pub fn disable_output(&mut self) {
        self.output.write = false;
    }

    pub fn set_output_path(&mut self, path: impl Into<String>) {
        self.output.path = parse_output_folder(path.into());
    }

    pub fn set_log_stdout(&mut self, stdout: bool) {
        self.log.stdout = stdout;
    }

    pub fn set_write_logs(&mut self, write: bool) {
        self.log.write = write;
    }

    pub fn set_log_level(&mut self, level: LogLevel) {
        self.log.level = level;
    }

    pub fn set_progress(&mut self, progress: bool) {
        self.config.progress = progress;
    }

    pub fn initialize_logs(&mut self) -> Result<()> {
        crate::routines::logger::setup_log(self)
    }

    /// Writes a copy of the settings to file
    /// The is written to output folder specified in the [Output] and is named `settings.json`.
    pub fn write(&self) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;

        let outputfile = OutputFile::new(self.output.path.as_str(), "settings.json")?;
        let mut file = outputfile.file_owned();
        std::io::Write::write_all(&mut file, serialized.as_bytes())?;
        Ok(())
    }
}

/// General configuration settings
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// Maximum number of cycles to run
    pub cycles: usize,
    /// Denotes the algorithm to use
    pub algorithm: Algorithm,
    /// If true (default), cache predicted values
    pub cache: bool,
    /// Should a progress bar be displayed for the first cycle
    ///
    /// The progress bar is not written to logs, but is written to stdout. It incurs a minor performance penalty.
    pub progress: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            cycles: 100,
            algorithm: Algorithm::NPAG,
            cache: true,
            progress: true,
        }
    }
}

/// Defines a parameter to be estimated
///
/// In non-parametric algorithms, parameters must be bounded. The lower and upper bounds are defined by the `lower` and `upper` fields, respectively.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Parameter {
    pub(crate) name: String,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}

impl Parameter {
    /// Create a new parameter
    pub fn new(name: impl Into<String>, lower: f64, upper: f64) -> Self {
        Self {
            name: name.into(),
            lower,
            upper,
        }
    }
}

/// This structure contains information on all [Parameter]s to be estimated
#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct Parameters {
    pub(crate) parameters: Vec<Parameter>,
}

impl Parameters {
    pub fn new() -> Self {
        Parameters {
            parameters: Vec::new(),
        }
    }

    pub fn add(mut self, name: impl Into<String>, lower: f64, upper: f64) -> Parameters {
        let parameter = Parameter::new(name, lower, upper);
        self.parameters.push(parameter);
        self
    }

    // Get a parameter by name
    pub fn get(&self, name: impl Into<String>) -> Option<&Parameter> {
        let name = name.into();
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Get the names of the parameters
    pub fn names(&self) -> Vec<String> {
        self.parameters.iter().map(|p| p.name.clone()).collect()
    }
    /// Get the ranges of the parameters
    ///
    /// Returns a vector of tuples, where each tuple contains the lower and upper bounds of the parameter
    pub fn ranges(&self) -> Vec<(f64, f64)> {
        self.parameters.iter().map(|p| (p.lower, p.upper)).collect()
    }

    /// Get the number of parameters
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if the parameters are empty
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Iterate over the parameters
    pub fn iter(&self) -> std::slice::Iter<'_, Parameter> {
        self.parameters.iter()
    }
}

impl IntoIterator for Parameters {
    type Item = Parameter;
    type IntoIter = std::vec::IntoIter<Parameter>;

    fn into_iter(self) -> Self::IntoIter {
        self.parameters.into_iter()
    }
}

impl From<Vec<Parameter>> for Parameters {
    fn from(parameters: Vec<Parameter>) -> Self {
        Parameters { parameters }
    }
}

/// This struct contains advanced options and hyperparameters
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Advanced {
    /// The minimum distance required between a candidate point and the existing grid (THETA_D)
    ///
    /// This is general for all non-parametric algorithms
    pub min_distance: f64,
    /// Maximum number of steps in Nelder-Mead optimization
    /// This is used in the [NPOD](crate::algorithms::npod) algorithm, specifically in the [D-optimizer](crate::routines::optimization::d_optimizer)
    pub nm_steps: usize,
    /// Tolerance (in standard deviations) for the Nelder-Mead optimization
    ///
    /// This is used in the [NPOD](crate::algorithms::npod) algorithm, specifically in the [D-optimizer](crate::routines::optimization::d_optimizer)
    pub tolerance: f64,
    /// Configuration specific to SAEM algorithm
    pub saem: SaemSettings,
}

impl Default for Advanced {
    fn default() -> Self {
        Advanced {
            min_distance: 1e-4,
            nm_steps: 100,
            tolerance: 1e-6,
            saem: SaemSettings::default(),
        }
    }
}

impl Advanced {
    /// Get SAEM-specific settings
    pub fn saem(&self) -> &SaemSettings {
        &self.saem
    }
}

/// SAEM algorithm-specific configuration
///
/// These settings control the behavior of the Stochastic Approximation
/// Expectation-Maximization (SAEM) algorithm.
///
/// # Algorithm Phases
///
/// SAEM proceeds in phases:
/// 1. **Burn-in** (`burn_in` iterations): MCMC chains warm up, no statistics collected
/// 2. **Exploration/SA** (`k1_iterations`): Step size γₖ = 1, simulated annealing active
/// 3. **Smoothing** (`k2_iterations`): Step size γₖ = 1/(k-K₁+1), averaging for convergence
///
/// # R saemix Correspondence
///
/// | R saemix parameter | This field |
/// |-------------------|------------|
/// | `nbiter.saemix[1]` | `k1_iterations` |
/// | `nbiter.saemix[2]` | `k2_iterations` |
/// | `nbiter.burn` | `burn_in` |
/// | `nbiter.sa` | `sa_iterations` |
/// | `alpha.sa` | `sa_cooling_factor` |
/// | `stepsize.rw` | `mcmc_step_size` |
/// | `nb.chains` | `n_chains` |
/// | `transform.par` | `transform_par` |
/// | `map` | `compute_map` |
/// | `fim` | `compute_fim` |
/// | `ll.is` | `compute_ll_is` |
/// | `ll.gq` | `compute_ll_gq` |
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct SaemSettings {
    /// Number of exploration phase iterations (K₁)
    ///
    /// During this phase, γₖ = 1 (full update) and simulated annealing may be active.
    /// R saemix default: 300
    pub k1_iterations: usize,

    /// Number of smoothing phase iterations (K₂)
    ///
    /// During this phase, γₖ = 1/(k-K₁+1), providing averaging for convergence.
    /// R saemix default: 100
    pub k2_iterations: usize,

    /// Number of burn-in iterations before collecting sufficient statistics
    ///
    /// MCMC chains need to reach their stationary distribution before
    /// statistics are meaningful.
    /// R saemix default: 5
    pub burn_in: usize,

    /// Number of simulated annealing iterations at the start
    ///
    /// During SA, the likelihood is raised to a power T < 1 (temperature),
    /// allowing broader exploration. T = α^k where k is iteration.
    /// R saemix default: 0 (disabled)
    pub sa_iterations: usize,

    /// Simulated annealing cooling factor (α)
    ///
    /// Temperature T = α^k. Values close to 1 give slower cooling.
    /// R saemix default: 0.97
    pub sa_cooling_factor: f64,

    /// MCMC step size multiplier (τ) for random walk kernels
    ///
    /// Proposal variance is τ² × Ω for random walk updates.
    /// Larger values → bigger jumps, potentially lower acceptance.
    /// R saemix default: 0.4
    pub mcmc_step_size: f64,

    /// Initial variance scaling for random walk proposals
    ///
    /// R saemix default: 0.5 (rw.init)
    pub rw_init: f64,

    /// Number of MCMC chains per subject
    ///
    /// Multiple chains can improve mixing but increase computation.
    /// R saemix default: 1
    pub n_chains: usize,

    /// Number of MCMC iterations within each E-step
    ///
    /// More iterations → better approximation of E[h(φ)|y] but slower.
    /// R saemix default: 1-3
    pub mcmc_iterations: usize,

    /// Minimum variance for Ω diagonal elements
    ///
    /// Prevents Ω from collapsing to zero during optimization.
    /// This is especially important early in the algorithm.
    pub omega_min_variance: f64,

    /// Whether to use the Metropolis-within-Gibbs sampler
    ///
    /// If true, updates parameters one at a time (Gibbs-style).
    /// If false, updates all parameters jointly.
    pub use_gibbs: bool,

    /// Number of kernels to use in the MCMC sampler
    ///
    /// The SAEM implementation supports 4 MCMC kernels:
    /// 1. Individual random walk on η
    /// 2. Block random walk using Cholesky of Ω  
    /// 3. Joint random walk on all subjects
    /// 4. MAP-based proposal (mode-jumping)
    ///
    /// Values 1-4 select which kernels to use. Default uses all 4.
    pub n_kernels: usize,

    /// Parameter transformation codes matching R saemix transform.par
    ///
    /// For each parameter: 0=Normal, 1=LogNormal, 2=Probit (0-1), 3=Logit (0-1)
    /// If empty or shorter than n_params, defaults to LogNormal (1) for all.
    ///
    /// Example: vec![1, 1, 0] means param1=LogNormal, param2=LogNormal, param3=Normal
    pub transform_par: Vec<u8>,

    // === Post-hoc computation options (R saemix: map, fim, ll.is, ll.gq) ===
    /// Compute MAP (Maximum A Posteriori) individual parameter estimates
    ///
    /// R saemix default: true
    pub compute_map: bool,

    /// Compute Fisher Information Matrix and standard errors
    ///
    /// R saemix default: true
    pub compute_fim: bool,

    /// Compute log-likelihood by importance sampling
    ///
    /// R saemix default: true
    pub compute_ll_is: bool,

    /// Compute log-likelihood by Gaussian quadrature
    ///
    /// R saemix default: false
    pub compute_ll_gq: bool,

    /// Number of samples for importance sampling
    ///
    /// R saemix default: 5000 (nmc.is)
    pub n_mc_is: usize,

    /// Degrees of freedom for Student's t distribution in importance sampling
    ///
    /// R saemix default: 4 (nu.is)
    pub nu_is: usize,

    /// Number of nodes for Gaussian quadrature
    ///
    /// R saemix default: 12 (nnodes.gq)
    pub n_nodes_gq: usize,

    /// Number of SDs to span for Gaussian quadrature
    ///
    /// R saemix default: 4 (nsd.gq)
    pub n_sd_gq: f64,

    /// Display progress every N iterations
    ///
    /// R saemix default: 100 (nbdisplay)
    pub display_progress: usize,

    /// Seed for random number generator
    ///
    /// R saemix default: 123456 (seed)
    pub seed: u64,

    /// Fix the random seed (reproducibility)
    ///
    /// R saemix default: true (fix.seed)
    pub fix_seed: bool,
}

impl Default for SaemSettings {
    fn default() -> Self {
        Self {
            k1_iterations: 300,
            k2_iterations: 100,
            burn_in: 5,
            sa_iterations: 0,
            sa_cooling_factor: 0.97,
            mcmc_step_size: 0.4,
            rw_init: 0.5,
            n_chains: 1,
            mcmc_iterations: 1,
            omega_min_variance: 1e-6,
            use_gibbs: false,
            n_kernels: 4,
            // Default to LogNormal (1) for all params - will be extended if needed
            transform_par: vec![],
            // Post-hoc computation (matching R saemix defaults)
            compute_map: true,
            compute_fim: true,
            compute_ll_is: true,
            compute_ll_gq: false,
            n_mc_is: 5000,
            nu_is: 4,
            n_nodes_gq: 12,
            n_sd_gq: 4.0,
            display_progress: 10,
            seed: 123456,
            fix_seed: true,
        }
    }
}

impl SaemSettings {
    /// Get the total number of SAEM iterations (K₁ + K₂)
    pub fn total_iterations(&self) -> usize {
        self.k1_iterations + self.k2_iterations
    }

    /// Check if we're in the exploration phase (k ≤ K₁)
    pub fn is_exploration_phase(&self, iteration: usize) -> bool {
        iteration <= self.k1_iterations
    }

    /// Check if we're in the smoothing phase (k > K₁)
    pub fn is_smoothing_phase(&self, iteration: usize) -> bool {
        iteration > self.k1_iterations
    }

    /// Check if simulated annealing is active at this iteration
    pub fn is_sa_active(&self, iteration: usize) -> bool {
        self.sa_iterations > 0 && iteration <= self.sa_iterations
    }

    /// Get the simulated annealing temperature at this iteration
    ///
    /// Returns 1.0 if SA is not active.
    pub fn sa_temperature(&self, iteration: usize) -> f64 {
        if self.is_sa_active(iteration) {
            self.sa_cooling_factor.powi(iteration as i32)
        } else {
            1.0
        }
    }

    /// Get the step size γₖ at iteration k
    ///
    /// - During exploration (k ≤ K₁): γₖ = 1
    /// - During smoothing (k > K₁): γₖ = 1/(k - K₁ + 1)
    pub fn step_size(&self, iteration: usize) -> f64 {
        if iteration <= self.k1_iterations {
            1.0
        } else {
            let k_smooth = iteration - self.k1_iterations;
            1.0 / (k_smooth as f64 + 1.0)
        }
    }

    /// Get the transform code for a parameter index
    ///
    /// Returns the transform code from `transform_par` if specified,
    /// otherwise defaults to LogNormal (1) for positive-bounded parameters.
    ///
    /// Codes: 0=Normal, 1=LogNormal, 2=Probit, 3=Logit
    pub fn get_transform(&self, param_idx: usize) -> u8 {
        self.transform_par.get(param_idx).copied().unwrap_or(1) // Default: LogNormal
    }

    /// Get transform codes for all parameters, extending with defaults if needed
    ///
    /// Returns a vector of transform codes, one per parameter.
    /// If `transform_par` is shorter than `n_params`, fills with LogNormal (1).
    pub fn get_transforms(&self, n_params: usize) -> Vec<u8> {
        let mut transforms = self.transform_par.clone();
        while transforms.len() < n_params {
            transforms.push(1); // Default: LogNormal
        }
        transforms.truncate(n_params);
        transforms
    }

    /// Set transform codes from parameter ranges
    ///
    /// Automatically determines appropriate transforms based on bounds:
    /// - Both bounds finite and positive → LogNormal (1)
    /// - Lower = 0, Upper = 1 → Logit (3)
    /// - Lower = -∞, Upper = +∞ → Normal (0)
    pub fn infer_transforms_from_ranges(&mut self, ranges: &[(f64, f64)]) {
        self.transform_par = ranges
            .iter()
            .map(|(lower, upper)| {
                if *lower >= 0.0 && *upper > 0.0 && lower.is_finite() && upper.is_finite() {
                    1 // LogNormal for positive parameters
                } else if (*lower - 0.0).abs() < 1e-10 && (*upper - 1.0).abs() < 1e-10 {
                    3 // Logit for 0-1 bounded
                } else {
                    0 // Normal for unbounded
                }
            })
            .collect();
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
/// This struct contains the convergence criteria for the algorithm
pub struct Convergence {
    /// The objective function convergence criterion for the algorithm
    ///
    /// The objective function is the negative log likelihood
    /// Previously referred to as THETA_G
    pub likelihood: f64,
    /// The PYL convergence criterion for the algorithm
    ///
    /// P(Y|L) represents the probability of the observation given its weighted support
    /// Previously referred to as THETA_F
    pub pyl: f64,
    /// Precision convergence criterion for the algorithm
    ///
    /// The precision variable, sometimes referred to as `eps`, is the distance from existing points in the grid to the candidate point. A candidate point is suggested at a distance of `eps` times the range of the parameter.
    /// For example, if the parameter `alpha` has a range of `[0.0, 1.0]`, and `eps` is `0.1`, then the candidate point will be at a distance of `0.1 * (1.0 - 0.0) = 0.1` from the existing grid point(s).
    /// Previously referred to as THETA_E
    pub eps: f64,
}

impl Default for Convergence {
    fn default() -> Self {
        Convergence {
            likelihood: 1e-4,
            pyl: 1e-2,
            eps: 1e-2,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Predictions {
    /// The interval for which predictions are generated
    pub idelta: f64,
    /// The time after the last dose for which predictions are generated
    ///
    /// Predictions will always be generated until the last event (observation or dose) in the data.
    /// This setting is used to generate predictions beyond the last event if the `tad` if sufficiently large.
    /// This can be useful for generating predictions for a subject who only received a dose, but has no observations.
    pub tad: f64,
}

impl Default for Predictions {
    fn default() -> Self {
        Predictions {
            idelta: 0.12,
            tad: 0.0,
        }
    }
}

impl Predictions {
    /// Validate the prediction settings
    pub fn validate(&self) -> Result<()> {
        if self.idelta < 0.0 {
            bail!("The interval for predictions must be non-negative");
        }
        if self.tad < 0.0 {
            bail!("The time after dose for predictions must be non-negative");
        }
        Ok(())
    }
}

/// The log level, which can be one of the following:
/// - `TRACE`
/// - `DEBUG`
/// - `INFO` (Default)
/// - `WARN`
/// - `ERROR`
#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub enum LogLevel {
    TRACE,
    DEBUG,
    #[default]
    INFO,
    WARN,
    ERROR,
}

impl From<LogLevel> for tracing::Level {
    fn from(log_level: LogLevel) -> tracing::Level {
        match log_level {
            LogLevel::TRACE => tracing::Level::TRACE,
            LogLevel::DEBUG => tracing::Level::DEBUG,
            LogLevel::INFO => tracing::Level::INFO,
            LogLevel::WARN => tracing::Level::WARN,
            LogLevel::ERROR => tracing::Level::ERROR,
        }
    }
}

impl AsRef<str> for LogLevel {
    fn as_ref(&self) -> &str {
        match self {
            LogLevel::TRACE => "trace",
            LogLevel::DEBUG => "debug",
            LogLevel::INFO => "info",
            LogLevel::WARN => "warn",
            LogLevel::ERROR => "error",
        }
    }
}

impl Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Log {
    /// The maximum log level to display, as defined by [LogLevel]
    ///
    /// [LogLevel] is a thin wrapper around `tracing::Level`, but can be serialized
    pub level: LogLevel,
    /// Should the logs be written to a file
    ///
    /// If true, a file will be created in the output folder with the name `log.txt`, or, if [Output::write] is false, in the current directory.
    pub write: bool,
    /// Define if logs should be written to stdout
    pub stdout: bool,
}

impl Default for Log {
    fn default() -> Self {
        Log {
            level: LogLevel::INFO,
            write: false,
            stdout: true,
        }
    }
}

/// Configuration for the output files
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Output {
    /// Whether to write the output files
    pub write: bool,
    /// The (relative) path to write the output files to
    pub path: String,
}

impl Default for Output {
    fn default() -> Self {
        let path = PathBuf::from("outputs/").to_string_lossy().to_string();

        Output { write: true, path }
    }
}

pub struct SettingsBuilder<State> {
    config: Option<Config>,
    parameters: Option<Parameters>,
    errormodels: Option<AssayErrorModels>,
    residual_error: Option<ResidualErrorModels>,
    predictions: Option<Predictions>,
    log: Option<Log>,
    prior: Option<Prior>,
    output: Option<Output>,
    convergence: Option<Convergence>,
    advanced: Option<Advanced>,
    _marker: std::marker::PhantomData<State>,
}

// Marker traits for builder states
pub trait AlgorithmDefined {}
pub trait ParametersDefined {}
pub trait AssayErrorModelDefined {}

// Implement marker traits for PhantomData states
pub struct InitialState;
pub struct AlgorithmSet;
pub struct ParametersSet;
pub struct ErrorSet;

// New states for algorithm-specific paths
pub struct NonParametricParametersSet;
pub struct NonParametricErrorSet;
// ParametricErrorSet is defined below after its impl block

// Initial state: no algorithm set yet
impl SettingsBuilder<InitialState> {
    pub fn new() -> Self {
        SettingsBuilder {
            config: None,
            parameters: None,
            errormodels: None,
            residual_error: None,
            predictions: None,
            log: None,
            prior: None,
            output: None,
            convergence: None,
            advanced: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the algorithm - this determines which error model type is required
    ///
    /// For non-parametric algorithms (NPAG, NPOD, etc.), use `set_error_models()`
    /// For parametric algorithms (SAEM, FOCE, etc.), use `set_residual_error()`
    pub fn set_algorithm(self, algorithm: Algorithm) -> SettingsBuilder<AlgorithmSet> {
        SettingsBuilder {
            config: Some(Config {
                algorithm,
                ..Config::default()
            }),
            parameters: self.parameters,
            errormodels: self.errormodels,
            residual_error: self.residual_error,
            predictions: self.predictions,
            log: self.log,
            prior: self.prior,
            output: self.output,
            convergence: self.convergence,
            advanced: self.advanced,
            _marker: std::marker::PhantomData,
        }
    }
}

impl Default for SettingsBuilder<InitialState> {
    fn default() -> Self {
        SettingsBuilder::new()
    }
}

// Algorithm is set, move to defining parameters
// We keep AlgorithmSet generic to maintain backward compatibility
impl SettingsBuilder<AlgorithmSet> {
    /// Set parameters and transition to algorithm-specific state
    ///
    /// The builder will detect whether the algorithm is parametric or non-parametric
    /// and guide you to the appropriate error model setter.
    pub fn set_parameters(self, parameters: Parameters) -> SettingsBuilder<ParametersSet> {
        SettingsBuilder {
            config: self.config,
            parameters: Some(parameters),
            errormodels: self.errormodels,
            residual_error: self.residual_error,
            predictions: self.predictions,
            log: self.log,
            prior: self.prior,
            output: self.output,
            convergence: self.convergence,
            advanced: self.advanced,
            _marker: std::marker::PhantomData,
        }
    }
}

// Parameters are set - this state allows setting either error model type
// for backward compatibility, but documents the preferred approach
impl SettingsBuilder<ParametersSet> {
    /// Set error models for **non-parametric** algorithms (NPAG, NPOD, etc.)
    ///
    /// These error models use observation-based sigma calculation and represent
    /// assay/measurement error.
    ///
    /// # Example
    /// ```ignore
    /// let settings = Settings::builder()
    ///     .set_algorithm(Algorithm::NPAG)
    ///     .set_parameters(params)
    ///     .set_error_models(error_models)  // For non-parametric
    ///     .build();
    /// ```
    pub fn set_error_models(self, ems: AssayErrorModels) -> SettingsBuilder<ErrorSet> {
        SettingsBuilder {
            config: self.config,
            parameters: self.parameters,
            errormodels: Some(ems),
            residual_error: self.residual_error,
            predictions: self.predictions,
            log: self.log,
            prior: self.prior,
            output: self.output,
            convergence: self.convergence,
            advanced: self.advanced,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set residual error models for **parametric** algorithms (SAEM, FOCE, etc.)
    ///
    /// These error models use prediction-based sigma calculation (matching R saemix)
    /// and represent residual unexplained variability.
    ///
    /// Note: For backward compatibility, you must also call `set_error_models()`
    /// to provide the observation-based error model (used for fallback/diagnostics).
    ///
    /// # Example
    /// ```ignore
    /// let settings = Settings::builder()
    ///     .set_algorithm(Algorithm::SAEM)
    ///     .set_parameters(params)
    ///     .set_residual_error(residual_error)  // For parametric
    ///     .build();  // No AssayErrorModels needed for parametric!
    /// ```
    pub fn set_residual_error(
        self,
        rem: ResidualErrorModels,
    ) -> SettingsBuilder<ParametricErrorSet> {
        SettingsBuilder {
            config: self.config,
            parameters: self.parameters,
            errormodels: self.errormodels,
            residual_error: Some(rem),
            predictions: self.predictions,
            log: self.log,
            prior: self.prior,
            output: self.output,
            convergence: self.convergence,
            advanced: self.advanced,
            _marker: std::marker::PhantomData,
        }
    }
}

// Parametric path: residual error is set, can build directly or add optional error models
impl SettingsBuilder<ParametricErrorSet> {
    /// Build the settings (parametric algorithms don't require AssayErrorModels)
    pub fn build(self) -> Settings {
        Settings {
            config: self.config.unwrap(),
            parameters: self.parameters.unwrap(),
            // Use empty AssayErrorModels for parametric algorithms
            errormodels: self.errormodels.unwrap_or_default(),
            residual_error: self.residual_error,
            predictions: self.predictions.unwrap_or_default(),
            log: self.log.unwrap_or_default(),
            prior: self.prior.unwrap_or_default(),
            output: self.output.unwrap_or_default(),
            convergence: self.convergence.unwrap_or_default(),
            advanced: self.advanced.unwrap_or_default(),
        }
    }

    /// Optionally set observation-based error models for diagnostics
    ///
    /// For parametric algorithms, observation-based error models are optional
    /// but can be useful for certain diagnostic calculations.
    pub fn set_error_models(self, ems: AssayErrorModels) -> SettingsBuilder<ErrorSet> {
        SettingsBuilder {
            config: self.config,
            parameters: self.parameters,
            errormodels: Some(ems),
            residual_error: self.residual_error,
            predictions: self.predictions,
            log: self.log,
            prior: self.prior,
            output: self.output,
            convergence: self.convergence,
            advanced: self.advanced,
            _marker: std::marker::PhantomData,
        }
    }
}

// Parametric path marker type
pub struct ParametricErrorSet;

// Error model is set, allow optional settings and final build
impl SettingsBuilder<ErrorSet> {
    pub fn build(self) -> Settings {
        Settings {
            config: self.config.unwrap(),
            parameters: self.parameters.unwrap(),
            errormodels: self.errormodels.unwrap(),
            residual_error: self.residual_error,
            predictions: self.predictions.unwrap_or_default(),
            log: self.log.unwrap_or_default(),
            prior: self.prior.unwrap_or_default(),
            output: self.output.unwrap_or_default(),
            convergence: self.convergence.unwrap_or_default(),
            advanced: self.advanced.unwrap_or_default(),
        }
    }

    /// Set optional residual error for parametric algorithms
    ///
    /// This allows adding residual error after setting error_models,
    /// for cases where the order doesn't matter.
    pub fn with_residual_error(mut self, rem: ResidualErrorModels) -> Self {
        self.residual_error = Some(rem);
        self
    }
}

fn parse_output_folder(path: String) -> String {
    // If the path doesn't contain a "#", just return it as is
    if !path.contains("#") {
        return path;
    }

    // If it does contain "#", perform the incrementation logic
    let mut num = 1;
    while std::path::Path::new(&path.replace("#", &num.to_string())).exists() {
        num += 1;
    }

    path.replace("#", &num.to_string())
}

#[cfg(test)]

mod tests {
    use pharmsol::{AssayErrorModel, ErrorPoly, ResidualErrorModels};

    use super::*;
    use crate::algorithms::Algorithm;

    #[test]
    fn test_builder() {
        let parameters = Parameters::new().add("Ke", 0.0, 5.0).add("V", 10.0, 200.0);

        let ems = AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::Proportional {
                    gamma: pharmsol::Factor::Variable(5.0),
                    poly: ErrorPoly::new(0.0, 0.1, 0.0, 0.0),
                },
            )
            .unwrap();
        let mut settings = SettingsBuilder::new()
            .set_algorithm(Algorithm::NPAG) // Step 1: Define algorithm
            .set_parameters(parameters) // Step 2: Define parameters
            .set_error_models(ems)
            .build();

        settings.set_cycles(100);

        assert_eq!(settings.config.algorithm, Algorithm::NPAG);
        assert_eq!(settings.config.cycles, 100);
        assert_eq!(settings.config.cache, true);
        assert_eq!(settings.parameters().names(), vec!["Ke", "V"]);
    }

    #[test]
    fn test_builder_parametric_with_residual_error() {
        // Test the parametric algorithm path with explicit ResidualErrorModels
        let parameters = Parameters::new().add("Ke", 0.0, 5.0).add("V", 10.0, 200.0);

        // Create residual error models for parametric algorithms (prediction-based sigma)
        let residual_error =
            ResidualErrorModels::new().add(0, pharmsol::ResidualErrorModel::combined(0.5, 0.1));

        // For SAEM, we still need AssayErrorModels for the likelihood computation
        let ems = AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::Proportional {
                    gamma: pharmsol::Factor::Variable(0.1),
                    poly: ErrorPoly::new(1.0, 0.0, 0.0, 0.0),
                },
            )
            .unwrap();

        let settings = SettingsBuilder::new()
            .set_algorithm(Algorithm::SAEM)
            .set_parameters(parameters)
            .set_residual_error(residual_error.clone())
            .set_error_models(ems)
            .build();

        assert_eq!(settings.config.algorithm, Algorithm::SAEM);

        // Verify residual error is stored and accessible
        let stored_residual = settings.residual_error().unwrap();
        assert_eq!(stored_residual.len(), 1);

        // Verify the combined error model calculates sigma from prediction
        let sigma = stored_residual.sigma(0, 100.0).unwrap();
        // sqrt(0.5² + 0.1² * 100²) = sqrt(0.25 + 100) = sqrt(100.25) ≈ 10.01
        assert!((sigma - 100.25_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_builder_saem_without_explicit_residual_error() {
        // Verify backward compatibility: SAEM works without explicit residual error
        // (it will convert from AssayErrorModels internally)
        let parameters = Parameters::new().add("Ke", 0.0, 5.0).add("V", 10.0, 200.0);

        let ems = AssayErrorModels::new()
            .add(
                0,
                AssayErrorModel::Proportional {
                    gamma: pharmsol::Factor::Variable(0.1),
                    poly: ErrorPoly::new(1.0, 0.0, 0.0, 0.0),
                },
            )
            .unwrap();

        let settings = SettingsBuilder::new()
            .set_algorithm(Algorithm::SAEM)
            .set_parameters(parameters)
            .set_error_models(ems)
            .build();

        assert_eq!(settings.config.algorithm, Algorithm::SAEM);
        // No explicit residual error set - SAEM will convert internally
        assert!(settings.residual_error().is_none());
    }
}
