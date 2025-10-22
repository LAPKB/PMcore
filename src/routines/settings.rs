use crate::algorithms::Algorithm;
use crate::routines::initialization::Prior;
use crate::routines::output::OutputFile;
use anyhow::{bail, Result};
use pharmsol::prelude::data::ErrorModels;

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
    /// Defines the error models and polynomials to be used
    pub(crate) errormodels: ErrorModels,
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

    pub fn errormodels(&self) -> &ErrorModels {
        &self.errormodels
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
}

impl Default for Advanced {
    fn default() -> Self {
        Advanced {
            min_distance: 1e-4,
            nm_steps: 100,
            tolerance: 1e-6,
        }
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
    errormodels: Option<ErrorModels>,
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
pub trait ErrorModelDefined {}

// Implement marker traits for PhantomData states
pub struct InitialState;
pub struct AlgorithmSet;
pub struct ParametersSet;
pub struct ErrorSet;

// Initial state: no algorithm set yet
impl SettingsBuilder<InitialState> {
    pub fn new() -> Self {
        SettingsBuilder {
            config: None,
            parameters: None,
            errormodels: None,
            predictions: None,
            log: None,
            prior: None,
            output: None,
            convergence: None,
            advanced: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn set_algorithm(self, algorithm: Algorithm) -> SettingsBuilder<AlgorithmSet> {
        SettingsBuilder {
            config: Some(Config {
                algorithm,
                ..Config::default()
            }),
            parameters: self.parameters,
            errormodels: self.errormodels,
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
impl SettingsBuilder<AlgorithmSet> {
    pub fn set_parameters(self, parameters: Parameters) -> SettingsBuilder<ParametersSet> {
        SettingsBuilder {
            config: self.config,
            parameters: Some(parameters),
            errormodels: self.errormodels,
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

// Parameters are set, move to defining error model
impl SettingsBuilder<ParametersSet> {
    pub fn set_error_models(self, ems: ErrorModels) -> SettingsBuilder<ErrorSet> {
        SettingsBuilder {
            config: self.config,
            parameters: self.parameters,
            errormodels: Some(ems),
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

// Error model is set, allow optional settings and final build
impl SettingsBuilder<ErrorSet> {
    pub fn build(self) -> Settings {
        Settings {
            config: self.config.unwrap(),
            parameters: self.parameters.unwrap(),
            errormodels: self.errormodels.unwrap(),
            predictions: self.predictions.unwrap_or_default(),
            log: self.log.unwrap_or_default(),
            prior: self.prior.unwrap_or_default(),
            output: self.output.unwrap_or_default(),
            convergence: self.convergence.unwrap_or_default(),
            advanced: self.advanced.unwrap_or_default(),
        }
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
    use pharmsol::{ErrorModel, ErrorPoly};

    use super::*;
    use crate::algorithms::Algorithm;

    #[test]
    fn test_builder() {
        let parameters = Parameters::new().add("Ke", 0.0, 5.0).add("V", 10.0, 200.0);

        let ems = ErrorModels::new()
            .add(
                0,
                ErrorModel::Proportional {
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
}
