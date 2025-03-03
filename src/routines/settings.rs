use super::output::OutputFile;
use crate::algorithms::Algorithm;
use anyhow::{bail, Result};
use config::Config as eConfig;
use pharmsol::prelude::data::ErrorType;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fmt::Display;

/// Contains all settings for PMcore
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Settings {
    /// General configuration settings
    config: Config,
    /// Parameters to be estimated
    parameters: Parameters,
    /// Defines the error model and polynomial to be used
    error: Error,
    /// Configuration for predictions
    predictions: Predictions,
    /// Configuration for logging
    log: Log,
    /// Configuration for (optional) prior
    prior: Prior,
    /// Configuration for the output files
    output: Output,
    /// Configuration for the convergence criteria
    convergence: Convergence,
    /// Advanced options, mostly hyperparameters, for the algorithm(s)
    advanced: Advanced,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            config: Config::default(),
            parameters: Parameters::new(),
            error: Error::default(),
            predictions: Predictions::default(),
            log: Log::default(),
            prior: Prior::default(),
            convergence: Convergence::default(),
            output: Output::default(),
            advanced: Advanced::default(),
        }
    }
}

impl Settings {
    /// Validate the settings
    pub fn validate(&self) -> Result<()> {
        self.error.validate()?;
        self.predictions.validate()?;
        Ok(())
    }

    /// Create a new settings object with default values
    pub fn new() -> Self {
        Settings::default()
    }

    pub fn set_config(&mut self, config: Config) {
        self.config = config;
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn set_parameters(&mut self, parameters: Parameters) {
        self.parameters = parameters;
    }

    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    pub fn set_error(&mut self, error: Error) {
        self.error = error;
    }

    pub fn error(&self) -> &Error {
        &self.error
    }

    pub fn set_predictions(&mut self, predictions: Predictions) {
        self.predictions = predictions;
    }

    pub fn predictions(&self) -> &Predictions {
        &self.predictions
    }

    pub fn set_log(&mut self, log: Log) {
        self.log = log;
    }

    pub fn log(&self) -> &Log {
        &self.log
    }

    pub fn set_prior(&mut self, prior: Prior) {
        self.prior = prior;
    }

    pub fn prior(&self) -> &Prior {
        &self.prior
    }

    pub fn set_output(&mut self, output: Output) {
        self.output = output;
    }

    pub fn output(&self) -> &Output {
        &self.output
    }

    pub fn set_convergence(&mut self, convergence: Convergence) {
        self.convergence = convergence;
    }

    pub fn convergence(&self) -> &Convergence {
        &self.convergence
    }

    pub fn set_advanced(&mut self, advanced: Advanced) {
        self.advanced = advanced;
    }

    pub fn advanced(&self) -> &Advanced {
        &self.advanced
    }

    pub fn set_cycles(&mut self, cycles: usize) {
        self.config.cycles = cycles;
    }

    pub fn set_algorithm(&mut self, algorithm: Algorithm) {
        self.config.algorithm = algorithm;
    }

    pub fn set_cache(&mut self, cache: bool) {
        self.config.cache = cache;
    }

    pub fn set_gamlam(&mut self, value: f64) {
        self.error.value = value;
    }

    pub fn set_error_type(&mut self, class: ErrorType) {
        self.error.class = class;
    }

    pub fn set_error_poly(&mut self, poly: (f64, f64, f64, f64)) {
        self.error.poly = poly;
    }

    pub fn set_error_value(&mut self, value: f64) {
        self.error.value = value;
    }

    pub fn set_idelta(&mut self, idelta: f64) {
        self.predictions.idelta = idelta;
    }

    pub fn set_tad(&mut self, tad: f64) {
        self.predictions.tad = tad;
    }

    pub fn set_log_level(&mut self, level: LogLevel) {
        self.log.level = level;
    }

    pub fn set_log_file(&mut self, file: String) {
        self.log.file = file;
    }

    pub fn set_prior_sampler(&mut self, sampler: String) {
        self.prior.sampler = sampler;
    }

    pub fn set_prior_points(&mut self, points: usize) {
        self.prior.points = points;
    }

    pub fn set_prior_seed(&mut self, seed: usize) {
        self.prior.seed = seed;
    }

    pub fn set_prior_file(&mut self, file: Option<String>) {
        self.prior.file = file;
    }

    pub fn set_output_write(&mut self, write: bool) {
        self.output.write = write;
    }

    pub fn set_output_path(&mut self, path: impl Into<String>) {
        self.output.path = path.into();
    }

    /// Writes a copy of the parsed settings to file
    /// The is written to output folder specified in the [Output] and is named `settings.json`.
    pub fn write(&self) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let outputfile = OutputFile::new(self.output.path.as_str(), "settings.json")?;
        let mut file = outputfile.file;
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
}

impl Default for Config {
    fn default() -> Self {
        Config {
            cycles: 100,
            algorithm: Algorithm::NPAG,
            cache: true,
        }
    }
}

/// Defines a parameter to be estimated
///
/// In non-parametric algorithms, parameters must be bounded. The lower and upper bounds are defined by the `lower` and `upper` fields, respectively.
/// Fixed parameters are unknown, but common among all subjects.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Parameter {
    name: String,
    lower: f64,
    upper: f64,
    fixed: bool,
}

impl Parameter {
    /// Create a new parameter
    pub fn new(name: impl Into<String>, lower: f64, upper: f64, fixed: bool) -> Result<Self> {
        if lower >= upper {
            bail!(format!(
                "In key '{}', lower bound ({}) is not less than upper bound ({})",
                name.into(),
                lower,
                upper
            ));
        }

        Ok(Self {
            name: name.into(),
            lower,
            upper,
            fixed,
        })
    }
}

/// This structure contains information on all [Parameter]s to be estimated
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Parameters {
    parameters: Vec<(String, Parameter)>,
}

impl Parameters {
    /// Create a new set of parameters
    pub fn new() -> Self {
        Parameters {
            parameters: Vec::new(),
        }
    }

    /// Create a new builder for parameters
    pub fn builder() -> ParametersBuilder {
        ParametersBuilder::new()
    }

    // Get a parameter by name
    pub fn get(&self, name: impl Into<String>) -> Option<&Parameter> {
        let name = name.into();
        self.parameters
            .iter()
            .find(|(n, _)| n == &name)
            .map(|(_, p)| p)
    }

    /// Get the names of the parameters
    pub fn names(&self) -> Vec<String> {
        self.parameters
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get the ranges of the parameters
    ///
    /// Returns a vector of tuples, where each tuple contains the lower and upper bounds of the parameter
    pub fn ranges(&self) -> Vec<(f64, f64)> {
        self.parameters
            .iter()
            .map(|(_, p)| (p.lower, p.upper))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (String, Parameter)> {
        self.parameters.iter()
    }
}

impl IntoIterator for Parameters {
    type Item = (String, Parameter);
    type IntoIter = std::vec::IntoIter<(String, Parameter)>;

    fn into_iter(self) -> Self::IntoIter {
        self.parameters.into_iter()
    }
}

/// Builder for creating a set of parameters
pub struct ParametersBuilder {
    parameters: Vec<(String, Parameter)>,
}

impl ParametersBuilder {
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
        }
    }

    pub fn add(mut self, name: impl Into<String>, lower: f64, upper: f64, fixed: bool) -> Self {
        let name_string = name.into();
        if let Ok(parameter) = Parameter::new(&name_string, lower, upper, fixed) {
            self.parameters.push((name_string, parameter));
        }
        self
    }

    pub fn build(self) -> Result<Parameters> {
        Ok(Parameters {
            parameters: self.parameters,
        })
    }
}

/// Defines the error model and polynomial to be used
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Error {
    /// The initial value of `gamma` or `lambda`
    pub value: f64,
    /// The error class, either `additive` or `proportional`
    #[serde(skip)]
    pub class: ErrorType,
    /// The assay error polynomial
    pub poly: (f64, f64, f64, f64),
}

impl Default for Error {
    fn default() -> Self {
        Error {
            value: 0.0,
            class: ErrorType::Add,
            poly: (0.0, 0.1, 0.0, 0.0),
        }
    }
}

impl Error {
    pub fn new(value: f64, class: ErrorType, poly: (f64, f64, f64, f64)) -> Self {
        Error { value, class, poly }
    }

    pub fn validate(&self) -> Result<()> {
        if self.value < 0.0 {
            bail!(format!(
                "Error value must be non-negative, got {}",
                self.value
            ));
        }
        Ok(())
    }

    pub fn error_type(&self) -> ErrorType {
        self.class.clone()
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
            min_distance: 0.12,
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
/// - `INFO`
/// - `WARN`
/// - `ERROR`
///
/// The default log level is `INFO`
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
    /// The maximum log level to display
    ///
    /// The log level is defined as a string, and can be one of the following:
    /// - `trace`
    /// - `debug`
    /// - `info`
    /// - `warn`
    /// - `error`
    pub level: LogLevel,
    /// The file to write the log to
    pub file: String,
    /// Whether to write logs
    ///
    /// If set to `false`, a global subscriber will not be set by PMcore.
    /// This can be useful when the user wants to use a custom subscriber for a third-party library, or perform benchmarks.
    pub write: bool,
}

impl Default for Log {
    fn default() -> Self {
        Log {
            level: LogLevel::INFO,
            file: String::from("log.txt"),
            write: true,
        }
    }
}

/// Configuration for the prior
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Prior {
    /// The sampler to use for the prior if not supplied
    pub sampler: String,
    /// The number of points to generate for the prior
    pub points: usize,
    /// The seed for the random number generator
    pub seed: usize,
    /// Optionally, the path to a file containing the prior in a CSV-format
    ///
    /// The file should contain the prior in a CSV format, with the first row containing the parameter names, and the subsequent rows containing the values for each parameter.
    /// The `prob` column is optional, and will if present be ignored
    pub file: Option<String>,
}

impl Default for Prior {
    fn default() -> Self {
        Prior {
            sampler: String::from("sobol"),
            points: 2048,
            seed: 22,
            file: None,
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
        Output {
            write: true,
            path: String::from("outputs/"),
        }
    }
}

impl Output {
    /// Parses the output folder location
    ////
    /// If a `#` symbol is found, it will automatically increment the number by one.
    pub fn parse_output_folder(&mut self) -> Result<()> {
        if self.path.is_empty() || self.path.is_empty() {
            // Set a default path if none is provided
            self.path = Output::default().path;
        }

        let folder = &self.path;

        // Check for the `#` symbol to replace with an incremented number
        let count = folder.matches('#').count();
        match count {
            0 => Ok(()),
            1 => {
                let mut folder = folder.clone();
                let mut num = 1;
                while std::path::Path::new(&folder.replace("#", &num.to_string())).exists() {
                    num += 1;
                }
                folder = folder.replace("#", &num.to_string());
                self.path = folder;
                Ok(())
            }
            _ => {
                bail!("Only one `#` symbol is allowed in the setting folder path. Rename the `output_folder` setting in the configuration file and re-run the program.")
            }
        }
    }
}

/// Parses the settings from a TOML configuration file
///
/// This function parses the settings from a TOML configuration file. The settings are validated, and a copy of the settings is written to file.
///
/// Entries in the TOML file may be overridden by environment variables. The environment variables must be prefixed with `PMCORE_`, and the TOML entry must be in uppercase. For example, the TUI may be disabled by setting the environment variable `PMCORE_CONFIG_TUI=false` A single underscore, `_`, is used as the separator for nested entries.
pub fn read(path: impl Into<String>) -> Result<Settings, anyhow::Error> {
    let settings_path = path.into();

    let parsed = eConfig::builder()
        .add_source(config::File::with_name(&settings_path).format(config::FileFormat::Toml))
        .add_source(config::Environment::with_prefix("PMCORE").separator("_"))
        .build()?;

    // Deserialize settings to the Settings struct
    let mut settings: Settings = parsed.try_deserialize()?;

    // Validate entries
    settings.validate()?;

    // Parse the output folder
    settings.output.parse_output_folder()?;

    // Write a copy of the settings to file if output is enabled
    if settings.output.write {
        if let Err(error) = settings.write() {
            bail!("Could not write settings to file: {}", error);
        }
    }

    Ok(settings) // Return the settings wrapped in Ok
}

pub struct SettingsBuilder<State> {
    config: Option<Config>,
    parameters: Option<Parameters>,
    error: Option<Error>,
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
            error: None,
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
            error: self.error,
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
            error: self.error,
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
    pub fn set_error_model(self, error: Error) -> SettingsBuilder<ErrorSet> {
        SettingsBuilder {
            config: self.config,
            parameters: self.parameters,
            error: Some(error),
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
    pub fn set_cycles(mut self, cycles: usize) -> Self {
        self.config.as_mut().unwrap().cycles = cycles;
        self
    }

    pub fn set_cache(mut self, cache: bool) -> Self {
        self.config.as_mut().unwrap().cache = cache;
        self
    }

    pub fn set_predictions(mut self, predictions: Predictions) -> Self {
        self.predictions = Some(predictions);
        self
    }

    pub fn set_log(mut self, log: Log) -> Self {
        self.log = Some(log);
        self
    }

    pub fn set_prior(mut self, prior: Prior) -> Self {
        self.prior = Some(prior);
        self
    }

    pub fn set_output(mut self, output: Output) -> Self {
        self.output = Some(output);
        self
    }

    pub fn set_convergence(mut self, convergence: Convergence) -> Self {
        self.convergence = Some(convergence);
        self
    }

    pub fn set_advanced(mut self, advanced: Advanced) -> Self {
        self.advanced = Some(advanced);
        self
    }

    pub fn build(self) -> Settings {
        Settings {
            config: self.config.unwrap(),
            parameters: self.parameters.unwrap(),
            error: self.error.unwrap(),
            predictions: self.predictions.unwrap_or_default(),
            log: self.log.unwrap_or_default(),
            prior: self.prior.unwrap_or_default(),
            output: self.output.unwrap_or_default(),
            convergence: self.convergence.unwrap_or_default(),
            advanced: self.advanced.unwrap_or_default(),
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    use crate::algorithms::Algorithm;
    use pharmsol::prelude::data::ErrorType;

    #[test]
    fn test_builder() {
        let parameters = Parameters::builder()
            .add("Ke", 0.0, 5.0, false)
            .add("V", 10.0, 200.0, true)
            .build()
            .unwrap();

        let settings = SettingsBuilder::new()
            .set_algorithm(Algorithm::NPAG) // Step 1: Define algorithm
            .set_parameters(parameters) // Step 2: Define parameters
            .set_error_model(Error {
                value: 0.1,
                class: ErrorType::Add,
                poly: (0.0, 0.1, 0.0, 0.0),
            }) // Step 3: Define error model
            .set_cycles(100) // Optional: Set cycles
            .set_cache(true) // Optional: Set cache
            .build(); // Final step

        assert_eq!(settings.config.algorithm, Algorithm::NPAG);
        assert_eq!(settings.config.cycles, 100);
        assert_eq!(settings.config.cache, true);
        assert_eq!(settings.parameters().names(), vec!["Ke", "V"]);
    }
}
