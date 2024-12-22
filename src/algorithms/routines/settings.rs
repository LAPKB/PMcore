#![allow(dead_code)]

use crate::algorithms::AlgorithmType;

use super::output::OutputFile;
use anyhow::{bail, Result};
use config::Config as eConfig;
use pharmsol::prelude::data::ErrorType;
use serde::Deserialize;
use serde_derive::Serialize;
use serde_json;

/// Contains all settings for PMcore
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Settings {
    /// General configuration settings
    pub config: Config,
    /// Parameters to be estimated
    pub parameters: Parameters,
    /// Defines the error model and polynomial to be used
    pub error: Error,
    /// Configuration for predictions
    pub predictions: Predictions,
    /// Configuration for logging
    pub log: Log,
    /// Configuration for (optional) prior
    pub prior: Prior,
    /// Configuration for the output files
    pub output: Output,
    /// Configuration for the convergence criteria
    pub convergence: Convergence,
    /// Advanced options, mostly hyperparameters, for the algorithm(s)
    pub advanced: Advanced,
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

    pub fn new() -> Self {
        Settings::default()
    }
}

/// General configuration settings
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// Maximum number of cycles to run
    pub cycles: usize,
    /// Denotes the algorithm to use
    pub algorithm: AlgorithmType,
    /// If true (default), cache predicted values
    pub cache: bool,
    /// Vector of IDs to include
    pub include: Option<Vec<String>>,
    /// Vector of IDs to exclude
    pub exclude: Option<Vec<String>>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            cycles: 100,
            algorithm: AlgorithmType::NPAG,
            cache: true,
            include: None,
            exclude: None,
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
    parameters: Vec<Parameter>,
}

impl Parameters {
    /// Create a new set of parameters
    pub fn new() -> Self {
        Parameters {
            parameters: Vec::new(),
        }
    }

    /// Add a parameter to the set
    pub fn add(
        mut self,
        name: impl Into<String>,
        lower: f64,
        upper: f64,
        fixed: bool,
    ) -> Result<Self> {
        let parameter = Parameter::new(name, lower, upper, fixed)?;
        self.parameters.push(parameter);
        Ok(self)
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
    pub fn ranges(&self) -> Vec<(f64, f64)> {
        self.parameters.iter().map(|p| (p.lower, p.upper)).collect()
    }
}

/// Defines the error model and polynomial to be used
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct Error {
    /// The initial value of `gamma` or `lambda`
    pub value: f64,
    /// The error class, either `additive` or `proportional`
    pub class: String,
    /// The assay error polynomial
    pub poly: (f64, f64, f64, f64),
}

impl Default for Error {
    fn default() -> Self {
        Error {
            value: 0.0,
            class: "additive".to_string(),
            poly: (0.0, 0.1, 0.0, 0.0),
        }
    }
}

impl Error {
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
        match self.class.to_lowercase().as_str() {
            "additive" | "l" | "lambda"  => ErrorType::Add,
            "proportional" | "g" | "gamma"  => ErrorType::Prop,
            _ => panic!("Error class '{}' not supported. Possible classes are 'gamma' (proportional) or 'lambda' (additive)", self.class),
        }
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
    pub level: String,
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
            level: String::from("info"),
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
        if self.path.is_empty() || self.path == "" {
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
        if let Err(error) = write_settings_to_file(&settings) {
            bail!("Could not write settings to file: {}", error);
        }
    }

    Ok(settings) // Return the settings wrapped in Ok
}

/// Writes a copy of the parsed settings to file
///
/// This function writes a copy of the parsed settings to file.
/// The file is written to output folder specified in the [settings](crate::routines::settings::Settings::paths), and is named `settings.json`.
pub fn write_settings_to_file(settings: &Settings) -> Result<()> {
    let serialized = serde_json::to_string_pretty(settings)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let outputfile = OutputFile::new(settings.output.path.as_str(), "settings.json")?;
    let mut file = outputfile.file;
    std::io::Write::write_all(&mut file, serialized.as_bytes())?;
    Ok(())
}
