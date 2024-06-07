#![allow(dead_code)]

use config::Config as eConfig;
use pharmsol::prelude::data::ErrorType;
use serde::Deserialize;
use serde_derive::Serialize;
use serde_json;
use std::collections::HashMap;
use toml::{map::Map, Table, Value};

/// Contains all settings for PMcore
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Paths to the data, log and prior files
    pub paths: Paths,
    /// General configuration settings
    pub config: Config,
    /// Random parameters to be estimated
    pub random: Random,
    /// Parameters which are estimated, but fixed for the population
    pub fixed: Option<Fixed>,
    /// Parameters which are held constant
    pub constant: Option<Constant>,
    /// Defines the error model and polynomial to be used
    pub error: Error,
}

/// This struct contains the paths to the data, log and prior files.
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Paths {
    /// Path to the data file, see `datafile::parse` for details.
    pub data: String,
    /// If provided, the log file will be written to this path.
    pub log: Option<String>,
    /// If provided, PMcore will use this prior instead of a "uniform" prior, see `sobol::generate` for details.
    pub prior: Option<String>,
    /// If provided, and [Config::output] is true, PMcore will write the output to this **relative** path. Defaults to `outputs/`
    #[serde(default = "default_output_folder")]
    pub output_folder: Option<String>,
}

/// General configuration settings
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Maximum number of cycles
    #[serde(default = "default_cycles")]
    pub cycles: usize,
    /// Denotes the algorithm to use, `NPAG` is the only supported algorithm for now.
    pub engine: String,
    #[serde(default = "default_seed")]
    /// Default seed for the initialization
    pub seed: usize,
    /// Default number of points in the initial grid
    #[serde(default = "default_10k")]
    pub init_points: usize,
    #[serde(default = "default_false")]
    pub tui: bool,
    /// If true (default), write outputs to files. Output path is set with [Paths::output_folder]
    #[serde(default = "default_true")]
    pub output: bool,
    /// If true (default), cache predicted values
    #[serde(default = "default_true")]
    pub cache: bool,
    /// The interval (in hours) at which to generate output predictions
    #[serde(default = "default_idelta")]
    pub idelta: f64,
    /// Maximum log level for the logger
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Vector if IDs to exclude
    pub exclude: Option<Vec<String>>,
    /// Generate predictions at [Config::idelta] intervals to this many hours after the last dose
    #[serde(default = "default_tad")]
    pub tad: f64,
    #[serde(default = "default_sampler")]
    pub sampler: Option<String>,
}

/// Random parameters to be estimated
///
/// This struct contains the random parameters to be estimated. The parameters are specified as a hashmap, where the key is the name of the parameter, and the value is a tuple containing the upper and lower bounds of the parameter.
///
/// # Example
///
/// ```toml
/// [random]
/// alpha = [0.0, 1.0]
/// beta = [0.0, 1.0]
/// ```
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Random {
    #[serde(flatten)]
    pub parameters: Map<String, Value>,
}

impl Random {
    /// Get the upper and lower bounds of a random parameter from its key
    pub fn get(&self, key: &str) -> Option<(f64, f64)> {
        self.parameters
            .get(key)
            .and_then(|v| v.as_array())
            .map(|v| {
                let lower = v[0].as_float().unwrap();
                let upper = v[1].as_float().unwrap();
                (lower, upper)
            })
    }

    /// Returns a vector of the names of the random parameters
    pub fn names(&self) -> Vec<String> {
        self.parameters.keys().cloned().collect()
    }

    /// Returns a vector of the upper and lower bounds of the random parameters
    pub fn ranges(&self) -> Vec<(f64, f64)> {
        self.parameters
            .values()
            .map(|v| {
                let lower = v.as_array().unwrap()[0].as_float().unwrap();
                let upper = v.as_array().unwrap()[1].as_float().unwrap();
                (lower, upper)
            })
            .collect()
    }

    /// Validate the boundaries of the random parameters
    pub fn validate(&self) -> Result<(), String> {
        for (key, range) in &self.parameters {
            let range = range.as_array().unwrap();
            let lower = range[0].as_float().unwrap();
            let upper = range[1].as_float().unwrap();
            if lower >= upper {
                return Err(format!(
                    "In key '{}', lower bound ({}) is not less than upper bound ({})",
                    key, lower, upper
                ));
            }
        }
        Ok(())
    }
}

/// Parameters which are estimated, but fixed for the population
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Fixed {
    #[serde(flatten)]
    pub parameters: HashMap<String, f64>,
}

/// Parameters which are held constant
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Constant {
    #[serde(flatten)]
    pub parameters: HashMap<String, f64>,
}

/// Defines the error model and polynomial to be used
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Error {
    pub value: f64,
    pub class: String,
    pub poly: (f64, f64, f64, f64),
}

impl Error {
    pub fn validate(&self) -> Result<(), String> {
        if self.value < 0.0 {
            return Err(format!(
                "Error value must be non-negative, got {}",
                self.value
            ));
        }

        _ = self.error_type();

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

/// Parses the settings from a TOML configuration file
///
/// This function parses the settings from a TOML configuration file. The settings are validated, and a copy of the settings is written to file.
///
/// Entries in the TOML file may be overridden by environment variables. The environment variables must be prefixed with `PMCORE__`, and the TOML entry must be in uppercase. For example, the TUI may be disabled by setting the environment variable `PMCORE__CONFIG__TUI=false` Note that a double underscore, `__`, is used as the separator, as some settings may contain a single underscore, such as `PMCORE__CONFIG__LOG_LEVEL`.
pub fn read_settings(path: String) -> Result<Settings, config::ConfigError> {
    let settings_path = path;

    let parsed = eConfig::builder()
        .add_source(config::File::with_name(&settings_path).format(config::FileFormat::Toml))
        .add_source(config::Environment::with_prefix("PMCORE").separator("__"))
        .build()?;

    // Deserialize settings to the Settings struct
    let settings: Settings = parsed.try_deserialize()?;

    // Validate entries
    settings
        .random
        .validate()
        .map_err(config::ConfigError::Message)?;
    settings
        .error
        .validate()
        .map_err(config::ConfigError::Message)?;

    // Write a copy of the settings to file if output is enabled
    if settings.config.output {
        if let Err(error) = write_settings_to_file(&settings) {
            eprintln!("Could not write settings to file: {}", error);
        }
    }

    Ok(settings) // Return the settings wrapped in Ok
}

/// Writes a copy of the parsed settings to file
///
/// This function writes a copy of the parsed settings to file. The file is written to the current working directory, and is named `settings.json`.
pub fn write_settings_to_file(settings: &Settings) -> Result<(), std::io::Error> {
    let serialized = serde_json::to_string_pretty(settings)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let mut file = crate::routines::output::create_output_file(settings, "settings.json")?;
    std::io::Write::write_all(&mut file, serialized.as_bytes())?;
    Ok(())
}

// *********************************
// Default values for deserializing
// *********************************
fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_seed() -> usize {
    347
}

fn default_idelta() -> f64 {
    0.0 //0.12
}

fn default_tad() -> f64 {
    0.0
}

fn default_10k() -> usize {
    10_000
}

fn default_cycles() -> usize {
    100
}

fn default_output_folder() -> Option<String> {
    Some("outputs/".to_string())
}

fn default_sampler() -> Option<String> {
    Some("sobol".to_string())
}
