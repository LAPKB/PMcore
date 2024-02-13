#![allow(dead_code)]

use config::Config as eConfig;
use serde::Deserialize;
use serde_derive::Serialize;
use serde_json;
use std::collections::HashMap;

/// Contains all settings PMcore
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub paths: Paths,
    pub config: Config,
    pub random: Random,
    pub fixed: Option<Fixed>,
    pub constant: Option<Constant>,
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
}

/// General configuration settings
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub cycles: usize,
    pub engine: String,
    #[serde(default = "default_seed")]
    pub seed: usize,
    #[serde(default = "default_10k")]
    pub init_points: usize,
    #[serde(default = "default_false")]
    pub tui: bool,
    #[serde(default = "default_true")]
    pub output: bool,
    #[serde(default = "default_true")]
    pub cache: bool,
    #[serde(default = "default_idelta")]
    pub idelta: f64,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    pub exclude: Option<Vec<String>>,
    #[serde(default = "default_tad")]
    pub tad: f64,
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
    pub parameters: HashMap<String, (f64, f64)>,
}

impl Random {
    /// Get the upper and lower bounds of a random parameter from its key
    pub fn get(&self, key: &str) -> Option<&(f64, f64)> {
        self.parameters.get(key)
    }

    /// Returns a vector of tuples containing the names and ranges of the random parameters
    pub fn names_and_ranges(&self) -> Vec<(String, (f64, f64))> {
        let mut pairs: Vec<(String, (f64, f64))> = self
            .parameters
            .iter()
            .map(|(key, &(upper, lower))| (key.clone(), (upper, lower)))
            .collect();

        // Sorting alphabetically by name
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        pairs
    }
    /// Returns a vector of the names of the random parameters
    pub fn names(&self) -> Vec<String> {
        self.names_and_ranges()
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    /// Returns a vector of the upper and lower bounds of the random parameters
    pub fn ranges(&self) -> Vec<(f64, f64)> {
        self.names_and_ranges()
            .into_iter()
            .map(|(_, range)| range)
            .collect()
    }

    /// Validate the boundaries of the random parameters
    pub fn validate(&self) -> Result<(), String> {
        for (key, &(lower, upper)) in &self.parameters {
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
        Ok(())
    }
}

/// Parses the settings from a TOML configuration file
///
/// This function parses the settings from a TOML configuration file. The settings are validated, and a copy of the settings is written to file.
///
/// Entries in the TOML file may be overridden by environment variables. The environment variables must be prefixed with `PMCORE_`, and the TOML entry must be in uppercase. For example, the TUI may be disabled by setting the environment variable `PMCORE_TUI=false`.
pub fn read_settings(path: String) -> Result<Settings, config::ConfigError> {
    let settings_path = path;

    let parsed = eConfig::builder()
        .add_source(config::File::with_name(&settings_path).format(config::FileFormat::Toml))
        .add_source(config::Environment::with_prefix("PMCORE").separator("_"))
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
        write_settings_to_file(&settings).expect("Could not write settings to file");
    }

    Ok(settings) // Return the settings wrapped in Ok
}

/// Writes a copy of the parsed settings to file
///
/// This function writes a copy of the parsed settings to file. The file is written to the current working directory, and is named `settings.json`.
pub fn write_settings_to_file(settings: &Settings) -> Result<(), std::io::Error> {
    let serialized = serde_json::to_string_pretty(settings)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let file_path = "settings.json";
    let mut file = std::fs::File::create(file_path)?;
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
    0.12
}

fn default_tad() -> f64 {
    0.0
}

fn default_10k() -> usize {
    10_000
}
