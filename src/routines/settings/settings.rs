#![allow(dead_code)]

use std::collections::HashMap;
use config::Config as eConfig;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub run: Run,
}

#[derive(Debug, Deserialize, Clone)]
struct Run {
    pub paths: Paths,
    pub config: Config,
    pub random: Random,
    pub fixed: Option<Fixed>,
    pub constant: Option<Constant>,
    pub error: Error,
}

#[derive(Debug, Deserialize, Clone)]
struct Paths {
    pub data: String,
    pub log: Option<String>,
    pub prior: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct Config {
    pub cycles: usize,
    pub engine: String,
    #[serde(default = "default_seed")]
    pub seed: usize,
    #[serde(default)] // Defaults to FALSE
    pub tui: bool,
    #[serde(default = "default_true")]
    pub output: bool,
    #[serde(default = "default_true")]
    pub cache: bool,
    #[serde(default = "default_idelta")]
    pub idelta: f64,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

#[derive(Debug, Deserialize, Clone)]
struct Random {
    #[serde(flatten)]
    pub parameters: HashMap<String, [f64; 2]>,
}

#[derive(Debug, Deserialize, Clone)]
struct Fixed {
    #[serde(flatten)]
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Deserialize, Clone)]
struct Constant {
    #[serde(flatten)]
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Deserialize, Clone)]
struct Error {
    pub value: f64,
    pub class: String,
    pub poly: (f64, f64, f64, f64),
}

pub fn read_settings() -> Result<Settings, config::ConfigError> {
    let parsed = eConfig::builder()
        .add_source(
            config::File::with_name("examples/new_settings.toml")
                .format(config::FileFormat::Toml),
        )
        .add_source(config::Environment::with_prefix("NPCORE").separator("_"))
        .build()?;

    let settings: Settings = parsed.try_deserialize()?;
    Ok(settings)  // Return the settings wrapped in Ok
}

// *********************************
// Default values for deserializing
// *********************************
fn default_true() -> bool {
    true
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
