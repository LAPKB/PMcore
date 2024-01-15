#![allow(dead_code)]

use config::Config as eConfig;
use serde::Deserialize;
use serde_derive::Serialize;
use std::collections::HashMap;
use serde_json;

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Settings {
    pub paths: Paths,
    pub config: Config,
    pub random: Random,
    pub fixed: Option<Fixed>,
    pub constant: Option<Constant>,
    pub error: Error,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Paths {
    pub data: String,
    pub log: Option<String>,
    pub prior: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
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

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Random {
    #[serde(flatten)]
    pub parameters: HashMap<String, (f64, f64)>,
}

impl Random {
    pub fn get(&self, key: &str) -> Option<&(f64, f64)> {
        self.parameters.get(key)
    }

    pub fn ranges(&self) -> Vec<(f64, f64)> {
        self.parameters.values().map(|&value| value).collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.parameters.keys().map(|key| key.to_string()).collect()
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Fixed {
    #[serde(flatten)]
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Constant {
    #[serde(flatten)]
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Error {
    pub value: f64,
    pub class: String,
    pub poly: (f64, f64, f64, f64),
}

pub fn read_settings(path: String) -> Result<Settings, config::ConfigError> {
    let settings_path = path;

    let parsed = eConfig::builder()
        .add_source(config::File::with_name(&settings_path).format(config::FileFormat::Toml))
        .add_source(config::Environment::with_prefix("NPCORE").separator("_"))
        .build()?;

    let settings: Settings = parsed.try_deserialize()?;

    // Write settings to file
    write_settings_to_file(&settings).expect("Could not write settings to file");

    Ok(settings) // Return the settings wrapped in Ok
}

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
