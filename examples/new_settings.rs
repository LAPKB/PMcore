#![allow(dead_code)]

use std::collections::HashMap;
use config::Config as eConfig;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Settings {
    run: Vec<Run>,
}

#[derive(Debug, Deserialize)]
struct Run {
    paths: Paths,
    config: Config,
    random: Random,
    fixed: Option<Fixed>,
    constant: Option<Constant>,
    error: Error,
}

#[derive(Debug, Deserialize)]
struct Paths {
    data: String,
    log: Option<String>,
    prior: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Config {
    cycles: usize,
    engine: String,
    #[serde(default = "default_seed")]
    seed: usize,
    #[serde(default)] // Defaults to FALSE
    tui: bool,
    #[serde(default = "default_true")]
    output: bool,
    #[serde(default = "default_true")]
    cache: bool,
    #[serde(default = "default_idelta")]
    idelta: f64,
    #[serde(default = "default_log_level")]
    log_level: String,
}

#[derive(Debug, Deserialize)]
struct Random {
    #[serde(flatten)]
    parameters: HashMap<String, [f64; 2]>,
}

#[derive(Debug, Deserialize)]
struct Fixed {
    #[serde(flatten)]
    parameters: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct Constant {
    #[serde(flatten)]
    parameters: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct Error {
    value: f64,
    class: String,
    poly: [f64; 4],
}

fn main() -> Result<(), config::ConfigError> {
    let parsed = eConfig::builder()
        .add_source(
            config::File::with_name("examples/new_settings.toml").format(config::FileFormat::Toml),
        )
        .add_source(config::Environment::with_prefix("NPCORE").separator("_"))
        .build()?;

    let settings: Settings = parsed.try_deserialize()?;

    println!("{:#?}", settings);
    Ok(())
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
