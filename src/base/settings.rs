use serde_derive::Deserialize;
use toml::value::Array;
use std::fs;
use std::process::exit;
use toml;

#[derive(Deserialize, Clone)]
pub struct Data {
    pub paths: Paths,
    pub config: Config,
}

#[derive(Deserialize, Clone)]
pub struct Paths {
    pub data: String,
    pub log_out: Option<String>,
    pub prior_dist: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct Config {
    pub cycles: usize,
    pub engine: String,
    pub init_points: usize,
    pub seed: u32,
    pub tui: bool,
    pub pmetrics_outputs: Option<bool>,
    pub exclude: Option<Array>

}

pub fn read(filename: String) -> Data {
    let contents = match fs::read_to_string(&filename) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("ERROR: Could not read file {}", &filename);
            exit(1);
        }
    };

    let config: Data = match toml::from_str(&contents) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}", e);
            eprintln!("ERROR: Unable to load data from {}", &filename);
            exit(1);
        }
    };
    config
}
