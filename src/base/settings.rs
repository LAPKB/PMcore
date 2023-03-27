use serde_derive::Deserialize;
use std::fs;
use std::process::exit;
use toml::{self, Table};
use toml::value::Array;

#[derive(Deserialize, Clone, Debug)]
pub struct Data {
    pub paths: Paths,
    pub config: Config,
    pub randfix: Option<Table>,
    pub constant: Option<Table>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Paths {
    pub data: String,
    pub log_out: Option<String>,
    pub prior_dist: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub cycles: usize,
    pub engine: String,
    pub init_points: usize,
    pub seed: u32,
    pub tui: bool,
    pub parameter_names: Vec<String>,
    parameter_ranges: Vec<Vec<f64>>,
    pub param_ranges: Option<Vec<(f64,f64)>>,
    pub pmetrics_outputs: Option<bool>,
    pub exclude: Option<Array>,
    
}

pub fn read(filename: String) -> Data {
    let contents = match fs::read_to_string(&filename) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("ERROR: Could not read file {}", &filename);
            exit(1);
        }
    };

    let mut config: Data = match toml::from_str(&contents) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}", e);
            eprintln!("ERROR: Unable to load data from {}", &filename);
            exit(1);
        }
    };
    let mut p_r = vec![];
    for range in  &config.config.parameter_ranges{
        if range.len() != 2{
            eprintln!("ERROR: Ranges can only have 2 elements, {} found", range.len());
            eprintln!("ERROR: In {:?}", range);
            exit(1);
        }
        p_r.push((*range.get(0).unwrap(),*range.get(1).unwrap()));
    }
    config.config.param_ranges = Some(p_r);
    config
}
