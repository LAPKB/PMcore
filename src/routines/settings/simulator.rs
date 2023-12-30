use serde_derive::Deserialize;
use std::fs;
use std::process::exit;
use toml;

#[derive(Deserialize, Clone, Debug)]
pub struct Settings {
    pub paths: Paths,
    pub config: Config,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Paths {
    pub data: String,
    pub theta: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub idelta: Option<f64>,
    pub tad: Option<f64>,
}

pub fn read(filename: String) -> Settings {
    let contents = match fs::read_to_string(&filename) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}", e);
            eprintln!("ERROR: Could not read file {}", &filename);
            exit(1);
        }
    };
    let parse: Settings = match toml::from_str(&contents) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}", e);
            eprintln!("ERROR: Unable to load data from {}", &filename);
            exit(1);
        }
    };
    Settings {
        paths: Paths {
            data: parse.paths.data,
            theta: parse.paths.theta,
        },
        config: Config {
            idelta: parse.config.idelta,
            tad: parse.config.tad,
        },
    }
}
