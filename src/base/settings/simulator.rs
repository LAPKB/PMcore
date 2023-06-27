use serde_derive::Deserialize;
use std::fs;
use std::process::exit;
use toml;

#[derive(Deserialize, Clone, Debug)]
pub struct Data {
    pub paths: Paths,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Paths {
    pub data: String,
    pub theta: String,
}

pub fn read(filename: String) -> Data {
    let contents = match fs::read_to_string(&filename) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}", e);
            eprintln!("ERROR: Could not read file {}", &filename);
            exit(1);
        }
    };
    let parse: Data = match toml::from_str(&contents) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}", e);
            eprintln!("ERROR: Unable to load data from {}", &filename);
            exit(1);
        }
    };
    Data {
        paths: Paths {
            data: parse.paths.data,
            theta: parse.paths.theta,
        },
    }
}
