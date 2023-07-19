use serde_derive::Deserialize;
use std::fs;
use std::process::exit;
use toml::value::Array;
use toml::{self, Table};

#[derive(Deserialize, Clone, Debug)]
pub struct Data {
    pub computed: Computed,
    pub parsed: Parsed,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Computed {
    pub random: Range,
    pub constant: Single,
    pub fixed: Single,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Error {
    pub value: f64,
    pub class: String,
    pub poly: (f64, f64, f64, f64),
}

#[derive(Deserialize, Clone, Debug)]
pub struct Range {
    pub names: Vec<String>,
    pub ranges: Vec<(f64, f64)>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Single {
    pub names: Vec<String>,
    pub values: Vec<f64>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Parsed {
    pub paths: Paths,
    pub config: Config,
    pub random: Table,
    pub fixed: Option<Table>,
    pub constant: Option<Table>,
    pub error: Error,
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
    pub pmetrics_outputs: Option<bool>,
    pub exclude: Option<Array>,
    pub cache: Option<bool>,
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

    let parsed: Parsed = match toml::from_str(&contents) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}", e);
            eprintln!("ERROR: Unable to load data from {}", &filename);
            exit(1);
        }
    };
    //Pri
    let mut pr = vec![];
    let mut pn = vec![];
    for (name, range) in &parsed.random {
        let range = range.as_array().unwrap();
        if range.len() != 2 {
            eprintln!(
                "ERROR: Ranges can only have 2 elements, {} found",
                range.len()
            );
            eprintln!("ERROR: In {:?}: {:?}", name, range);
            exit(1);
        }
        pn.push(name.clone());
        pr.push((range[0].as_float().unwrap(), range[1].as_float().unwrap()));
    }
    //Constant
    let mut cn = vec![];
    let mut cv = vec![];
    if let Some(constant) = &parsed.constant {
        for (name, value) in constant {
            cn.push(name.clone());
            cv.push(value.as_float().unwrap());
        }
    }

    //Randfix
    let mut rn = vec![];
    let mut rv = vec![];
    if let Some(randfix) = &parsed.fixed {
        for (name, value) in randfix {
            rn.push(name.clone());
            rv.push(value.as_float().unwrap());
        }
    }

    Data {
        computed: Computed {
            random: Range {
                names: pn,
                ranges: pr,
            },
            constant: Single {
                names: cn,
                values: cv,
            },
            fixed: Single {
                names: rn,
                values: rv,
            },
        },
        parsed,
    }
}
