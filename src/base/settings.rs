    use serde_derive::Deserialize;
    use std::fs;
    use std::process::exit;
    use toml;

    const FILENAME: &str =  "config.toml";

    #[derive(Deserialize)]
    pub struct Data {
        pub paths: Paths,
        pub config: Config
    }

    #[derive(Deserialize)]
    pub struct Paths {
        pub data: String
    }

    #[derive(Deserialize)]
    pub struct Config {
        pub cycles: u32,
        pub engine: String
    }

    pub fn read() -> Data{
        let contents = match fs::read_to_string(FILENAME){
            Ok(c) => c,
            Err(_) => {
                eprintln!("ERROR: Could not read file {}", FILENAME);
                exit(1);
            }
        };

        let config: Data = match toml::from_str(&contents){
            Ok(d) => d,
            Err(e) => {
                eprintln!("{}",e);
                eprintln!("ERROR: Unable to load data from {}", FILENAME);
                exit(1);
            }
        };
        config
    }