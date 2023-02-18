    use serde_derive::Deserialize;
    use std::fs;
    use std::process::exit;
    use toml;


    #[derive(Deserialize)]
    pub struct Data {
        pub paths: Paths,
        pub config: Config
    }

    #[derive(Deserialize)]
    pub struct Paths {
        pub data: String,
        pub log_out: Option<String>
    }

    #[derive(Deserialize)]
    pub struct Config {
        pub cycles: u32,
        pub engine: String,
        pub init_points: usize
    }

    pub fn read(filename: String) -> Data{
        let contents = match fs::read_to_string(&filename){
            Ok(c) => c,
            Err(_) => {
                eprintln!("ERROR: Could not read file {}", &filename);
                exit(1);
            }
        };

        let config: Data = match toml::from_str(&contents){
            Ok(d) => d,
            Err(e) => {
                eprintln!("{}",e);
                eprintln!("ERROR: Unable to load data from {}", &filename);
                exit(1);
            }
        };
        config
    }