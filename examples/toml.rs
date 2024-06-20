use pmcore::prelude::*;
use settings::read_settings;

fn main() {
    let path = "examples/bimodal_ke/config.toml".to_string();
    for i in 0..10 {
        let s = read_settings(path.clone()).unwrap();
        let keys: Vec<&String> = s.random.parameters.keys().collect();
        // let values = s.random.parameters.values();
        println!("{}: {:?}", i, keys);
    }
}
