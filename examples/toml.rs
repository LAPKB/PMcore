use pmcore::prelude::*;
use settings::read_settings;

fn main() {
    let path = "examples/bimodal_ke/config.toml".to_string();
    for i in 0..1000 {
        let s = read_settings(path.clone()).unwrap();
        let keys: Vec<&String> = s.random.parameters.keys().collect();
        let values = s.random.parameters.values().collect::<Vec<&(f64, f64)>>();
        if keys != ["Ke", "V"] {
            println!("{}: {:?}", i, keys);
        }
        if values != [&(0.001, 3.0), &(25.0, 250.0)] {
            println!("{}: {:?}", i, values);
        }
    }
}
