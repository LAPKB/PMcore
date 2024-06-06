use pmcore::prelude::*;
use settings::read_settings;

fn main() {
    let path = "examples/bimodal_ke/config.toml".to_string();
    let s = read_settings(path).unwrap();

    dbg!(s.random.parameters);
}