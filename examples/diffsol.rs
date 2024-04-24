use std::path::Path;

use pmcore::routines::data::parse_pmetrics::read_pmetrics;

fn main() {
    let data = read_pmetrics(Path::new("examples/data/bimodal_ke.csv")).unwrap();
    println!("{}", data);
}
