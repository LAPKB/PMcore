use pmcore::prelude::data::read_pmetrics;

fn main() {
    let path = std::path::Path::new("examples/data/bimodal_ke.csv");
    //let path = std::path::Path::new("examples/data/bimodal_ke_blocks.csv");
    let data = read_pmetrics(path).unwrap();

    println!("{}", data);
}
