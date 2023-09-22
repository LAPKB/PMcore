use csv::ReaderBuilder;
use linfa_linalg::qr::QR;
use ndarray::{Array2, Axis};
use ndarray_csv::Array2Reader;
use np_core::prelude::{linalg::faer_qr_decomp, Permutation, PermuteArray};
use std::fs::File;

fn main() {
    let file = File::open("examples/data/n_psi.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
    let (r, perm) = faer_qr_decomp(&array_read);
    for i in 0..30 {
        println!("{:?}", r[[i, i]]);
    }
    println!("-------------------------------------");
    let n_psi = array_read.permute_axis(
        Axis(1),
        &Permutation {
            indices: perm.clone(),
        },
    );
    let r = n_psi.qr().unwrap().into_r();
    for i in 0..30 {
        println!("{:?}", r[[i, i]]);
    }
}
