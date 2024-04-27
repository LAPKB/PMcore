use csv::ReaderBuilder;
use faer::Mat;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use std::fs::File;

fn main() {
    // let file = File::open("examples/data/n_psi.csv").unwrap();
    // let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    // let array_read: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
    // let (r, perm) = faer_qr_decomp(&array_read);
    // for i in 0..30 {
    //     println!("{:?}", r[[i, i]]);
    // }
    // println!("-------------------------------------");
    // let n_psi = array_read.permute_axis(
    //     Axis(1),
    //     &Permutation {
    //         indices: perm.clone(),
    //     },
    // );
    // let r = n_psi.qr().unwrap().into_r();
    // for i in 0..30 {
    //     println!("{:?}", r[[i, i]]);
    // }

    let file = File::open("examples/data/n_psi.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();

    let x = Mat::from_fn(array_read.nrows(), array_read.ncols(), |i, j| {
        array_read[[i, j]]
    });
    let x: Mat<f64> = x;
    let qr = x.col_piv_qr();
    let r_mat = qr.compute_r();
    let (forward, inverse) = qr.col_permutation().arrays();
    let mut r: Array2<f64> = Array2::zeros((r_mat.nrows(), r_mat.ncols()));
    for i in 0..r.nrows() {
        for j in 0..r.ncols() {
            r[[i, j]] = if i <= j { r_mat.read(i, j) } else { 0.0 }
            // r[[i, j]] = qr.read(i, j);
        }
    }
    // for i in 0..30 {
    //     println!("{:?}", r[[i, i]]);
    // }
    dbg!(forward);
    dbg!(inverse);
}
