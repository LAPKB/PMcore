use npcore::routines::evaluation::{ipm, ipm_faer};

fn main() {
    let matrix = ndarray::array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10., 11., 12.]
    ];
    let res = ipm::burke(&matrix);
    dbg!(res.unwrap());
    let res = ipm_faer::burke(&matrix);
}
