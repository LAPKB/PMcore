use linfa_linalg::qr::QR;
use ndarray::{array, Array, Array1, Array2, Axis};
use ndarray_stats::DeviationExt;
use np_core::prelude::{PermuteArray, SortArray};
fn norm_zero(a: &Array1<f64>) -> f64 {
    let zeros: Array1<f64> = Array::zeros(a.len());
    a.l2_dist(&zeros).unwrap()
}
fn main() {
    let mut x: Array2<f64> = array![
        [0.4, 0.3, 0.2, 0.1, 0.0],
        [0.0, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.4, 0.0, 0.3, 0.2],
        [0.4, 0.2, 0.1, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let perm = x.sort_axis_by(Axis(1), |i, j| {
        norm_zero(&x.column(i).to_owned()) > norm_zero(&x.column(j).to_owned())
    });

    let a = x.qr().unwrap();
    // dbg!(&a);
    // let q = &a.generate_q();
    let r = a.into_r();
    // dbg!(&q);
    dbg!(&r);
    // dbg!(q.dot(&r));
    dbg!(&perm.indices);

    x = x.permute_axis(Axis(1), &perm);

    let a = x.qr().unwrap();
    // dbg!(&a);
    // let q = &a.generate_q();
    let r = a.into_r();
    // dbg!(&q);
    dbg!(&r);
    // dbg!(q.dot(&r));
    dbg!(&perm.indices);
}
