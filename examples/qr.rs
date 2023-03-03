use linfa_linalg::qr::QR;
use ndarray::{Array2, array, Axis};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

fn main(){
    let x:Array2<f64> = array![
        [0.4,0.3,0.2,0.1,0.0],
        [0.4,0.0,0.3,0.2,0.1],
        [0.4,0.1,0.0,0.3,0.2],
        [0.4,0.2,0.1,0.0,0.3],
        [0.3,0.4,0.2,0.1,0.0],
    ];

    // let a = x.qr().unwrap();
    // dbg!(&a);
    // let q = &a.generate_q();
    // let r = a.into_r();
    // dbg!(&q);
    // dbg!(&r);
    // dbg!(q.dot(&r));

    let mut n_psi = x.clone();
        n_psi.axis_iter_mut(Axis(0)).into_par_iter().for_each(
            |mut row| row /= row.sum()
        );
    dbg!(n_psi);
}