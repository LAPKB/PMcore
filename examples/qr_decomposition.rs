use linfa_linalg::qr::QR;
use ndarray::{Array2, array};

fn main(){
    let x:Array2<f64> = array![
        [0.4,0.3,0.2,0.1,0.0],
        [0.4,0.0,0.3,0.2,0.1],
        [0.4,0.1,0.0,0.3,0.2],
        [0.4,0.2,0.1,0.0,0.3],
        [0.3,0.4,0.2,0.1,0.0],
    ];

    let a = x.qr().unwrap();
    dbg!(&a);
    let q = &a.generate_q();
    let r = a.into_r();
    dbg!(&q);
    dbg!(&r);
    dbg!(q.dot(&r));

}