use ndarray::prelude::*;
use np_core::prelude::*;

fn main() {
    let a = array![
        [0.4,0.0,0.2,0.1,0.3],
        [0.4,0.1,0.3,0.2,0.0],
        [0.4,0.2,0.0,0.3,0.1],
        [0.4,0.3,0.1,0.0,0.2],
        [0.3,0.0,0.2,0.1,0.4],
    ];

    let perm = a.sort_axis_by(Axis(1), |i, j| a.column(i).sum() > a.column(j).sum());
    println!("{:?}", perm);

    println!("{:?}", a);
    let c = a.permute_axis(Axis(1), &perm);
    println!("{:?}", c);
}
