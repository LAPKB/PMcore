use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut, StackReq};
use faer_core::{Mat, Parallelism};
use faer_qr::col_pivoting::compute;
use ndarray::{array, Array2};

fn main() {
    let x = array![
        [0.4, 0.3, 0.2, 0.1, 0.0],
        [0.0, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.4, 0.0, 0.3, 0.2],
        [0.4, 0.2, 0.1, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    // 1.2 1.7 0.8 0.7 0.6
    // 1 0 2 3 4
    let x = Mat::with_dims(x.nrows(), x.ncols(), |i, j| x[[i, j]]);
    let rank = x.nrows().min(x.ncols());
    let blocksize = compute::recommended_blocksize::<f64>(x.nrows(), x.ncols());
    let mut mem =
            GlobalMemBuffer::new(StackReq::any_of([
                compute::qr_in_place_req::<f64>(
                    x.nrows(),
                    x.ncols(),
                    blocksize,
                    Parallelism::None,
                    Default::default(),
                )
                .unwrap(),
                faer_core::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_req::<
                    f64,
                >(x.nrows(), blocksize, x.ncols())
                .unwrap(),
            ]));
    let mut stack = DynStack::new(&mut mem);
    let mut qr = x;
    let mut h_factor = Mat::zeros(blocksize, rank);
    let size = qr.nrows().min(qr.ncols());
    let mut perm = vec![0; size];
    let mut perm_inv = vec![0; size];
    compute::qr_in_place(
        qr.as_mut(),
        h_factor.as_mut(),
        &mut perm,
        &mut perm_inv,
        Parallelism::Rayon(0),
        stack.rb_mut(),
        Default::default(),
    );
    // now the Householder bases are in the strictly lower trapezoidal part of `a`, and the
    // matrix R is in the upper triangular part of `qr`.
    dbg!(&qr);
    dbg!(&perm);
    dbg!(&perm_inv);

    let mut qr: Array2<f64> = Array2::zeros((qr.nrows(), qr.ncols()));
    for i in 0..qr.nrows() {
        for j in 0..qr.ncols() {
            qr[[i, j]] = qr[[perm_inv[i], j]];
        }
    }

    // let mut solution = b.clone();

    // // compute Q^H×B
    // householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
    //     qr.as_ref(),
    //     h_factor.as_ref(),
    //     Conj::Yes,
    //     solution.as_mut(),
    //     Parallelism::None,
    //     stack.rb_mut(),
    // );

    // solution.resize_with(rank, b.ncols(), |_, _| unreachable!());
}
