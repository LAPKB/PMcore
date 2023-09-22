use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut, StackReq};
use faer_core::{Mat, Parallelism};
use faer_qr::col_pivoting::compute;
use ndarray::Array2;

pub fn faer_qr_decomp(x: &Array2<f64>) -> (Array2<f64>, Vec<usize>) {
    let x = Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[[i, j]]);

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
        Parallelism::None,
        stack.rb_mut(),
        Default::default(),
    );
    let mut r: Array2<f64> = Array2::zeros((qr.nrows(), qr.ncols()));
    for i in 0..qr.nrows() {
        for j in 0..qr.ncols() {
            r[[i, j]] = if i <= j { qr.read(i, j) } else { 0.0 }
            // r[[i, j]] = qr.read(i, j);
        }
    }
    // dbg!(&r);
    // dbg!(&perm);

    (r, perm)
}
