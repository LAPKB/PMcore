use core::mem::MaybeUninit;
use dyn_stack::{DynArray, ReborrowMut};
use dyn_stack::{DynStack, StackReq};

fn main() {
    const A: usize = 10;
    let mut buf = [MaybeUninit::uninit(); StackReq::new::<f64>(A).unaligned_bytes_required()];
    let mut stack = DynStack::new(&mut buf);
    let (mut array, _) = stack.rb_mut().make_with::<f64, _>(A, |i| i as f64);

    // We can read from the arrays,
    assert_eq!(array[0], 0.0);
    assert_eq!(array[1], 1.0);
    assert_eq!(array[2], 2.0);

    array[1] = 3.0;
    assert_eq!(array[1], 3.0);
}
