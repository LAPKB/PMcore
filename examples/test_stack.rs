use std::hint::black_box;

const N: usize = 1_000_000;

type V = nalgebra::SVector<f64, N>;
fn main() {
    println!("Allocating...{N}");
    let stack = V::zeros();
    black_box(stack);
    println!("I survived the stack war!");
}

// cargo build --release --example test_stack
