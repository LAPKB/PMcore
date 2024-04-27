macro_rules! fetch_params {
    ($p:expr, $($name:ident),*) => {
        let p = $p;
        let mut idx = 0;
        $(
            let $name = p.get(idx).unwrap().clone();
            idx += 1;
        )*
    };
}

fn main() {
    let p = [1, 2, 3, 4, 5];
    fetch_params!(p, k1, k2, k3, k4, k5);
    println!(
        "k1 = {}, k2 = {}, k3 = {}, k4 = {}, k5 = {}",
        k1, k2, k3, k4, k5
    ); // prints "k1 = 1, k2 = 2, k3 = 3, k4 = 4, k5 = 5"
}
