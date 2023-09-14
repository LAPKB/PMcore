use ndarray::{array, s, Array2};

fn main() {
    let mut n_psi = array![[0.4, 0.3, 0.2, 0.1, 0.0]];
    if n_psi.ncols() > n_psi.nrows() {
        let nrows = n_psi.nrows();
        let ncols = n_psi.ncols();

        let diff = ncols - nrows;
        let zeros = Array2::<f64>::zeros((diff, ncols));
        let mut new_n_psi = Array2::<f64>::zeros((nrows + diff, ncols));
        new_n_psi.slice_mut(s![..nrows, ..]).assign(&n_psi);
        new_n_psi.slice_mut(s![nrows.., ..]).assign(&zeros);
        n_psi = new_n_psi;
    }
    dbg!(n_psi);
}
