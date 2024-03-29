use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, OwnedRepr};
use sobol_burley::sample;

/// Generates a 2-dimensional array containing a Sobol sequence within the given ranges
/// # Returns
/// A 2D array where each row is a point, and each column corresponds to a parameter.
pub fn generate(
    n_points: usize,
    range_params: &Vec<(f64, f64)>,
    seed: usize,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let n_params = range_params.len();
    let mut seq = Array::<f64, _>::zeros((n_points, n_params).f());
    for i in 0..n_points {
        let mut row = seq.slice_mut(s![i, ..]);
        let mut point: Vec<f64> = Vec::new();
        for j in 0..n_params {
            point.push(sample(i.try_into().unwrap(), j.try_into().unwrap(), seed as u32) as f64)
        }
        row.assign(&Array::from(point));
    }
    for i in 0..n_params {
        let mut column = seq.slice_mut(s![.., i]);
        let (min, max) = range_params.get(i).unwrap();
        column.par_mapv_inplace(|x| min + x * (max - min));
    }
    seq
}

//TODO: It should be possible to avoid one of the for-loops
//this improvement should happen automatically if switching columns with rows.
//theta0 = hcat([a .+ (b - a) .* Sobol.next!(s) for i = 1:n_theta0]...)

//TODO: Implement alternative samplers, such as uniform and Normal distributions
