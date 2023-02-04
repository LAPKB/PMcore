use sobol_burley::sample;
use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, OwnedRepr};
pub fn sobol_seq(n_points: usize, range_params: Vec<(f32,f32)>, seed: u32) -> ArrayBase<OwnedRepr<f32>,Dim<[usize; 2]>>{
    let n_params = range_params.len();
    let mut seq = Array::<f32, _>::zeros((n_points, n_params).f());
    for i in 0..n_points {
        let mut row = seq.slice_mut(s![i,..]);
        let mut point = Vec::new();
        for j in 0..n_params{
            point.push(sample(i.try_into().unwrap(), j.try_into().unwrap(), seed))
        }
        row.assign(&Array::from(point));
    }
    for i in 0..n_params{
        let mut column = seq.slice_mut(s![..,i]);
        let (min, max) = range_params.get(i).unwrap();
        let scaled_column = column.mapv(|x| min + x * (max-min));
        column.assign(&scaled_column);
    }
    seq
}

//TODO: It should be possible to avoid one of the for-loops

//theta0 = hcat([a .+ (b - a) .* Sobol.next!(s) for i = 1:n_theta0]...)