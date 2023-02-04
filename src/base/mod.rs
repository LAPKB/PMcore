use sobol_burley::sample;
use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, OwnedRepr};
pub fn sobol_seq(n_points: usize, n_params: usize, _seed: u32) -> ArrayBase<OwnedRepr<f32>,Dim<[usize; 2]>>{
    let mut seq = Array::<f32, _>::zeros((n_points, n_params).f());
    for i in 0..n_points {
        let mut row = seq.slice_mut(s![i,..]);
        let mut point = Vec::new();
        for j in 0..n_params{
            point.push(sample(i.try_into().unwrap(), j.try_into().unwrap(), 0))
        }
        row.assign(&Array::from(point));
    }
    seq
}