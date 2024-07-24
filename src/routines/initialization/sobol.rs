use anyhow::Result;
use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, OwnedRepr};
use sobol_burley::sample;

/// Generates a 2-dimensional array containing a Sobol sequence within the given ranges.
///
/// This function samples the space using a Sobol sequence of `n_points` points. distributed along `range_params.len()` dimensions.
///
/// This function is used to initialize an optimization algorithm.
/// The generated Sobol sequence provides the initial sampling of the step to be used in the first cycle of the optimization algorithm.
///
/// # Arguments
///
/// * `n_points` - The number of points in the Sobol sequence.
/// * `range_params` - A vector of tuples, where each tuple represents the minimum and maximum value of a parameter.
/// * `seed` - The seed for the Sobol sequence generator.
///
/// # Returns
///
/// A 2D array where each row is a point in the Sobol sequence, and each column corresponds to a parameter.
/// The value of each parameter is scaled to be within the corresponding range.
///
pub fn generate(
    n_points: usize,
    range_params: &Vec<(f64, f64)>,
    seed: usize,
) -> Result<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> {
    let n_params = range_params.len();
    let mut seq = Array::<f64, _>::zeros((n_points, n_params).f());
    for i in 0..n_points {
        let mut row = seq.slice_mut(s![i, ..]);
        let mut point: Vec<f64> = Vec::new();
        for j in 0..n_params {
            point.push(sample(i.try_into()?, j.try_into()?, seed as u32) as f64)
        }
        row.assign(&Array::from(point));
    }
    for i in 0..n_params {
        let mut column = seq.slice_mut(s![.., i]);
        let (min, max) = range_params.get(i).unwrap();
        column.par_mapv_inplace(|x| min + x * (max - min));
    }
    Ok(seq)
}

#[cfg(test)]
use crate::prelude::*;

#[test]
fn basic_sobol() {
    assert_eq!(
        initialization::sobol::generate(5, &vec![(0., 1.), (0., 1.), (0., 1.)], 347).unwrap(),
        ndarray::array![
            [0.10731887817382813, 0.14647412300109863, 0.5851038694381714],
            [0.9840304851531982, 0.7633365392684937, 0.19097506999969482],
            [0.38477110862731934, 0.734661340713501, 0.2616291046142578],
            [0.7023299932479858, 0.41038262844085693, 0.9158684015274048],
            [0.6016758680343628, 0.6171295642852783, 0.6263971328735352]
        ]
    )
}

#[test]
fn scaled_sobol() {
    assert_eq!(
        initialization::sobol::generate(5, &vec![(0., 1.), (0., 2.), (-1., 1.)], 347).unwrap(),
        ndarray::array![
            [
                0.10731887817382813,
                0.29294824600219727,
                0.17020773887634277
            ],
            [0.9840304851531982, 1.5266730785369873, -0.6180498600006104],
            [0.38477110862731934, 1.469322681427002, -0.4767417907714844],
            [0.7023299932479858, 0.8207652568817139, 0.8317368030548096],
            [0.6016758680343628, 1.2342591285705566, 0.2527942657470703]
        ]
    )
}

//TODO: It should be possible to avoid one of the for-loops
//this improvement should happen automatically if switching columns with rows.
//theta0 = hcat([a .+ (b - a) .* Sobol.next!(s) for i = 1:n_theta0]...)

//TODO: Implement alternative samplers, such as uniform and Normal distributions
