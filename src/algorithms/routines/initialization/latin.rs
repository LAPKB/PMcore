use anyhow::Result;
use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, OwnedRepr};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Generates a 2-dimensional array containing Latin Hypercube Sampling points within the given ranges.
///
/// This function samples the space using a Latin Hypercube Sampling of `n_points` points, distributed along `range_params.len()` dimensions.
///
/// # Arguments
///
/// * `n_points` - The number of points in the Latin Hypercube Sampling.
/// * `range_params` - A vector of tuples, where each tuple represents the minimum and maximum value of a parameter.
/// * `seed` - The seed for the random number generator.
///
/// # Returns
///
/// A 2D array where each row is a point in the Latin Hypercube Sampling, and each column corresponds to a parameter.
/// The value of each parameter is scaled to be within the corresponding range.
///
pub fn generate(
    n_points: usize,
    range_params: &[(f64, f64)],
    seed: usize,
) -> Result<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>> {
    let n_params = range_params.len();
    let mut seq = Array::<f64, _>::zeros((n_points, n_params).f());
    let mut rng = StdRng::seed_from_u64(seed as u64);

    for j in 0..n_params {
        let (min, max) = range_params[j];
        let mut intervals: Vec<f64> = (0..n_points).map(|i| i as f64).collect();
        intervals.shuffle(&mut rng);

        for i in 0..n_points {
            let u = Uniform::new(0.0, 1.0);
            let value = u.sample(&mut rng);
            seq[[i, j]] = min + ((intervals[i] + value) / n_points as f64) * (max - min);
        }
    }
    Ok(seq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_lhs() {
        let result = generate(5, &[(0., 1.), (0., 100.), (0., 1000.)], 42).unwrap();
        assert_eq!(result.shape(), &[5, 3]);
        assert_eq!(
            result,
            array![
                [0.08118035164615534, 97.33000555311399, 729.2901013820507],
                [0.8068685635909911, 34.201431099281876, 969.15809885882],
                [0.682991369237072, 56.04329822650589, 299.54535567497913],
                [0.5474848855448787, 16.367227512053397, 501.1955274259167],
                [0.36985032088988323, 72.73656820048906, 11.698952289960252]
            ]
        );
    }
}
