use anyhow::Result;
use ndarray::prelude::*;
use ndarray::{Array, ArrayBase, OwnedRepr};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;

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
    range_params: &Vec<(f64, f64)>,
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
            let value = rng.random::<f64>();
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
        let result = generate(5, &vec![(0., 1.), (0., 100.), (0., 1000.)], 42).unwrap();
        assert_eq!(result.shape(), &[5, 3]);
        // Check that the values are within the expected range
        for i in 0..5 {
            assert!(result[[i, 0]] >= 0. && result[[i, 0]] <= 1.);
            assert!(result[[i, 1]] >= 0. && result[[i, 1]] <= 100.);
            assert!(result[[i, 2]] >= 0. && result[[i, 2]] <= 1000.);
        }

        // Check that the values are different
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert_ne!(result[[i, 0]], result[[j, 0]]);
                    assert_ne!(result[[i, 1]], result[[j, 1]]);
                    assert_ne!(result[[i, 2]], result[[j, 2]]);
                }
            }
        }
    }
}
