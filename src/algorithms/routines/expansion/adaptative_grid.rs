use ndarray::{Array, Array2};

use crate::algorithms::routines::condensation::prune::prune;

/// Implements the adaptive grid algorithm for support point expansion.
///
/// This function generates up to 2 new support points in each dimension for each existing support point.
/// New support points are symmetrically placed around the original support point, at a distance of `eps` * (range_max - range_min).
/// If the new support point is too close to an existing support point, or it is outside the given range, it is discarded.
///
/// # Arguments
///
/// * `theta` - A mutable reference to a 2D array representing the existing support points.
/// * `eps` - A floating-point value representing the fraction of the range to use for generating new support points.
/// * `ranges` - A slice of tuples representing the range of values for each dimension.
/// * `min_dist` - A floating-point value representing the minimum distance between support points.
///
/// # Returns
///
/// A 2D array containing the updated support points after the adaptive grid expansion.
///
pub fn adaptative_grid(
    theta: &mut Array2<f64>,
    eps: f64,
    ranges: &[(f64, f64)],
    min_dist: f64,
) -> Array2<f64> {
    let old_theta = theta.clone();
    for spp in old_theta.rows() {
        for (j, val) in spp.into_iter().enumerate() {
            let l = eps * (ranges[j].1 - ranges[j].0); //abs?
                                                       // dbg!(val + l);
                                                       // dbg!(val - l);
            if val + l < ranges[j].1 {
                let mut plus = Array::zeros(spp.len());
                plus[j] = l;
                plus = plus + spp;
                prune(theta, plus, ranges, min_dist);
                // (n_spp, _) = theta.dim();
            } else {
                // tracing::debug!(
                //     "AG: Rejected point. Out of bounds. p:{}, p+eps:{}",
                //     val,
                //     val + l,
                // );
            }
            if val - l > ranges[j].0 {
                let mut minus = Array::zeros(spp.len());
                minus[j] = -l;
                minus = minus + spp;
                prune(theta, minus, ranges, min_dist);
                // (n_spp, _) = theta.dim();
            } else {
                // tracing::debug!(
                //     "AG: Rejected point. Out of bounds. p:{}, p-eps:{}",
                //     val,
                //     val - l
                // );
            }
        }
    }
    if theta.nrows() != (old_theta.nrows() + 2 * old_theta.ncols()) {
        // tracing::debug!(
        //     "3) The adaptive grid tried to add {} support points, from those {} were rejected.",
        //     2 * old_theta.ncols(),
        //     2 * old_theta.ncols() + old_theta.nrows() - theta.nrows()
        // );
    }
    theta.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adaptive_grid_1d() {
        // Initial theta: [[0.5]]
        let mut theta = array![[0.5]];
        let eps = 0.1;
        let ranges = [(0.0, 1.0)];
        let min_dist = 0.05;

        // Call adaptative_grid
        let new_theta = adaptative_grid(&mut theta, eps, &ranges, min_dist);

        // Expected theta: [[0.5], [0.6], [0.4]]
        let expected_theta = array![[0.5], [0.6], [0.4]];

        // Assert that new_theta matches expected_theta
        assert_eq!(new_theta, expected_theta);
    }

    #[test]
    fn test_adaptive_grid_2d() {
        // Initial theta: [[0.5, 0.5]]
        let mut theta = array![[0.5, 0.5]];
        let eps = 0.1;
        let ranges = [(0.0, 1.0), (0.0, 1.0)];
        let min_dist = 0.05;

        // Call adaptative_grid
        let new_theta = adaptative_grid(&mut theta, eps, &ranges, min_dist);

        // Expected new points are:
        // For dimension 0: [0.6, 0.5], [0.4, 0.5]
        // For dimension 1: [0.5, 0.6], [0.5, 0.4]
        // So expected theta is [[0.5, 0.5], [0.6, 0.5], [0.4, 0.5], [0.5, 0.6], [0.5, 0.4]]
        let expected_theta = array![[0.5, 0.5], [0.6, 0.5], [0.4, 0.5], [0.5, 0.6], [0.5, 0.4]];

        // Assert that new_theta matches expected_theta
        assert_eq!(new_theta, expected_theta);
    }

    #[test]
    fn test_adaptive_grid_min_dist() {
        // Initial theta: [[0.5]]
        let mut theta = array![[0.5]];
        let eps = 0.1;
        let ranges = [(0.0, 1.0)];
        let min_dist = 0.2;

        // Call adaptative_grid
        let new_theta = adaptative_grid(&mut theta, eps, &ranges, min_dist);

        // Since min_dist is 0.2, the new points at 0.6 and 0.4 are too close to 0.5 (distance 0.1)
        // So no new points should be added
        let expected_theta = array![[0.5]];

        // Assert that new_theta matches expected_theta
        assert_eq!(new_theta, expected_theta);
    }

    #[test]
    fn test_adaptive_grid_out_of_bounds() {
        // Initial theta: [[0.95]]
        let mut theta = array![[0.95]];
        let eps = 0.1;
        let ranges = [(0.0, 1.0)];
        let min_dist = 0.05;

        // Call adaptative_grid
        let new_theta = adaptative_grid(&mut theta, eps, &ranges, min_dist);

        // val + l = 0.95 + 0.1 = 1.05 > 1.0, so point at 1.05 is out of bounds and should not be added
        // val - l = 0.95 - 0.1 = 0.85, which is within range
        // So only [0.85] should be added
        let expected_theta = array![[0.95], [0.85]];

        // Assert that new_theta matches expected_theta
        assert_eq!(new_theta, expected_theta);
    }
}
