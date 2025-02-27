use crate::routines::condensation::prune;
use faer::{Mat, Row};

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
    theta: &mut Mat<f64>,
    eps: f64,
    ranges: &[(f64, f64)],
    min_dist: f64,
) -> Mat<f64> {
    let old_theta = theta.clone();
    // For each support point (row) in theta
    for spp in old_theta.row_iter() {
        // For each parameter in the support point
        for (j, val) in spp.iter().enumerate() {
            // Calculate the distance at which the new points will be generated
            let l = eps * (ranges[j].1 - ranges[j].0);

            // Add the distance to the current support point
            if val + l < ranges[j].1 {
                let mut plus = Row::zeros(spp.ncols());
                plus[j] = l;
                plus = plus + spp;
                prune(theta, plus, ranges, min_dist);
            }

            // Subtract the distance from the current support point
            if val - l > ranges[j].0 {
                let mut minus = Row::zeros(spp.ncols());
                minus[j] = -l;
                minus = minus + spp;
                prune(theta, minus, ranges, min_dist);
            }
        }
    }
    theta.to_owned()
}
