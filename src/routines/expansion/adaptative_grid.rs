use faer::Row;

use crate::structs::theta::Theta;

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
pub fn adaptative_grid(theta: &mut Theta, eps: f64, ranges: &[(f64, f64)], min_dist: f64) {
    let mut points_to_add = Vec::new();

    // Collect all points first to avoid borrowing conflicts
    for spp in theta.matrix().row_iter() {
        for (j, val) in spp.iter().enumerate() {
            let l = eps * (ranges[j].1 - ranges[j].0); //abs?
            if val + l < ranges[j].1 {
                let mut plus = Row::zeros(spp.ncols());
                plus[j] = l;
                plus = plus + spp;
                points_to_add.push(plus.iter().copied().collect());
            }
            if val - l > ranges[j].0 {
                let mut minus = Row::zeros(spp.ncols());
                minus[j] = -l;
                minus = minus + spp;
                points_to_add.push(minus.iter().copied().collect());
            }
        }
    }

    // Now add all the points after the immutable borrow is released
    for point in points_to_add {
        theta.suggest_point(point, min_dist, ranges);
    }
}
