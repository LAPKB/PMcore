use faer::Mat;
use faer::Row;

/// Prunes the `theta` array based on the `candidate` array and `limits`.
///
/// This function checks if the `candidate` support point is within the limits defined
/// by the user and also if is not too close to the current support points.
///
/// If the `candidate` is within the limits and is not too close to the current support points,
/// it is added to `theta`. Otherwise, it is discarded.
///
/// # Arguments
///
/// * `theta` - Current Support points.
/// * `candidate` - Candidate support point.
/// * `limits` - (min, max) limits for each dimension.
/// * `min_dist` - The minimum allowed distance between the candidate and the current support points.
pub fn prune(theta: &mut Mat<f64>, candidate: Row<f64>, limits: &[(f64, f64)], min_dist: f64) {
    for spp in theta.row_iter() {
        let mut dist: f64 = 0.;
        for (i, val) in candidate.clone().iter().enumerate() {
            dist += (val - spp.get(i)).abs() / (limits[i].1 - limits[i].0);
        }
        if dist <= min_dist {
            tracing::debug!(
                 "Prune rejected point {:#?} for being too close to an existing support points (distance {} >= {}.",
                 candidate,
                 dist,
                    min_dist
            );
            return;
        }
    }

    // Convert the candidate Row to a 1xN matrix
    let row = candidate;

    // Create a new matrix by vertical concatenation
    let new_theta = Mat::<f64>::from_fn(theta.nrows() + 1, theta.ncols(), |i, j| {
        if i < theta.nrows() {
            *theta.get(i, j)
        } else {
            *row.get(j)
        }
    });

    // Update the original matrix with the new concatenated matrix
    *theta = new_theta;
}
