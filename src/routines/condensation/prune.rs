use ndarray::{Array1, Array2};

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
pub fn prune(
    theta: &mut Array2<f64>,
    candidate: Array1<f64>,
    limits: &[(f64, f64)],
    min_dist: f64,
) {
    for spp in theta.rows() {
        let mut dist: f64 = 0.;
        for (i, val) in candidate.clone().into_iter().enumerate() {
            dist += (val - spp.get(i).unwrap()).abs() / (limits[i].1 - limits[i].0);
        }
        if dist <= min_dist {
            panic!("point discarded");
            tracing::debug!(
                "Prune: Rejected point:{}. Too close to existing support points dist:{}.",
                candidate,
                dist
            );
            return;
        }
    }
    theta.push_row(candidate.view()).unwrap();
}
