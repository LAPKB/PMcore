use crate::structs::theta::Theta;
use anyhow::Result;
use faer::Row;

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
    theta: &mut Theta,
    eps: f64,
    ranges: &[(f64, f64)],
    min_dist: f64,
) -> Result<()> {
    let mut candidates = Vec::new();

    // Collect all points first to avoid borrowing conflicts
    for spp in theta.matrix().row_iter() {
        for (j, val) in spp.iter().enumerate() {
            let l = eps * (ranges[j].1 - ranges[j].0); //abs?
            if val + l < ranges[j].1 {
                let mut plus = Row::zeros(spp.ncols());
                plus[j] = l;
                plus += spp;
                candidates.push(plus.iter().copied().collect::<Vec<f64>>());
            }
            if val - l > ranges[j].0 {
                let mut minus = Row::zeros(spp.ncols());
                minus[j] = -l;
                minus += spp;
                candidates.push(minus.iter().copied().collect::<Vec<f64>>());
            }
        }
    }

    // Option 1: Check all points against the original theta, then add them
    let keep = candidates
        .iter()
        .filter(|point| theta.check_point(point, min_dist))
        .cloned()
        .collect::<Vec<_>>();

    for point in keep {
        theta.add_point(point.as_slice())?;
    }

    Ok(())

    // Option 2: Check and add points one by one
    // Now add all the points after the immutable borrow is released
    //for point in candidates {
    //    theta.suggest_point(point, min_dist, ranges);
    //}
}
/*
#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::theta::Theta;
    use faer::mat;

    #[test]
    fn test_expected() {
        let original = Theta::from(mat![[1.0, 10.0]]);

        let ranges = [(0.0, 1.0), (0.0, 10.0)];
        let eps = 0.1;
        let min_dist = 0.05;

        let mut theta = original.clone();
        adaptative_grid(&mut theta, eps, &ranges, min_dist);

        let expected = mat![[1.0, 10.0], [0.9, 10.0], [1.0, 9.0]];

        // Check that both matrices have the same number of rows
        assert_eq!(
            theta.matrix().nrows(),
            expected.nrows(),
            "Number of points in theta doesn't match expected"
        );

        // Check that all points in expected are in theta
        for i in 0..expected.nrows() {
            let expected_point = expected.row(i);
            let mut found = false;

            for j in 0..theta.matrix().nrows() {
                let theta_point = theta.matrix().row(j);

                // Check if points match (within small epsilon for floating-point comparison)
                if (expected_point[0] - theta_point[0]).abs() < 1e-10
                    && (expected_point[1] - theta_point[1]).abs() < 1e-10
                {
                    found = true;
                    break;
                }
            }

            assert!(
                found,
                "Expected point [{}, {}] not found in theta",
                expected_point[0], expected_point[1]
            );
        }

        // Check that all points in theta are in expected
        for i in 0..theta.matrix().nrows() {
            let theta_point = theta.matrix().row(i);
            let mut found = false;

            for j in 0..expected.nrows() {
                let expected_point = expected.row(j);

                // Check if points match (within small epsilon)
                if (theta_point[0] - expected_point[0]).abs() < 1e-10
                    && (theta_point[1] - expected_point[1]).abs() < 1e-10
                {
                    found = true;
                    break;
                }
            }

            assert!(
                found,
                "Point [{}, {}] in theta was not expected",
                theta_point[0], theta_point[1]
            );
        }
    }

    #[test]
    fn test_basic_expansion() {
        // Create initial theta with a single point [0.5, 0.5]
        let mut theta = Theta::from(mat![[0.5, 0.5]]);

        // Define ranges for two dimensions
        let ranges = [(0.0, 1.0), (0.0, 1.0)];

        // Set expansion parameters
        let eps = 0.1;
        let min_dist = 0.05;

        // Apply adaptive grid
        adaptative_grid(&mut theta, eps, &ranges, min_dist);

        // Should generate 4 new points around the original:
        // [0.6, 0.5], [0.4, 0.5], [0.5, 0.6], [0.5, 0.4]
        // Total 5 points including the original
        assert_eq!(theta.matrix().nrows(), 5);

        // Verify the original point is preserved
        let matrix = theta.matrix();
        let mut has_original = false;

        for i in 0..matrix.nrows() {
            let row = matrix.row(i);
            if (row[0] - 0.5).abs() < 1e-10 && (row[1] - 0.5).abs() < 1e-10 {
                has_original = true;
                break;
            }
        }
        assert!(has_original, "Original point should be preserved");

        // Verify expansion points were created
        let expected_points = vec![(0.6, 0.5), (0.4, 0.5), (0.5, 0.6), (0.5, 0.4)];
        for (x, y) in expected_points {
            let mut found = false;
            for i in 0..matrix.nrows() {
                let row = matrix.row(i);
                if (row[0] - x).abs() < 1e-10 && (row[1] - y).abs() < 1e-10 {
                    found = true;
                    break;
                }
            }
            assert!(found, "Expected point ({}, {}) not found", x, y);
        }
    }

    #[test]
    fn test_boundary_conditions() {
        // Create initial theta with points near boundaries
        let mut theta = Theta::from(mat![
            [0.05, 0.5], // Near lower boundary in x
            [0.95, 0.5], // Near upper boundary in x
            [0.5, 0.05], // Near lower boundary in y
            [0.5, 0.95], // Near upper boundary in y
        ]);

        let ranges = [(0.0, 1.0), (0.0, 1.0)];
        let eps = 0.1;
        let min_dist = 0.05;

        // Store original count
        let original_count = theta.matrix().nrows();

        adaptative_grid(&mut theta, eps, &ranges, min_dist);

        // Each point should generate fewer than 4 new points due to boundaries
        assert!(theta.matrix().nrows() > original_count);
        assert!(theta.matrix().nrows() < original_count + 4 * 4);

        // Verify no points are outside the range
        let matrix = theta.matrix();
        for i in 0..matrix.nrows() {
            let row = matrix.row(i);
            assert!(row[0] >= ranges[0].0 && row[0] <= ranges[0].1);
            assert!(row[1] >= ranges[1].0 && row[1] <= ranges[1].1);
        }
    }

    #[test]
    fn test_min_distance_constraint() {
        // Create initial theta with close points
        let mut theta = Theta::from(mat![
            [0.5, 0.5],
            [0.55, 0.5], // Close to first point
        ]);

        let ranges = [(0.0, 1.0), (0.0, 10.0)];
        let eps = 0.1;
        let min_dist = 0.15; // Large enough to prevent some points from being added

        adaptative_grid(&mut theta, eps, &ranges, min_dist);

        // We should have fewer points than the maximum possible expansion
        // due to the minimum distance constraint
        assert!(theta.matrix().nrows() < 2 + 2 * 4);
    }
}
 */
