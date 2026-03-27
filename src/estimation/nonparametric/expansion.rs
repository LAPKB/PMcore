use crate::estimation::nonparametric::Theta;
use anyhow::Result;
use faer::Row;

/// Implements the adaptive grid algorithm for support point expansion.
pub fn adaptative_grid(
    theta: &mut Theta,
    eps: f64,
    ranges: &[(f64, f64)],
    min_dist: f64,
) -> Result<()> {
    let mut candidates = Vec::new();

    for spp in theta.matrix().row_iter() {
        for (j, val) in spp.iter().enumerate() {
            let l = eps * (ranges[j].1 - ranges[j].0);
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

    let keep = candidates
        .iter()
        .filter(|point| theta.check_point(point, min_dist))
        .cloned()
        .collect::<Vec<_>>();

    for point in keep {
        theta.add_point(point.as_slice())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ParameterSpace, ParameterSpec};
    use faer::mat;

    #[test]
    fn adaptive_grid_expands_points_within_bounds() {
        let parameters = ParameterSpace::new()
            .add(ParameterSpec::bounded("x", 0.0, 1.0))
            .add(ParameterSpec::bounded("y", 0.0, 1.0));
        let mut theta = Theta::from_parts(mat![[0.5, 0.5]], parameters).unwrap();
        let ranges = [(0.0, 1.0), (0.0, 1.0)];

        adaptative_grid(&mut theta, 0.1, &ranges, 0.05).unwrap();

        assert_eq!(theta.matrix().nrows(), 5);
    }
}