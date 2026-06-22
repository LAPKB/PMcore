use anyhow::Result;
use faer::Mat;
use rand::prelude::*;
use rand::rngs::StdRng;

use crate::estimation::nonparametric::Theta;
use crate::model::{BoundedParameter, ParameterSpace};

pub fn generate(parameters: &ParameterSpace<BoundedParameter>, points: usize, seed: usize) -> Result<Theta> {
    let ranges = parameters.finite_ranges();
    let mut rng = StdRng::seed_from_u64(seed as u64);

    let mut intervals = Vec::new();
    for _ in 0..ranges.len() {
        let mut param_intervals: Vec<f64> = (0..points).map(|i| i as f64).collect();
        param_intervals.shuffle(&mut rng);
        intervals.push(param_intervals);
    }

    let rand_matrix = Mat::from_fn(points, ranges.len(), |i, j| {
        let interval = intervals[j][i];
        let random_offset = rng.random::<f64>();
        let unscaled = (interval + random_offset) / points as f64;
        let (lower, upper) = ranges[j];
        lower + unscaled * (upper - lower)
    });

    Theta::from_parts(rand_matrix, parameters.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BoundedParameter, ParameterSpace};

    #[test]
    fn latin_generate_produces_requested_shape() {
        let params = ParameterSpace::<BoundedParameter>::new()
            .add("a", 0.0, 1.0)
            .add("b", 0.0, 1.0)
            .add("c", 0.0, 1.0);

        let theta = generate(&params, 10, 22).unwrap();
        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 3);
    }
}
