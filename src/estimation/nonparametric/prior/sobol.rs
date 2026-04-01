use anyhow::Result;
use faer::Mat;
use sobol_burley::sample;

use crate::estimation::nonparametric::Theta;
use crate::model::ParameterSpace;

pub fn generate(
    parameters: impl Into<ParameterSpace>,
    points: usize,
    seed: usize,
) -> Result<Theta> {
    let parameters = parameters.into();
    let seed = seed as u32;
    let ranges = parameters.finite_ranges()?;

    let rand_matrix = Mat::from_fn(points, ranges.len(), |i, j| {
        let unscaled = sample((i).try_into().unwrap(), j.try_into().unwrap(), seed) as f64;
        let (lower, upper) = ranges[j];
        lower + unscaled * (upper - lower)
    });

    Theta::from_parts(rand_matrix, parameters.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ParameterSpace, ParameterSpec};

    #[test]
    fn sobol_generate_produces_requested_shape() {
        let params = ParameterSpace::new()
            .add(ParameterSpec::bounded("a", 0.0, 1.0))
            .add(ParameterSpec::bounded("b", 0.0, 1.0))
            .add(ParameterSpec::bounded("c", 0.0, 1.0));

        let theta = generate(&params, 10, 22).unwrap();
        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 3);
    }
}
