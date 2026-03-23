use anyhow::Result;
use faer::Mat;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;

use crate::prelude::Parameters;
use crate::structs::theta::Theta;

/// Generates an instance of [Theta] using Latin Hypercube Sampling.
///
/// # Arguments
///
/// * `parameters` - The [Parameters] struct, which contains the parameters to be sampled.
/// * `points` - The number of points to generate, i.e. the number of rows in the matrix.
/// * `seed` - The seed for the Sobol sequence generator.
///
/// # Returns
///
/// [Theta], a structure that holds the support point matrix
///
pub fn generate(parameters: &Parameters, points: usize, seed: usize) -> Result<Theta> {
    let params: Vec<(String, f64, f64)> = parameters
        .iter()
        .map(|p| (p.name.clone(), p.lower, p.upper))
        .collect();

    // Initialize random number generator with the provided seed
    let mut rng = StdRng::seed_from_u64(seed as u64);

    // Create and shuffle intervals for each parameter
    let mut intervals = Vec::new();
    for _ in 0..params.len() {
        let mut param_intervals: Vec<f64> = (0..points).map(|i| i as f64).collect();
        param_intervals.shuffle(&mut rng);
        intervals.push(param_intervals);
    }

    let rand_matrix = Mat::from_fn(points, params.len(), |i, j| {
        // Get the interval for this parameter and point
        let interval = intervals[j][i];
        let random_offset = rng.random::<f64>();
        // Calculate normalized value in [0,1]
        let unscaled = (interval + random_offset) / points as f64;
        // Scale to parameter range
        let (_name, lower, upper) = params.get(j).unwrap(); // Fixed: use j instead of i
        lower + unscaled * (upper - lower)
    });

    let theta = Theta::from_parts(rand_matrix, parameters.clone())?;

    Ok(theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Parameters;
    use faer::mat;

    #[test]
    fn test_latin_hypercube() {
        let params = Parameters::new()
            .add("a", 0.0, 1.0)
            .add("b", 0.0, 1.0)
            .add("c", 0.0, 1.0);

        let theta = generate(&params, 10, 22).unwrap();

        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 3);
    }

    #[test]
    fn test_latin_hypercube_values() {
        let params = Parameters::new()
            .add("a", 0.0, 1.0)
            .add("b", 0.0, 1.0)
            .add("c", 0.0, 1.0);

        let theta = generate(&params, 10, 22).unwrap();

        let expected = mat![
            [0.9318592685623417, 0.5609665425179973, 0.3351914901515939], //
            [0.5470144220416706, 0.13513808559222779, 0.1067962439473777], //
            [0.34525902829190547, 0.4636722699673962, 0.9142146621998218], //
            [0.24828355387285125, 0.8638104433695395, 0.41653980640777954], //
            [0.7642037770085612, 0.6806932027789437, 0.5608053599272136], //
            [0.19409389824004936, 0.9378790633419902, 0.6039530631991072], //
            [0.04886813284275151, 0.7140428162864041, 0.7855069414226704], //
            [0.6987026842780971, 0.32378779989236495, 0.8888807957183007], //
            [0.4221279608793599, 0.08001464382386277, 0.20689573661666943], //
            [0.8310112718320113, 0.29390050406905127, 0.04806137233953963], //
        ];

        assert_eq!(theta.matrix().to_owned(), expected);
    }
}
