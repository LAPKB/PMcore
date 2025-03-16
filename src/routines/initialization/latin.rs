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
    let params: Vec<(String, f64, f64, bool)> = parameters
        .iter()
        .map(|p| (p.name.clone(), p.lower, p.upper, p.fixed))
        .collect();

    // Random parameters are sampled from the Sobol sequence
    let random_params: Vec<(String, f64, f64)> = params
        .iter()
        .filter(|(_, _, _, fixed)| !fixed)
        .map(|(name, lower, upper, _)| (name.clone(), *lower, *upper))
        .collect();

    // Initialize random number generator with the provided seed
    let mut rng = StdRng::seed_from_u64(seed as u64);

    // Create and shuffle intervals for each parameter
    let mut intervals = Vec::new();
    for _ in 0..random_params.len() {
        let mut param_intervals: Vec<f64> = (0..points).map(|i| i as f64).collect();
        param_intervals.shuffle(&mut rng);
        intervals.push(param_intervals);
    }

    let rand_matrix = Mat::from_fn(points, random_params.len(), |i, j| {
        // Get the interval for this parameter and point
        let interval = intervals[j][i];
        let random_offset = rng.random::<f64>();
        // Calculate normalized value in [0,1]
        let unscaled = (interval + random_offset) / points as f64;
        // Scale to parameter range
        let (_name, lower, upper) = random_params.get(j).unwrap(); // Fixed: use j instead of i
        lower + unscaled * (upper - lower)
    });

    // Fixed parameters are initialized to the middle of their range
    let fixed_params: Vec<(String, f64)> = params
        .iter()
        .filter(|(_, _, _, fixed)| *fixed)
        .map(|(name, lower, upper, _)| (name.clone(), (upper - lower) / 2.0))
        .collect();

    let theta = Theta::from_parts(rand_matrix, random_params, fixed_params);

    Ok(theta)
}
