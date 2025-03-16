use crate::structs::theta::Theta;
use anyhow::Result;
use faer::Mat;

use sobol_burley::sample;

use crate::prelude::Parameters;

/// Generates an instance of [Theta] from a Sobol sequence.
///
/// The sequence samples [0, 1), and the values are scaled to the parameter ranges.
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
    let seed = seed as u32;
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

    let rand_matrix = Mat::from_fn(points, random_params.len(), |i, j| {
        let unscaled = sample((i).try_into().unwrap(), j.try_into().unwrap(), seed) as f64;
        let (_name, lower, upper) = random_params.get(j).unwrap();
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
