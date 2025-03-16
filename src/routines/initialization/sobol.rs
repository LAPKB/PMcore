use crate::structs::theta::Theta;
use anyhow::Result;
use faer::Mat;

use sobol_burley::sample;

use crate::prelude::Parameters;

/// Generates a 2-dimensional array containing a Sobol sequence within the given ranges.
///
/// This function samples the space using a Sobol sequence of `n_points` points. distributed along `range_params.len()` dimensions.
///
/// This function is used to initialize an optimization algorithm.
/// The generated Sobol sequence provides the initial sampling of the step to be used in the first cycle of the optimization algorithm.
///
/// # Arguments
///
/// * `n_points` - The number of points in the Sobol sequence.
/// * `range_params` - A vector of tuples, where each tuple represents the minimum and maximum value of a parameter.
/// * `seed` - The seed for the Sobol sequence generator.
///
/// # Returns
///
/// A 2D array where each row is a point in the Sobol sequence, and each column corresponds to a parameter.
/// The value of each parameter is scaled to be within the corresponding range.
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

    let rand_matrix = Mat::from_fn(points, random_params.len(), |i, j| {
        let unscaled = sample((i).try_into().unwrap(), j.try_into().unwrap(), seed as u32) as f64;
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
