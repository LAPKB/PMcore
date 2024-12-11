use anyhow::Result;
use ndarray::{s, Array2, Array3};
use pharmsol::{Data, Equation, ErrorModel};
use rand_distr::{Distribution, Normal};

/// Estimate the likelihood surface of a given model, distribution, and data
pub fn surface(
    theta: &Array2<f64>,
    data: &Data,
    model: impl Equation,
    errmod: ErrorModel,
) -> Result<Array3<f64>> {
    // Setup the random number generator
    let nsamples: usize = 100;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Initialize the Array3 to store the likelihood surface
    let mut sarray: Array3<f64> = Array3::zeros((theta.nrows() * nsamples, theta.ncols(), 1));

    // For each support point in theta, draw samples from the multivariate normal distribution
    // and evaluate the likelihood of the model given the data
    for (idx, spp) in theta.outer_iter().enumerate() {
        let orig = spp.to_owned();

        for i in 0..nsamples {
            let mut new = orig.clone();
            for (j, val) in orig.iter().enumerate() {
                new[j] = val + normal.sample(&mut rng);
            }

            let likelihood: f64 = data
                .get_subjects()
                .iter()
                .map(|s| model.estimate_likelihood(s, &new.to_vec(), &errmod, true))
                .sum();

            // Insert the likelihood into the likelihood surface
            sarray
                .slice_mut(s![idx * nsamples + i, .., ..])
                .assign(&Array2::from_elem((1, 1), likelihood));
        }
    }

    // Return the likelihood surface
    Ok(sarray)
}
