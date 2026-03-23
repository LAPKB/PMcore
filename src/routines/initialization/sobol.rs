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
    let params: Vec<(String, f64, f64)> = parameters
        .iter()
        .map(|p| (p.name.clone(), p.lower, p.upper))
        .collect();

    let rand_matrix = Mat::from_fn(points, params.len(), |i, j| {
        let unscaled = sample((i).try_into().unwrap(), j.try_into().unwrap(), seed) as f64;
        let (_name, lower, upper) = params.get(j).unwrap();
        lower + unscaled * (upper - lower)
    });

    let theta = Theta::from_parts(rand_matrix, parameters.clone())?;
    Ok(theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Parameters;

    #[test]
    fn test_sobol() {
        let params = Parameters::new()
            .add("a", 0.0, 1.0)
            .add("b", 0.0, 1.0)
            .add("c", 0.0, 1.0);

        let theta = generate(&params, 10, 22).unwrap();

        assert_eq!(theta.nspp(), 10);
        assert_eq!(theta.matrix().ncols(), 3);
    }

    #[test]
    fn test_sobol_ranges() {
        let params = Parameters::new()
            .add("a", 0.0, 1.0)
            .add("b", 0.0, 1.0)
            .add("c", 0.0, 1.0);

        let theta = generate(&params, 10, 22).unwrap();

        theta.matrix().row_iter().for_each(|row| {
            row.iter().for_each(|&value| {
                assert!(value >= 0.0 && value <= 1.0);
            });
        });
    }

    #[test]
    fn test_sobol_values() {
        use faer::mat;
        let params = Parameters::new()
            .add("a", 0.0, 1.0)
            .add("b", 0.0, 1.0)
            .add("c", 0.0, 1.0);

        let theta = generate(&params, 10, 22).unwrap();

        let expected = mat![
            [0.05276215076446533, 0.609707236289978, 0.29471302032470703], //
            [0.6993427276611328, 0.4142681360244751, 0.6447571516036987],  //
            [0.860404372215271, 0.769607663154602, 0.1742185354232788],    //
            [0.3863574266433716, 0.07018685340881348, 0.9825305938720703], //
            [0.989533543586731, 0.19934570789337158, 0.4716176986694336],  //
            [0.29962968826293945, 0.899970293045044, 0.5400241613388062],  //
            [0.5577576160430908, 0.6990838050842285, 0.859503984451294],   //
            [
                0.19194257259368896,
                0.31645333766937256,
                0.042426824569702150
            ], //
            [0.8874167203903198, 0.5214653015136719, 0.5899909734725952],  //
            [0.35627472400665283, 0.4780532121658325, 0.42954015731811523]  //
        ];

        assert_eq!(theta.matrix().to_owned(), expected);
    }
}
