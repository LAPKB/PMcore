use crate::prelude::*;
use datafile::Scenario;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::{Array, Array2};
use sigma::Sigma;

const FRAC_1_SQRT_2PI: f64 =
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

/// Calculate the Ψ (psi) matrix, which contains the likelihood of each support point (column)
/// for each subject (row).
///
/// The Ψ (psi) matrix represents the likelihood of each subject's observations under different
/// scenarios or support points. It is computed based on the provided predicted values `ypred`,
/// a list of `Scenario` structures, and a sigma estimation method `sig`.
///
/// # Arguments
///
/// * `ypred` - A 2D Array containing predicted values for each subject and support point.
///            Rows represent subjects, and columns represent support points.
///
/// * `scenarios` - A reference to a Vec<Scenario> containing information about different scenarios.
///
/// * `sig` - A trait object implementing the Sigma trait, used to estimate sigma values for likelihood
///           calculations.
///
/// # Returns
///
/// A 2D Array of f64 representing the Ψ (psi) matrix. Each element of this matrix represents the
/// likelihood of a subject's observations under a specific support point scenario.
///
/// # Example
///
/// ```
/// use ndarray::{Array2, Array1};
/// use your_module::{calculate_psi, Scenario, Sigma};
///
/// // Define your scenarios and predicted values (ypred) here.
///
/// // Calculate the Ψ (psi) matrix.
/// let psi_matrix = calculate_psi(&ypred, &scenarios, &YourSigmaImplementation);
/// ```
///
/// In this example, `psi_matrix` will contain the likelihood values for each subject and scenario.
///
/// Note: This function assumes that the input data structures are correctly formatted and consistent.
///
pub fn calculate_psi<S>(
    ypred: &Array2<Array1<f64>>,
    scenarios: &Vec<Scenario>,
    sig: &S,
) -> Array2<f64>
where
    S: Sigma + Sync,
{
    let mut prob = Array2::<f64>::zeros((scenarios.len(), ypred.ncols()).f());
    // let mut prob2 = Array2::from_elem((3, 4), (0usize, 0usize, 0.0f64));
    prob.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(j, mut element)| {
                    let scenario = scenarios.get(i).unwrap();
                    let yobs = Array::from(scenario.obs.clone());
                    let sigma = sig.sigma(&yobs);
                    let ll = normal_likelihood(ypred.get((i, j)).unwrap(), &yobs, &sigma);
                    if ll.is_nan() || ll.is_infinite() {
                        log::info!(
                            "NaN or Inf Likelihood detected!\nLL:{:?}\nypred: {:?}\nsubject: {}\nSpp: {}",
                            ll,
                            &ypred.get((i, j)),
                            i,
                            j
                        )
                    }
                    element.fill(ll);
                });
        });
    prob
}

/// Calculate the normal likelihood
pub fn normal_likelihood(ypred: &Array1<f64>, yobs: &Array1<f64>, sigma: &Array1<f64>) -> f64 {
    let diff = (yobs - ypred).mapv(|x| x.powi(2));
    let two_sigma_sq = (2.0 * sigma).mapv(|x| x.powi(2));
    let aux_vec = FRAC_1_SQRT_2PI * (-&diff / two_sigma_sq).mapv(|x| x.exp()) / sigma;
    aux_vec.product()
}
