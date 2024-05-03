use crate::prelude::*;
use datafile::Scenario;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::{Array, Array2};
use sigma::Sigma;

const FRAC_1_SQRT_2PI: f64 =
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 / 2.0;

/// Calculate the Ψ (psi) matrix, which contains the likelihood of each support point (column) for each subject (row)
fn calculate_psi<S>(ypred: &Array2<Array1<f64>>, scenarios: &Vec<Scenario>, sig: &S) -> Array2<f64>
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
                        tracing::info!(
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
