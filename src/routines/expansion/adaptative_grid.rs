use ndarray::{Array, Array2};

use crate::routines::condensation::prune::prune;

pub fn adaptative_grid(
    theta: &mut Array2<f64>,
    eps: f64,
    ranges: &[(f64, f64)],
    min_dist: f64,
) -> Array2<f64> {
    let old_theta = theta.clone();
    for spp in old_theta.rows() {
        for (j, val) in spp.into_iter().enumerate() {
            let l = eps * (ranges[j].1 - ranges[j].0); //abs?
            if val + l < ranges[j].1 {
                let mut plus = Array::zeros(spp.len());
                plus[j] = l;
                plus = plus + spp;
                prune(theta, plus, ranges, min_dist);
                // (n_spp, _) = theta.dim();
            }
            if val - l > ranges[j].0 {
                let mut minus = Array::zeros(spp.len());
                minus[j] = -l;
                minus = minus + spp;
                prune(theta, minus, ranges, min_dist);
                // (n_spp, _) = theta.dim();
            }
        }
    }
    theta.to_owned()
}
