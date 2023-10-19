use ndarray::{Array, Array1, Array2};

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
                evaluate_spp(theta, plus, ranges, min_dist);
                // (n_spp, _) = theta.dim();
            }
            if val - l > ranges[j].0 {
                let mut minus = Array::zeros(spp.len());
                minus[j] = -l;
                minus = minus + spp;
                evaluate_spp(theta, minus, ranges, min_dist);
                // (n_spp, _) = theta.dim();
            }
        }
    }
    theta.to_owned()
}

pub fn evaluate_spp(
    theta: &mut Array2<f64>,
    candidate: Array1<f64>,
    limits: &[(f64, f64)],
    min_dist: f64,
) {
    for spp in theta.rows() {
        let mut dist: f64 = 0.;
        for (i, val) in candidate.clone().into_iter().enumerate() {
            dist += (val - spp.get(i).unwrap()).abs() / (limits[i].1 - limits[i].0);
        }
        if dist <= min_dist {
            return;
        }
    }
    theta.push_row(candidate.view()).unwrap();
}
