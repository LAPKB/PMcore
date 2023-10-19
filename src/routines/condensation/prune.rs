use ndarray::{Array1, Array2};

pub fn prune(
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
