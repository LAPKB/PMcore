//! Frank-Wolfe (Conditional Gradient) optimizer for weights on the simplex
// Each step: linear minimization oracle, then convex combination

use crate::structs::psi::Psi;
use crate::structs::weights::Weights;
use anyhow::Result;

/// Frank-Wolfe optimizer for weights on the simplex.
/// Returns (Weights, objective value). Only psi is input.
pub fn frank_wolfe_weights(psi: &Psi) -> Result<(Weights, f64)> {
    let n = psi.matrix().nrows();
    let m = psi.matrix().ncols();
    let max_iter = 1000;
    let tol = 1e-8;
    let eps = 1e-12;
    let mut x = vec![1.0 / m as f64; m];
    let mut obj_prev = f64::NEG_INFINITY;
    let mut s = vec![0.0; n];
    let mut g = vec![0.0; m];
    for t in 0..max_iter {
        // s = psi * x
        for i in 0..n {
            s[i] = 0.0;
            for j in 0..m {
                s[i] += psi.matrix().get(i, j) * x[j];
            }
            s[i] = s[i].max(eps);
        }
        // g = psi^T * (1.0 / s)
        for j in 0..m {
            g[j] = 0.0;
            for i in 0..n {
                g[j] += psi.matrix().get(i, j) / s[i];
            }
        }
        // Linear minimization oracle: minimize -g^T s over simplex
        let min_idx = g
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let mut s_fw = vec![0.0; m];
        s_fw[min_idx] = 1.0;
        // Step size: 2/(t+2) (standard for FW)
        let gamma = 2.0 / ((t + 2) as f64);
        for j in 0..m {
            x[j] = (1.0 - gamma) * x[j] + gamma * s_fw[j];
        }
        // Objective
        let obj = s.iter().map(|&si: &f64| si.ln()).sum::<f64>();
        if (obj - obj_prev).abs() < tol * obj.abs().max(1.0) {
            return Ok((Weights::from_vec(x), obj));
        }
        obj_prev = obj;
    }
    let obj = s.iter().map(|&si: &f64| si.ln()).sum::<f64>();
    Ok((Weights::from_vec(x), obj))
}
