//! EM (multiplicative update) optimizer for weights on the simplex
// See docs/optimizers_for_weights.md for algorithm details.

use crate::structs::psi::Psi;
use crate::structs::weights::Weights;
use anyhow::Result;

/// EM (multiplicative update) optimizer for weights on the simplex.
/// Returns (Weights, objective value). Only psi is input.
pub fn em_weights(psi: &Psi) -> Result<(Weights, f64)> {
    let n = psi.matrix().nrows();
    let m = psi.matrix().ncols();
    let max_iter = 5000;
    // relative tolerance on improvement in objective
    let tol = 1e-30;
    // small floor to avoid division by zero and underflow
    let eps = 1e-30;
    let mut x = vec![1.0 / m as f64; m];
    let mut obj_prev = f64::NEG_INFINITY;
    let mut s = vec![0.0; n];
    let mut g = vec![0.0; m];
    for _ in 0..max_iter {
        // s = psi * x
        for i in 0..n {
            s[i] = 0.0;
            for j in 0..m {
                s[i] += psi.matrix().get(i, j) * x[j];
            }
            // ensure s is finite and not too small
            let si = s[i];
            if !si.is_finite() || si <= 0.0 {
                s[i] = eps;
            } else {
                s[i] = si.max(eps);
            }
        }
        // g = psi^T * (1.0 / s)
        for j in 0..m {
            g[j] = 0.0;
            for i in 0..n {
                g[j] += psi.matrix().get(i, j) / s[i];
            }
        }
        // x = x * g
        let mut x_sum = 0.0;
        for j in 0..m {
            x[j] *= g[j];
            x[j] = x[j].max(eps);
            x_sum += x[j];
        }
        // normalize x
        for j in 0..m {
            x[j] /= x_sum;
        }
        // objective
        // compute objective robustly; clamp very large or small logs
        let obj = s
            .iter()
            .map(|&si| {
                if si.is_finite() {
                    si.ln()
                } else {
                    (si.max(eps)).ln()
                }
            })
            .sum::<f64>();
        if (obj - obj_prev).abs() < tol * obj.abs().max(1.0) {
            return Ok((Weights::from_vec(x), obj));
        }
        obj_prev = obj;
    }
    let obj = s.iter().map(|&si| si.ln()).sum::<f64>();
    Ok((Weights::from_vec(x), obj))
}
