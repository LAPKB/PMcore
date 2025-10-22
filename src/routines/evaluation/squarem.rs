//! SQUAREM/Anderson acceleration for EM optimizer
// See docs/optimizers_for_weights.md for algorithm details.

use crate::structs::psi::Psi;
use crate::structs::weights::Weights;
use anyhow::Result;

/// SQUAREM acceleration for EM optimizer. Returns (Weights, objective value). Only psi is input.
pub fn squarem_weights(psi: &Psi) -> Result<(Weights, f64)> {
    let n = psi.matrix().nrows();
    let m = psi.matrix().ncols();
    let max_iter = 1000;
    let tol = 1e-8;
    let eps = 1e-12;
    let mut x = vec![1.0 / m as f64; m];
    let mut obj_prev = f64::NEG_INFINITY;
    for _ in 0..max_iter {
        // EM step 1
        let x1 = em_step(psi, &x, eps);
        // EM step 2
        let x2 = em_step(psi, &x1, eps);
        // v = x1 - x
        let v: Vec<f64> = x1.iter().zip(&x).map(|(&a, &b)| a - b).collect();
        // w = (x2 - x1) - v
        let w: Vec<f64> = x2
            .iter()
            .zip(&x1)
            .zip(&v)
            .map(|((&a, &b), &v)| (a - b) - v)
            .collect();
        // alpha = - (v^T v) / (v^T w)
        let v_dot_v = v.iter().map(|&vi| vi * vi).sum::<f64>();
        let v_dot_w = v.iter().zip(&w).map(|(&vi, &wi)| vi * wi).sum::<f64>();
        let alpha = if v_dot_w.abs() > 1e-12 {
            -v_dot_v / v_dot_w
        } else {
            1.0
        };
        // x_new = x + 2*alpha*v + alpha^2*w
        let mut x_new = vec![0.0; m];
        for j in 0..m {
            x_new[j] = x[j] + 2.0 * alpha * v[j] + alpha * alpha * w[j];
            x_new[j] = x_new[j].max(eps);
        }
        // normalize
        let sum_x = x_new.iter().sum::<f64>().max(eps);
        for j in 0..m {
            x_new[j] /= sum_x;
        }
        // EM safeguard: fallback to EM step if SQUAREM step is not feasible
        if x_new.iter().any(|&v| !v.is_finite() || v < 0.0) {
            x_new = x1;
        }
        // objective
        let s: Vec<f64> = (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| psi.matrix().get(i, j) * x_new[j])
                    .sum::<f64>()
                    .max(eps)
            })
            .collect();
        let obj = s.iter().map(|&si| si.ln()).sum::<f64>();
        if (obj - obj_prev).abs() < tol * obj.abs().max(1.0) {
            return Ok((Weights::from_vec(x_new), obj));
        }
        obj_prev = obj;
        x = x_new;
    }
    // Final objective
    let s: Vec<f64> = (0..n)
        .map(|i| {
            (0..m)
                .map(|j| psi.matrix().get(i, j) * x[j])
                .sum::<f64>()
                .max(eps)
        })
        .collect();
    let obj = s.iter().map(|&si| si.ln()).sum::<f64>();
    Ok((Weights::from_vec(x), obj))
}

fn em_step(psi: &Psi, x: &[f64], eps: f64) -> Vec<f64> {
    let n = psi.matrix().nrows();
    let m = psi.matrix().ncols();
    let mut s = vec![0.0; n];
    let mut g = vec![0.0; m];
    for i in 0..n {
        s[i] = 0.0;
        for j in 0..m {
            s[i] += psi.matrix().get(i, j) * x[j];
        }
        s[i] = s[i].max(eps);
    }
    for j in 0..m {
        g[j] = 0.0;
        for i in 0..n {
            g[j] += psi.matrix().get(i, j) / s[i];
        }
    }
    let mut x_new = vec![0.0; m];
    let mut x_sum = 0.0;
    for j in 0..m {
        x_new[j] = x[j] * g[j];
        x_new[j] = x_new[j].max(eps);
        x_sum += x_new[j];
    }
    for j in 0..m {
        x_new[j] /= x_sum;
    }
    x_new
}
