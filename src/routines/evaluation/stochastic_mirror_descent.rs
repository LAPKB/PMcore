//! Stochastic Mirror Descent optimizer for weights on the simplex
// Uses minibatches of subjects for each update

use crate::structs::psi::Psi;
use crate::structs::weights::Weights;
use anyhow::Result;
use rand::rng;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

/// Stochastic Mirror Descent optimizer for weights on the simplex.
/// Returns (Weights, objective value). Only psi is input.
pub fn stochastic_mirror_descent_weights(psi: &Psi) -> Result<(Weights, f64)> {
    let n = psi.matrix().nrows();
    let m = psi.matrix().ncols();
    let batch_size = ((n as f64).sqrt().ceil() as usize).max(1);
    let max_iter = 1000;
    let tol = 1e-8;
    let eps = 1e-12;
    let mut x = vec![1.0 / m as f64; m]; // Initialize x
    let mut obj_prev = f64::NEG_INFINITY;
    let mut s = vec![0.0; n];
    let mut g = vec![0.0; m];
    let mut step = 1.0;
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng: ThreadRng = rng();
    for t in 0..max_iter {
        indices.shuffle(&mut rng);
        let batch = &indices[..batch_size.min(n)];
        // s = psi * x (only for batch)
        for &i in batch {
            s[i] = 0.0;
            for j in 0..m {
                s[i] += psi.matrix().get(i, j) * x[j];
            }
            s[i] = s[i].max(eps);
        }
        // g = psi^T * (1.0 / s) (only for batch)
        for j in 0..m {
            g[j] = 0.0;
            for &i in batch {
                g[j] += psi.matrix().get(i, j) / s[i];
            }
            g[j] *= (n as f64) / (batch.len() as f64); // scale up
        }
        // Mirror descent update: x_j <- x_j * exp(step * (g_j - 1))
        let mut x_sum = 0.0;
        for j in 0..m {
            x[j] *= (step * (g[j] - 1.0)).exp();
            x[j] = x[j].max(eps);
            x_sum += x[j];
        }
        for j in 0..m {
            x[j] /= x_sum;
        }
        // Objective (full data)
        for i in 0..n {
            s[i] = 0.0;
            for j in 0..m {
                s[i] += psi.matrix().get(i, j) * x[j];
            }
            s[i] = s[i].max(eps);
        }
        let obj = s.iter().map(|&si| si.ln()).sum::<f64>();
        if (obj - obj_prev).abs() < tol * obj.abs().max(1.0) {
            return Ok((Weights::from_vec(x), obj));
        }
        obj_prev = obj;
        step = 1.0 / ((t + 2) as f64).sqrt();
    }
    let obj = s.iter().map(|&si| si.ln()).sum::<f64>();
    Ok((Weights::from_vec(x), obj))
}
