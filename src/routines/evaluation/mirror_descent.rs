//! Entropic Mirror Descent optimizer for weights on the simplex
// See docs/optimizers_for_weights.md for algorithm details.

use crate::structs::psi::Psi;
use crate::structs::weights::Weights;
use anyhow::Result;

/// Entropic Mirror Descent optimizer for weights on the simplex.
/// Returns (Weights, objective value). Only psi is input.
pub fn mirror_descent_weights(psi: &Psi) -> Result<(Weights, f64)> {
    let n = psi.matrix().nrows();
    let m = psi.matrix().ncols();
    let max_iter = 5000;
    let tol = 1e-10;
    let eps = 1e-15;
    let mut x = vec![1.0 / m as f64; m];
    let mut obj_prev = f64::NEG_INFINITY;
    let mut s = vec![0.0; n];
    let mut g = vec![0.0; m];
    let mut step = 1.0;
    for _ in 0..max_iter {
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
        // Mirror descent update with backtracking line search to ensure objective improvement
        // placeholder for candidate sums handled locally
        // current objective
        let obj_current = s.iter().map(|&si| si.ln()).sum::<f64>();

        // Precompute max gradient deviation for clamping (not currently used but kept for future tuning)

        // Try candidate steps by backtracking to ensure increase in objective
        let mut candidate_step = step;
        let mut accepted = false;
        for _trial in 0..40 {
            // build candidate x (not yet normalized)
            let mut x_candidate = vec![0.0; m];
            let mut x_candidate_sum = 0.0;
            for j in 0..m {
                // clamp exponent argument to avoid overflow
                let arg = candidate_step * (g[j] - 1.0);
                let arg_clamped = arg.max(-50.0).min(50.0);
                let factor = arg_clamped.exp();
                let mut val = x[j] * factor;
                if !val.is_finite() || val <= 0.0 {
                    val = eps;
                }
                x_candidate[j] = val;
                x_candidate_sum += val;
            }
            // normalize
            for j in 0..m {
                x_candidate[j] /= x_candidate_sum;
            }

            // compute s_candidate and objective
            let mut s_candidate = vec![0.0; n];
            for i in 0..n {
                let mut si = 0.0;
                for j in 0..m {
                    si += psi.matrix().get(i, j) * x_candidate[j];
                }
                if !si.is_finite() || si <= 0.0 {
                    si = eps;
                }
                s_candidate[i] = si;
            }
            let obj_candidate = s_candidate.iter().map(|&si| si.ln()).sum::<f64>();

            // Accept candidate if objective improved (or not worse within small tolerance)
            if obj_candidate >= obj_current - 1e-14 {
                // accept
                x = x_candidate;
                // set s and obj accordingly
                s = s_candidate;
                accepted = true;
                break;
            }
            // otherwise reduce step
            candidate_step *= 0.5;
            if candidate_step < 1e-20 {
                break;
            }
        }
        if !accepted {
            // fallback: use clipped exponent with very small step
            let mut x_sum_fallback = 0.0;
            for j in 0..m {
                let arg = (step * 1e-6) * (g[j] - 1.0);
                let arg_clamped = arg.max(-50.0).min(50.0);
                let factor = arg_clamped.exp();
                x[j] *= factor;
                x[j] = x[j].max(eps);
                x_sum_fallback += x[j];
            }
            for j in 0..m {
                x[j] /= x_sum_fallback;
            }
            // recompute s if not accepted
            for i in 0..n {
                let mut si = 0.0;
                for j in 0..m {
                    si += psi.matrix().get(i, j) * x[j];
                }
                s[i] = si.max(eps);
            }
        }
        // Objective
        let obj = s.iter().map(|&si| si.ln()).sum::<f64>();
        if (obj - obj_prev).abs() < tol * obj.abs().max(1.0) {
            return Ok((Weights::from_vec(x), obj));
        }
        obj_prev = obj;
        // decrease step size slowly to encourage convergence
        step *= 0.995;
    }
    let obj = s.iter().map(|&si| si.ln()).sum::<f64>();
    Ok((Weights::from_vec(x), obj))
}
