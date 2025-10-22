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
    let max_iter = 10000;
    let tol = 1e-10;
    let eps = 1e-12;
    let mut x = vec![1.0 / m as f64; m];
    let mut obj_prev = f64::NEG_INFINITY;
    let mut s = vec![0.0; n];
    let mut g = vec![0.0; m];
    // convenience reference to matrix
    let psi_mat = psi.matrix();
    for t in 0..max_iter {
        // s = psi * x
        for i in 0..n {
            let mut si = 0.0;
            for j in 0..m {
                si += psi_mat.get(i, j) * x[j];
            }
            s[i] = si.max(eps);
        }
        // g = psi^T * (1.0 / s)
        for j in 0..m {
            let mut gj = 0.0;
            for i in 0..n {
                gj += psi_mat.get(i, j) / s[i];
            }
            g[j] = gj;
        }
        // Linear maximization oracle (since objective is concave): pick best coordinate
        let max_idx = g
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // Frank-Wolfe vertex (unit vector)
        let mut s_fw = vec![0.0; m];
        s_fw[max_idx] = 1.0;

        // Compute duality gap: s_fw^T g - x^T g
        let mut dot_x_g = 0.0;
        for j in 0..m {
            dot_x_g += x[j] * g[j];
        }
        let gap = g[max_idx] - dot_x_g;

        // Stop if gap small
        let gap_tol = 1e-8;
        if gap <= gap_tol {
            let obj = s.iter().map(|&si: &f64| si.ln()).sum::<f64>();
            return Ok((Weights::from_vec(x), obj));
        }

        // Prepare the column corresponding to s_fw (psi * e_max)
        let mut s_vertex = vec![0.0; n];
        for i in 0..n {
            s_vertex[i] = psi_mat.get(i, max_idx).max(eps);
        }

        // Exact line search along direction: maximize f(gamma) = sum ln((1-gamma)*s + gamma*s_vertex)
        // Solve f'(gamma) = 0 using Newton's method with safeguards
        let mut gamma = 2.0 / ((t + 2) as f64); // warm start
        gamma = gamma.max(0.0).min(1.0);
        for _li in 0..30 {
            let mut f1 = 0.0;
            let mut f2 = 0.0;
            let mut denom_ok = true;
            for i in 0..n {
                let denom = (1.0 - gamma) * s[i] + gamma * s_vertex[i];
                if denom <= 0.0 || !denom.is_finite() {
                    denom_ok = false;
                    break;
                }
                let delta = s_vertex[i] - s[i];
                f1 += delta / denom;
                f2 -= (delta * delta) / (denom * denom);
            }
            if !denom_ok {
                gamma = gamma.max(0.0).min(1.0);
                break;
            }
            if f1.abs() < 1e-14 {
                break;
            }
            if f2 == 0.0 {
                break;
            }
            // Newton update for maximization: gamma <- gamma - f'(gamma)/f''(gamma)
            let step = f1 / f2;
            let mut gamma_new = gamma - step;
            // clamp into [0,1]
            if gamma_new.is_nan() {
                break;
            }
            gamma_new = gamma_new.max(0.0).min(1.0);
            // If no change, stop
            if (gamma_new - gamma).abs() < 1e-14 {
                gamma = gamma_new;
                break;
            }
            // Accept only if objective increases; otherwise damp the step
            // compute objective at gamma and gamma_new
            let mut obj_curr = 0.0;
            let mut obj_new = 0.0;
            for i in 0..n {
                let denom_c = (1.0 - gamma) * s[i] + gamma * s_vertex[i];
                let denom_n = (1.0 - gamma_new) * s[i] + gamma_new * s_vertex[i];
                obj_curr += denom_c.ln();
                obj_new += denom_n.ln();
            }
            if obj_new >= obj_curr {
                gamma = gamma_new;
            } else {
                // damp: move halfway toward gamma_new
                gamma = (gamma + gamma_new) / 2.0;
            }
            // tiny change -> stop
            if (obj_new - obj_curr).abs() < 1e-14 {
                break;
            }
        }

        // Update x with the selected step size
        for j in 0..m {
            x[j] = (1.0 - gamma) * x[j] + gamma * s_fw[j];
        }
        // Update s to the new psi*x without recomputing full matmul: convex combination
        for i in 0..n {
            s[i] = (1.0 - gamma) * s[i] + gamma * s_vertex[i];
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
