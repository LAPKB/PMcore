//! Projected Gradient Descent (PGD) optimizer for weights on the simplex
// Each step: gradient step, then project onto the simplex (Duchi et al. 2008)

use crate::structs::psi::Psi;
use crate::structs::weights::Weights;
use anyhow::Result;

fn project_simplex(v: &[f64]) -> Vec<f64> {
    // Robust projection of v onto the simplex {x: x >= 0, sum x = 1}.
    // Handle NaN/Inf specially: if any +Inf, return a unit vector at that index.
    if v.iter().any(|x| x.is_nan()) {
        // Replace NaNs with large negative so they won't be selected
        // This is just a fallback; NaNs shouldn't normally appear.
        let mut v2 = v.to_vec();
        for val in &mut v2 {
            if val.is_nan() {
                *val = -1e300;
            }
        }
        return project_simplex(&v2);
    }
    if let Some((idx, _)) = v
        .iter()
        .enumerate()
        .find(|(_, &x)| x.is_infinite() && x.is_sign_positive())
    {
        let mut res = vec![0.0; v.len()];
        res[idx] = 1.0;
        return res;
    }

    // Replace negative infinity with a very large negative number
    let mut u = v.to_vec();
    for vi in &mut u {
        if vi.is_infinite() && vi.is_sign_negative() {
            *vi = -1e300;
        }
    }

    // Standard Duchi projection
    u.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut cssv = 0.0;
    let mut rho = 0usize;
    for (i, &ui) in u.iter().enumerate() {
        cssv += ui;
        let t = (cssv - 1.0) / (i as f64 + 1.0);
        if ui - t > 0.0 {
            rho = i + 1;
        }
    }
    // If algo failed to find rho (shouldn't happen), fall back to largest index
    if rho == 0 {
        // fallback: put all mass on argmax of original v
        let (max_idx, _) = v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let mut res = vec![0.0; v.len()];
        res[max_idx] = 1.0;
        return res;
    }
    let sum_top: f64 = u.iter().take(rho).sum();
    let theta = (sum_top - 1.0) / (rho as f64);
    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

/// Projected Gradient Descent optimizer for weights on the simplex.
/// Returns (Weights, objective value). Only psi is input.
pub fn pgd_weights(psi: &Psi) -> Result<(Weights, f64)> {
    let psi_mat = psi.matrix();
    let n = psi_mat.nrows();
    let m = psi_mat.ncols();
    let max_iter = 1000;
    let tol = 1e-14;
    let eps = 1e-14;
    let mut x = vec![1.0 / m as f64; m];
    let mut obj_prev = f64::NEG_INFINITY;
    let mut s = vec![0.0; n];
    let mut g = vec![0.0; m];
    let mut step = 1.0;
    for _iter in 0..max_iter {
        // s = psi * x
        for i in 0..n {
            s[i] = 0.0;
            for j in 0..m {
                s[i] += psi_mat.get(i, j) * x[j];
            }
            s[i] = s[i].max(eps);
        }
        // g = psi^T * (1.0 / s)
        for j in 0..m {
            g[j] = 0.0;
            for i in 0..n {
                g[j] += psi_mat.get(i, j) / s[i];
            }
        }

        // Stabilize step: cap step so that max |step*(g-1)| <= max_update
        let max_update = 1e2_f64;
        let max_g_minus_one = g.iter().map(|&gj| (gj - 1.0).abs()).fold(0.0_f64, f64::max);
        if max_g_minus_one > 0.0 {
            let cap = max_update / max_g_minus_one;
            if step > cap {
                step = cap;
            }
        }
        // Projected gradient step: x = Proj(x + step * (g - 1))
        // Safe step: limit the update magnitude to avoid Inf/NaN
        let mut update = Vec::with_capacity(m);
        for (&xj, &gj) in x.iter().zip(g.iter()) {
            let val = xj + step * (gj - 1.0);
            if val.is_nan() {
                update.push(0.0);
            } else if val.is_infinite() {
                if val.is_sign_positive() {
                    update.push(1e300);
                } else {
                    update.push(-1e300);
                }
            } else {
                update.push(val);
            }
        }
        let x_new = project_simplex(&update);
        // Objective
        let obj = s.iter().map(|&si| si.ln()).sum::<f64>();
        if x.iter()
            .zip(x_new.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            < tol
            || (obj - obj_prev).abs() < tol * obj.abs().max(1.0)
        {
            return Ok((Weights::from_vec(x_new), obj));
        }
        obj_prev = obj;
        x = x_new;
        step *= 0.99;
    }
    let obj = s.iter().map(|&si| si.ln()).sum::<f64>();
    Ok((Weights::from_vec(x), obj))
}
