/// L-BFGS optimizer with default arguments for benchmarking
pub fn lbfgs_weights_default(psi: &Psi) -> Result<(Weights, f64)> {
    lbfgs_weights(psi, 1000, 1e-8)
}
use crate::structs::psi::Psi;
use crate::structs::weights::Weights;
use anyhow::Result;
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

#[derive(Clone)]
struct SoftmaxOp {
    psi: Psi,
}

impl SoftmaxOp {
    fn new(psi: &Psi) -> Self {
        SoftmaxOp { psi: psi.clone() }
    }

    fn softmax(x: &[f64]) -> Vec<f64> {
        let max = x.iter().cloned().fold(std::f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }
}

impl CostFunction for SoftmaxOp {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        // p is y; compute x = softmax(y)
        let x = SoftmaxOp::softmax(p);
        let psi_mat = self.psi.matrix();
        let n_sub = psi_mat.nrows();
        let mut obj = 0.0;
        for i in 0..n_sub {
            let mut s = 0.0;
            for j in 0..x.len() {
                s += psi_mat.get(i, j) * x[j];
            }
            // guard
            let s = if s <= 0.0 { 1e-300 } else { s };
            obj += s.ln();
        }
        Ok(-obj)
    }
}

impl Gradient for SoftmaxOp {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let x = SoftmaxOp::softmax(p);
        let psi_mat = self.psi.matrix();
        let n_sub = psi_mat.nrows();
        let m = x.len();

        // compute s = psi * x
        let mut s_vec = vec![0.0; n_sub];
        for i in 0..n_sub {
            let mut sum = 0.0;
            for j in 0..m {
                sum += psi_mat.get(i, j) * x[j];
            }
            s_vec[i] = if sum <= 0.0 { 1e-300 } else { sum };
        }

        // compute g_x = psi^T * (1 ./ s)
        let mut g_x = vec![0.0; m];
        for j in 0..m {
            let mut sum = 0.0;
            for i in 0..n_sub {
                sum += psi_mat.get(i, j) / s_vec[i];
            }
            g_x[j] = sum;
        }

        // g_y = x .* (g_x - (x dot g_x))
        let x_dot_gx: f64 = x.iter().zip(g_x.iter()).map(|(a, b)| a * b).sum();
        let mut g_y = vec![0.0; m];
        for j in 0..m {
            g_y[j] = x[j] * (g_x[j] - x_dot_gx);
        }
        Ok(g_y.into_iter().map(|v| -v).collect())
    }
}

/// Optimize weights with L-BFGS on softmax parameters.
pub fn lbfgs_weights(psi: &Psi, max_iter: u64, tol: f64) -> Result<(Weights, f64)> {
    // Initialize y as log of uniform weights
    let m = psi.matrix().ncols();
    let y0 = vec![0.0f64; m];

    let op = SoftmaxOp::new(psi);
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 10).with_tolerance_grad(tol)?;

    let res = Executor::new(op, solver)
        .configure(|state| state.param(y0).max_iters(max_iter))
        .run()?;

    let best_y = res.state().best_param.as_ref().unwrap().clone();
    let x = SoftmaxOp::softmax(&best_y);
    // compute objective
    let mut obj = 0.0;
    for i in 0..psi.matrix().nrows() {
        let mut s = 0.0;
        for j in 0..m {
            s += psi.matrix().get(i, j) * x[j];
        }
        obj += s.ln();
    }

    Ok((Weights::from_vec(x), obj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routines::evaluation::ipm::burke;
    use faer::Mat;
    use std::time::Instant;

    #[test]
    fn timing_comparison_lbfgs_vs_burke_() {
        // sizes to test (n_sub, n_point)
        let sizes = [(10, 10), (100, 100), (500, 500), (1000, 1000)];
        println!("\nTiming comparison: lbfgs_weights vs burke\n");
        for &(n_sub, n_point) in &sizes {
            println!("\nMatrix size: n_sub = {}, n_point = {}", n_sub, n_point);
            let mat = Mat::from_fn(n_sub, n_point, |i, j| {
                if j == 0 {
                    10.0 + (n_sub as f64) * 0.1
                } else {
                    1.0 + 0.01 * (i as f64) + 0.005 * (j as f64)
                }
            });
            let psi = Psi::from(mat);

            let start_burke = Instant::now();
            let (_lam_burke, obj_burke) = burke(&psi).unwrap();
            let dur_burke = start_burke.elapsed();

            let start_lbfgs = Instant::now();
            let (lam_lbfgs, obj_lbfgs) = lbfgs_weights(&psi, 1000, 1e-8).unwrap();
            let dur_lbfgs = start_lbfgs.elapsed();

            println!(
                "burke: {:>10.2} ms | lbfgs: {:>10.2} ms",
                dur_burke.as_secs_f64() * 1000.0,
                dur_lbfgs.as_secs_f64() * 1000.0
            );

            // compare objectives
            println!("obj_burke: {}, obj_lbfgs: {}", obj_burke, obj_lbfgs);
            let rel_diff = (obj_lbfgs - obj_burke).abs() / (1.0 + obj_burke.abs());
            assert!(rel_diff < 1e-5, "LBFGS objective not close: {}", rel_diff);

            // verify simplex
            let sum_lbfgs: f64 = lam_lbfgs.iter().sum();
            assert!(
                (sum_lbfgs - 1.0).abs() < 1e-8,
                "LBFGS weights not normalized"
            );

            // Only comparing LBFGS vs Burke here (Burke is the gold standard)
        }
    }

    #[test]
    fn lbfgs_vs_burke_small() {
        let n_sub = 3;
        let n_point = 4;
        let mat = Mat::from_fn(n_sub, n_point, |i, j| {
            if j == 0 {
                10.0
            } else {
                1.0 + 0.1 * (i as f64)
            }
        });
        let psi = Psi::from(mat);

        let (_lam_burke, obj_burke) = burke(&psi).unwrap();
        let (lam_lbfgs, obj_lbfgs) = lbfgs_weights(&psi, 1000, 1e-8).unwrap();

        println!("obj_burke: {}, obj_lbfgs: {}", obj_burke, obj_lbfgs);
        assert!((obj_lbfgs - obj_burke).abs() / (1.0 + obj_burke.abs()) < 1e-5);
        // Check simplex property
        let sum: f64 = lam_lbfgs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }
}
