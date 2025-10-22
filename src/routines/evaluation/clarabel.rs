use crate::structs::psi::Psi;
use crate::structs::weights::Weights;
use anyhow::Result;
// Const parameters for Clarabel solver configuration
pub const CLARABEL_MAX_ITER: u32 = 1000;
pub const CLARABEL_TOL_FEAS: f64 = 1e-8;
pub const CLARABEL_TOL_GAP_REL: f64 = 1e-8;
pub const CLARABEL_EQUILIBRATE_ENABLE: bool = true;
/// Threshold for treating tiny negative weights as zero
pub const CLARABEL_NEG_TOL: f64 = 1e-9;

pub fn clarabel_weights(psi: &Psi) -> Result<(Weights, f64)> {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::*;

    // Extract dimensions
    let psi_mat = psi.matrix();
    let n_sub = psi_mat.nrows();
    let n_point = psi_mat.ncols();

    // Variable ordering: x (n_point), t (n_sub), u (n_sub)
    let nx = n_point;
    let nt = n_sub;
    let nu = n_sub;
    let nvar = nx + nt + nu;

    // Build P (zero) and q (objective: minimize sum(u))
    let p = CscMatrix::from(&vec![vec![0.0; nvar]; nvar]);
    let mut q = vec![0.0; nvar];
    for i in 0..nu {
        q[nx + nt + i] = 1.0; // u variables
    }

    // Build constraints rows
    let mut a_rows: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();

    use clarabel::solver::SupportedConeT;

    // 1) Exponential cones: for each i, 3 rows for (r=-u_i, s=1, t_i)
    for i in 0..n_sub {
        // row r: +1 at u_i
        let mut row_r = vec![0.0; nvar];
        row_r[nx + nt + i] = 1.0; // u_i
        a_rows.push(row_r);
        b.push(0.0);

        // row s: zeros, b = 1
        let row_s = vec![0.0; nvar];
        a_rows.push(row_s);
        b.push(1.0);

        // row t: -1 at t_i
        let mut row_t = vec![0.0; nvar];
        row_t[nx + i] = -1.0; // t_i
        a_rows.push(row_t);
        b.push(0.0);

        cones.push(SupportedConeT::ExponentialConeT());
    }

    // 2) t_i = sum_j psi_ij x_j  (n_sub rows, equality => ZeroCone)
    let mut eq_rows: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_sub {
        let mut row = vec![0.0; nvar];
        // x columns: negative psi row
        for j in 0..n_point {
            row[j] = -(*psi_mat.get(i, j));
        }
        // t_i column
        row[nx + i] = 1.0;
        eq_rows.push(row);
        b.push(0.0);
    }

    // 3) sum x = 1 (equality)
    let mut sum_row = vec![0.0; nvar];
    for j in 0..n_point {
        sum_row[j] = 1.0;
    }
    eq_rows.push(sum_row);
    b.push(1.0);

    // Append equality rows to A_rows and add a ZeroCone block
    for r in eq_rows.into_iter() {
        a_rows.push(r);
    }
    cones.push(SupportedConeT::ZeroConeT(n_sub + 1));

    // 4) Nonnegativity for x: -x <= 0 -> A rows: -I, b=0, Nonnegative cone
    let mut nonneg_rows: Vec<Vec<f64>> = Vec::new();
    for j in 0..n_point {
        let mut row = vec![0.0; nvar];
        row[j] = -1.0;
        nonneg_rows.push(row);
        b.push(0.0);
    }
    for r in nonneg_rows.into_iter() {
        a_rows.push(r);
    }
    cones.push(SupportedConeT::NonnegativeConeT(n_point));

    // Build A matrix from rows (dense assembly using CscMatrix::from)
    let a = CscMatrix::from(&a_rows);

    // Build settings using const parameters
    let settings = DefaultSettingsBuilder::default()
        .max_iter(CLARABEL_MAX_ITER)
        .tol_feas(CLARABEL_TOL_FEAS)
        .tol_gap_rel(CLARABEL_TOL_GAP_REL)
        .equilibrate_enable(CLARABEL_EQUILIBRATE_ENABLE)
        .verbose(false)
        .build()
        .unwrap();

    // Create solver and run
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings)?;
    solver.solve();

    // Extract solution x vector
    let sol = &solver.solution.x;
    if sol.len() != nvar {
        return Err(anyhow::anyhow!(
            "Clarabel returned unexpected solution size"
        ));
    }

    // Extract x variables
    let mut weights = Vec::with_capacity(n_point);
    for j in 0..n_point {
        weights.push(sol[j]);
    }

    // Normalize small negative entries using global threshold
    let neg_tol = CLARABEL_NEG_TOL;
    for w in weights.iter_mut() {
        if *w < 0.0 && *w > -neg_tol {
            *w = 0.0;
        }
    }

    // Ensure weights sum to 1 (numerical drift from solver may occur)
    let mut total: f64 = weights.iter().sum();
    if (total - 1.0).abs() > 1e-12 {
        if total <= 0.0 {
            return Err(anyhow::anyhow!(
                "Clarabel produced non-positive total weight"
            ));
        }
        for w in weights.iter_mut() {
            *w /= total;
        }
        // Recompute total (sanity)
        total = weights.iter().sum();
        debug_assert!((total - 1.0).abs() < 1e-12);
    }

    // Compute objective as sum ln(psi * lam) like Burke returns
    // Reconstruct lam as Weights and compute objective
    let lam_vec = weights.clone();

    // compute psi * lam
    let mut psi_dot: Vec<f64> = vec![0.0; n_sub];
    for i in 0..n_sub {
        let mut sum = 0.0;
        for j in 0..n_point {
            sum += psi_mat.get(i, j) * lam_vec[j];
        }
        psi_dot[i] = sum;
    }
    let obj = psi_dot.iter().map(|&v| v.ln()).sum();

    Ok((Weights::from_vec(weights), obj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routines::evaluation::ipm::burke;
    use crate::structs::psi::Psi;
    use faer::Mat;
    use std::time::Instant;
    #[test]
    fn timing_comparison_clarabel_vs_burke() {
        let sizes = [(10, 10), (100, 100), (500, 500), (1000, 1000)];
        println!("\nTiming comparison: clarabel_weights vs burke\n");
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
            let burke_result = burke(&psi);
            let dur_burke = start_burke.elapsed();

            let start_clarabel = Instant::now();
            let clarabel_result = clarabel_weights(&psi);
            let dur_clarabel = start_clarabel.elapsed();

            println!(
                "burke:    {:>10.2} ms {}",
                dur_burke.as_secs_f64() * 1000.0,
                if burke_result.is_ok() { "" } else { "(FAILED)" }
            );
            println!(
                "clarabel: {:>10.2} ms {}",
                dur_clarabel.as_secs_f64() * 1000.0,
                if clarabel_result.is_ok() {
                    ""
                } else {
                    "(FAILED)"
                }
            );
            if burke_result.is_ok() && clarabel_result.is_ok() && dur_burke.as_secs_f64() > 0.0 {
                println!(
                    "ratio clarabel/burke: {:.2}",
                    dur_clarabel.as_secs_f64() / dur_burke.as_secs_f64()
                );
            }
        }
    }
    #[test]
    fn clarabel_vs_burke_small() {
        // Small matrix test
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
        let (lam_clarabel, obj_clarabel) = clarabel_weights(&psi).unwrap();

        // Basic checks: sums to 1 and non-negativity (debug prints)
        let sum_clarabel: f64 = lam_clarabel.iter().sum();

        assert!(
            (sum_clarabel - 1.0).abs() < 1e-3,
            "Clarabel sum not ~1: {}",
            sum_clarabel
        );
        for v in lam_clarabel.iter() {
            assert!(v >= -1e-12);
        }

        // Compare objective values are similar
        assert!((obj_clarabel - obj_burke).abs() / (1.0 + obj_burke.abs()) < 1e-3);
    }

    #[test]
    fn clarabel_vs_burke_medium() {
        // Medium matrix test
        let n_sub = 10;
        let n_point = 20;
        let mat = Mat::from_fn(n_sub, n_point, |i, j| {
            if j == 0 {
                50.0
            } else {
                1.0 + 0.02 * (i as f64) + 0.01 * (j as f64)
            }
        });
        let psi = Psi::from(mat);

        let (_lam_burke, obj_burke) = burke(&psi).unwrap();
        let (lam_clarabel, obj_clarabel) = clarabel_weights(&psi).unwrap();

        // Basic checks
        let sum_clarabel: f64 = lam_clarabel.iter().sum();
        assert!(
            (sum_clarabel - 1.0).abs() < 1e-3,
            "Clarabel sum not ~1: {}",
            sum_clarabel
        );
        println!("Burke objective:   {}", obj_burke);
        println!("Clarabel objective: {}", obj_clarabel);

        // Compare objective values: allow slightly looser tolerance than small
        assert!((obj_clarabel - obj_burke).abs() / (1.0 + obj_burke.abs()) < 1e-3);
    }

    #[test]
    fn clarabel_vs_burke_large() {
        // Large-ish matrix test (keeps runtime reasonable)
        let n_sub = 30;
        let n_point = 80;
        let mat = Mat::from_fn(n_sub, n_point, |i, j| {
            if j == 0 {
                100.0
            } else {
                1.0 + 0.005 * (i as f64) + 0.002 * (j as f64)
            }
        });
        let psi = Psi::from(mat);

        let (_lam_burke, obj_burke) = burke(&psi).unwrap();
        let (lam_clarabel, obj_clarabel) = clarabel_weights(&psi).unwrap();

        // Basic checks
        let sum_clarabel: f64 = lam_clarabel.iter().sum();
        assert!(
            (sum_clarabel - 1.0).abs() < 1e-3,
            "Clarabel sum not ~1: {}",
            sum_clarabel
        );

        // Allow a little more tolerance for larger problems
        assert!((obj_clarabel - obj_burke).abs() / (1.0 + obj_burke.abs()) < 1e-3);
    }
}
