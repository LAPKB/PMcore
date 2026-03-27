use crate::estimation::nonparametric::{Psi, Weights};
use anyhow::bail;
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::{Col, Mat, Row};
use rayon::prelude::*;

/// Applies Burke's Interior Point Method (IPM) to solve a convex optimization problem.
pub fn burke(psi: &Psi) -> anyhow::Result<(Weights, f64)> {
    let mut psi = psi.matrix().to_owned();

    psi.row_iter_mut().try_for_each(|row| {
        row.iter_mut().try_for_each(|x| {
            if !x.is_finite() {
                bail!("Input matrix must have finite entries")
            } else {
                *x = x.abs();
                Ok(())
            }
        })
    })?;

    let (n_sub, n_point) = psi.shape();
    let ecol: Col<f64> = Col::from_fn(n_point, |_| 1.0);
    let erow: Row<f64> = Row::from_fn(n_sub, |_| 1.0);
    let mut plam: Col<f64> = &psi * &ecol;
    let eps: f64 = 1e-8;
    let mut sig: f64 = 0.0;
    let mut lam = ecol.clone();
    let mut w: Col<f64> = Col::from_fn(plam.nrows(), |i| 1.0 / plam.get(i));
    let mut ptw: Col<f64> = psi.transpose() * &w;

    let ptw_max = ptw.iter().fold(f64::NEG_INFINITY, |acc, &x| x.max(acc));
    let shrink = 2.0 * ptw_max;
    lam *= shrink;
    plam *= shrink;
    w /= shrink;
    ptw /= shrink;

    let mut y: Col<f64> = &ecol - &ptw;
    let mut r: Col<f64> = Col::from_fn(n_sub, |i| erow.get(i) - w.get(i) * plam.get(i));
    let mut norm_r: f64 = r.iter().fold(0.0, |max, &val| max.max(val.abs()));
    let sum_log_plam: f64 = plam.iter().map(|x| x.ln()).sum();
    let sum_log_w: f64 = w.iter().map(|x| x.ln()).sum();
    let mut gap: f64 = (sum_log_w + sum_log_plam).abs() / (1.0 + sum_log_plam);
    let mut mu = lam.transpose() * &y / n_point as f64;

    let mut psi_inner: Mat<f64> = Mat::zeros(psi.nrows(), psi.ncols());
    let n_threads = faer::get_global_parallelism().degree();
    let rows = psi.nrows();
    let mut output: Vec<Mat<f64>> = (0..n_threads).map(|_| Mat::zeros(rows, rows)).collect();
    let mut h: Mat<f64> = Mat::zeros(rows, rows);

    while mu > eps || norm_r > eps || gap > eps {
        let smu = sig * mu;
        let inner = Col::from_fn(lam.nrows(), |i| lam.get(i) / y.get(i));
        let w_plam = Col::from_fn(plam.nrows(), |i| plam.get(i) / w.get(i));

        if psi.ncols() > n_threads * 128 {
            psi_inner
                .par_col_partition_mut(n_threads)
                .zip(psi.par_col_partition(n_threads))
                .zip(inner.par_partition(n_threads))
                .zip(output.par_iter_mut())
                .for_each(|(((mut psi_inner, psi), inner), output)| {
                    psi_inner
                        .as_mut()
                        .col_iter_mut()
                        .zip(psi.col_iter())
                        .zip(inner.iter())
                        .for_each(|((col, psi_col), inner_val)| {
                            col.iter_mut().zip(psi_col.iter()).for_each(|(x, psi_val)| {
                                *x = psi_val * inner_val;
                            });
                        });
                    faer::linalg::matmul::triangular::matmul(
                        output.as_mut(),
                        faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                        faer::Accum::Replace,
                        &psi_inner,
                        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                        psi.transpose(),
                        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                        1.0,
                        faer::Par::Seq,
                    );
                });

            let mut first_iter = true;
            for output in &output {
                if first_iter {
                    h.copy_from(output);
                    first_iter = false;
                } else {
                    h += output;
                }
            }
        } else {
            psi_inner
                .as_mut()
                .col_iter_mut()
                .zip(psi.col_iter())
                .zip(inner.iter())
                .for_each(|((col, psi_col), inner_val)| {
                    col.iter_mut().zip(psi_col.iter()).for_each(|(x, psi_val)| {
                        *x = psi_val * inner_val;
                    });
                });
            faer::linalg::matmul::triangular::matmul(
                h.as_mut(),
                faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                faer::Accum::Replace,
                &psi_inner,
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                psi.transpose(),
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                1.0,
                faer::Par::Seq,
            );
        }

        for i in 0..h.nrows() {
            h[(i, i)] += w_plam[i];
        }

        let uph = match h.llt(faer::Side::Lower) {
            Ok(llt) => llt,
            Err(_) => {
                bail!("Error during Cholesky decomposition. The matrix might not be positive definite. This is usually due to model misspecification or numerical issues.")
            }
        };
        let uph = uph.L().transpose().to_owned();

        let smuyinv: Col<f64> = Col::from_fn(ecol.nrows(), |i| smu * (ecol[i] / y[i]));
        let psi_dot_muyinv: Col<f64> = &psi * &smuyinv;
        let rhsdw: Row<f64> = Row::from_fn(erow.ncols(), |i| erow[i] / w[i] - psi_dot_muyinv[i]);
        let mut dw = Mat::from_fn(rhsdw.ncols(), 1, |i, _j| *rhsdw.get(i));

        solve_lower_triangular_in_place(uph.transpose().as_ref(), dw.as_mut(), faer::Par::rayon(0));
        solve_upper_triangular_in_place(uph.as_ref(), dw.as_mut(), faer::Par::rayon(0));

        let dw = dw.col(0);
        let dy = -(psi.transpose() * dw);
        let inner_times_dy = Col::from_fn(ecol.nrows(), |i| inner[i] * dy[i]);
        let dlam: Row<f64> = Row::from_fn(ecol.nrows(), |i| smuyinv[i] - lam[i] - inner_times_dy[i]);

        let ratio_dlam_lam = Row::from_fn(lam.nrows(), |i| dlam[i] / lam[i]);
        let min_ratio_dlam = ratio_dlam_lam.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfpri: f64 = -1.0 / min_ratio_dlam.min(-0.5);
        alfpri = (0.99995 * alfpri).min(1.0);

        let ratio_dy_y = Row::from_fn(y.nrows(), |i| dy[i] / y[i]);
        let min_ratio_dy = ratio_dy_y.iter().cloned().fold(f64::INFINITY, f64::min);
        let ratio_dw_w = Row::from_fn(dw.nrows(), |i| dw[i] / w[i]);
        let min_ratio_dw = ratio_dw_w.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut alfdual = -1.0 / min_ratio_dy.min(-0.5);
        alfdual = alfdual.min(-1.0 / min_ratio_dw.min(-0.5));
        alfdual = (0.99995 * alfdual).min(1.0);

        lam += alfpri * dlam.transpose();
        w += alfdual * dw;
        y += alfdual * &dy;

        mu = lam.transpose() * &y / n_point as f64;
        plam = &psi * &lam;
        r = Col::from_fn(n_sub, |i| erow.get(i) - w.get(i) * plam.get(i));
        ptw -= alfdual * dy;

        norm_r = r.norm_max();
        let sum_log_plam: f64 = plam.iter().map(|x| x.ln()).sum();
        let sum_log_w: f64 = w.iter().map(|x| x.ln()).sum();
        gap = (sum_log_w + sum_log_plam).abs() / (1.0 + sum_log_plam);

        if mu < eps && norm_r > eps {
            sig = 1.0;
        } else {
            let candidate1 = (1.0 - alfpri).powi(2);
            let candidate2 = (1.0 - alfdual).powi(2);
            let candidate3 = (norm_r - mu) / (norm_r + 100.0 * mu);
            sig = candidate1.max(candidate2).max(candidate3).min(0.3);
        }
    }

    lam /= n_sub as f64;
    let obj = (psi * &lam).iter().map(|x| x.ln()).sum();
    let lam_sum: f64 = lam.iter().sum();
    lam = &lam / lam_sum;

    Ok((lam.into(), obj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    #[test]
    fn test_burke_identity() {
        let n = 100;
        let mat = Mat::identity(n, n);
        let psi = Psi::from(mat);
        let (lam, _) = burke(&psi).unwrap();

        let expected = 1.0 / n as f64;
        for i in 0..n {
            assert_relative_eq!(lam[i], expected, epsilon = 1e-10);
        }
        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_burke_uniform_square() {
        let n_sub = 10;
        let n_point = 10;
        let mat = Mat::from_fn(n_sub, n_point, |_, _| 1.0);
        let psi = Psi::from(mat);
        let (lam, _) = burke(&psi).unwrap();

        assert_relative_eq!(lam.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
        let expected = 1.0 / n_point as f64;
        for i in 0..n_point {
            assert_relative_eq!(lam[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_burke_with_non_finite_values() {
        let n_sub = 10;
        let n_point = 10;
        let mat = Mat::from_fn(n_sub, n_point, |i, j| if i == 0 && j == 0 { f64::NAN } else { 1.0 });
        let psi = Psi::from(mat);
        assert!(burke(&psi).is_err());
    }
}