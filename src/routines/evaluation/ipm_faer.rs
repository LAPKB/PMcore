use faer::{solvers::Solver, Faer, IntoFaer, IntoNdarray};
use faer_core::{inner::DenseOwn, mat, scale, unzipped, zipped, Mat, Matrix};
use std::error;

use ndarray::{ArrayBase, Dim, OwnedRepr};
use ndarray_stats::QuantileExt;
type OneDimArray = ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;

pub fn burke(
    psi: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> Result<(OneDimArray, f64), Box<dyn error::Error>> {
    let (row, col) = psi.dim();
    if psi.min().unwrap() < &0.0 {
        return Err("PSI contains negative elements".into());
    }
    let psi: Matrix<DenseOwn<f64>> = psi.view().into_faer().to_owned();
    let ecol = Mat::from_fn(col, 1, |_, _| 1.0);
    let mut plam = &psi * &ecol; //row x 1
    let eps = 1e-8;
    let mut sig = 0.0;
    let erow = Mat::from_fn(row, 1, |_, _| 1.0);
    let mut lam = ecol.clone();
    let mut w = plam.clone();
    //This loop can be replaced to a zip call
    w = zipped!(w.as_ref()).map(|unzipped!(w_i)| 1.0 / *w_i);
    // for i in 0..w.nrows() {
    //     // for j in 0..w.ncols() {
    //     // w is row x 1, this for loop should not be needed
    //     let mut w_aux = w.get_mut(i, 0);
    //     *w_aux = 1.0 / *w_aux;
    //     // }
    // }

    let psi_t = psi.transpose();
    let mut ptw: Matrix<DenseOwn<f64>> = psi_t * &w;
    // dbg!(&ptw);
    let mut max_ptw = f64::NEG_INFINITY;
    for i in 0..ptw.nrows() {
        if *ptw.get(i, 0) > max_ptw {
            max_ptw = *ptw.get(i, 0)
        }
    }
    let shrink = 2.0 * max_ptw;
    lam = scale(shrink) * lam;
    plam = scale(shrink) * plam;
    // w = scale(1.0 / shrink) * w;
    // ptw = scale(1.0 / shrink) * ptw;
    w = zipped!(w.as_ref()).map(|unzipped!(w_i)| *w_i / shrink);
    ptw = zipped!(ptw.as_ref()).map(|unzipped!(ptw_i)| *ptw_i / shrink);
    // dbg!(&w);
    // dbg!(&plam);
    // dbg!(&erow);
    let mut y = &ecol - &ptw;
    let mut r =
        &erow - zipped!(w.as_ref(), plam.as_ref()).map(|unzipped!(w_i, plam_i)| *w_i * *plam_i);
    let mut norm_r = r.norm_max();
    // dbg!(&y);
    // dbg!(&r);
    // dbg!(&norm_r);
    let mut sum_log_plam = 0.0;
    let mut sum_log_w = 0.0;
    // dbg!(w.ncols());
    // dbg!(w.nrows());
    // dbg!(&w);
    for i in 0..plam.nrows() {
        sum_log_plam += plam.get(i, 0).ln();
    }
    for i in 0..w.nrows() {
        sum_log_w += w.get(i, 0).ln();
    }
    let mut gap = (sum_log_w + sum_log_plam).abs() / (1. + sum_log_plam);
    // dbg!(sum_log_w);
    // dbg!(gap);
    let mut mu = (lam.transpose() * &y).get(0, 0) / col as f64;

    // dbg!(&mu);
    while mu > eps || norm_r > eps || gap > eps {
        let smu = sig * mu;
        let inner = zipped!(lam.as_ref(), y.as_ref()).map(|unzipped!(lam_i, y_i)| *lam_i / *y_i);
        // dbg!(&inner);
        let w_plam =
            zipped!(plam.as_ref(), w.as_ref()).map(|unzipped!(plam_i, w_i)| *plam_i / *w_i);
        // dbg!(&w_plam);
        let h: Matrix<DenseOwn<f64>> =
            (&psi * inner.as_ref().col(0).column_vector_as_diagonal()) * &psi.transpose();
        let mut aux: Matrix<DenseOwn<f64>> = Mat::zeros(row, row);
        for i in 0..row {
            let diag = aux.get_mut(i, i);
            *diag = *w_plam.get(i, 0);
        }
        let h = h + aux;
        // dbg!(&h);
        let uph = h.cholesky(faer_core::Side::Lower).unwrap();
        let smuyinv = scale(smu)
            * zipped!(ecol.as_ref(), y.as_ref()).map(|unzipped!(ecol_i, y_i)| *ecol_i / *y_i);
        let rhsdw = zipped!(erow.as_ref(), w.as_ref()).map(|unzipped!(erow_i, w_i)| *erow_i / *w_i)
            - &psi * &smuyinv;
        // dbg!(&rhsdw);
        let dw = uph.solve(&rhsdw);

        let dy = -psi.transpose() * &dw;

        let dlam = smuyinv
            - &lam
            - zipped!(inner.as_ref(), dy.as_ref()).map(|unzipped!(inner_i, dy_i)| *inner_i * *dy_i);
        let mut alfpri = -1.
            / (min(zipped!(dlam.as_ref(), lam.as_ref())
                .map(|unzipped!(dlam_i, lam_i)| *dlam_i / *lam_i))
            .min(-0.5));
        alfpri = (0.99995 * alfpri).min(1.0);
        let mut alfdual = -1.
            / (min(zipped!(dy.as_ref(), y.as_ref()).map(|unzipped!(dy_i, y_i)| *dy_i / *y_i))
                .min(-0.5));
        alfdual = alfdual.min(
            -1. / (min(zipped!(dw.as_ref(), w.as_ref()).map(|unzipped!(dw_i, w_i)| *dw_i / *w_i))
                .min(-0.5)),
        );
        alfdual = (0.99995 * alfdual).min(1.0);
        lam = lam + scale(alfpri) * dlam;
        w = w + scale(alfdual) * &dw;
        y = y + scale(alfdual) * &dy;
        mu = (lam.transpose() * &y).get(0, 0) / col as f64;
        plam = &psi * &lam;
        r = &erow - zipped!(w.as_ref(), plam.as_ref()).map(|unzipped!(w_i, plam_i)| *w_i * *plam_i);
        ptw = ptw - scale(alfdual) * dy;
        norm_r = r.norm_max();
        let mut sum_log_plam = 0.0;
        let mut sum_log_w = 0.0;

        for i in 0..plam.nrows() {
            sum_log_plam += plam.get(i, 0).ln();
        }
        for i in 0..w.nrows() {
            sum_log_w += w.get(i, 0).ln();
        }
        gap = (sum_log_w + sum_log_plam).abs() / (1. + sum_log_plam);
        if mu < eps && norm_r > eps {
            sig = 1.0;
        } else {
            sig = max(mat![[
                (1. - alfpri).powi(2),
                (1. - alfdual).powi(2),
                (norm_r - mu) / (norm_r + 100. * mu)
            ]])
            .min(0.3);
        }
    }
    lam = scale(1. / row as f64) * lam;
    let aux = psi * &lam;
    let mut obj = 0.0;

    for i in 0..aux.nrows() {
        obj += aux.get(i, 0).ln();
    }
    Ok((lam.as_ref().into_ndarray().column(0).to_owned(), obj))
    // matrix_ops::Ok(())
}
fn min(mat: Matrix<DenseOwn<f64>>) -> f64 {
    let mut min = f64::INFINITY;
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            if *mat.get(i, j) < min {
                min = *mat.get(i, j)
            }
        }
    }
    min
}
fn max(mat: Matrix<DenseOwn<f64>>) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            if *mat.get(i, j) > max {
                max = *mat.get(i, j)
            }
        }
    }
    max
}