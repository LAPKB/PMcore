use std::error;

use linfa_linalg::{cholesky::{Cholesky}, triangular::{SolveTriangular}};
use ndarray::{ArrayBase, Dim, OwnedRepr, Array, Array2, array};
use ndarray_stats::{QuantileExt, DeviationExt};
type OneDimArray = ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;


pub fn burke(psi: &ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>) -> Result<(OneDimArray, f64),Box<dyn error::Error>>{
    // psi.par_mapv_inplace(|x| x.abs());
    let (row,col) = psi.dim();
    // if row>col {
    //     return Err("The matrix PSI has row>col".into());
    // }
    if psi.min().unwrap() < &0.0 {
        return Err("PSI contains negative elements".into());
    }
    let ecol:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>> = Array::ones(col);
    let mut plam = psi.dot(&ecol);
    // if plam.min().unwrap() <= &1e-15 {
    //     return Err("The vector psi*e has a non-positive entry".into());
    // }
    let eps = 1e-8;
    let mut sig = 0.;
    let erow:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>>  = Array::ones(row);
    let mut lam = ecol.clone();
    let mut w = 1./&plam;
    let mut ptw = psi.t().dot(&w);
    let shrink = 2.**ptw.max().unwrap();
    lam *= shrink;
    plam *= shrink;
    w /= shrink;
    ptw /= shrink;
    let mut y = &ecol - &ptw;
    let mut r = &erow - &w*&plam;
    let mut norm_r = norm_inf(r);
    let sum_log_plam = plam.mapv(|x:f64| x.ln()).sum();
    let mut gap = (w.mapv(|x:f64| x.ln()).sum() + sum_log_plam).abs() / (1.+ sum_log_plam);
    let mut  mu = lam.t().dot(&y)/col as f64;


    while mu > eps || norm_r > eps || gap > eps {
        let smu = sig * mu;
        let inner = &lam/&y;//divide(&lam, &y);
        let w_plam = &plam/&w;//divide(&plam, &w);
        let h = 
            psi.dot(&Array2::from_diag(&inner))
                .dot(&psi.t()) + 
                Array2::from_diag(&w_plam);        
        let uph = h.cholesky()?;
        let uph = uph.t();
        let smuyinv = smu*(&ecol/&y);
        let rhsdw = &erow/&w - (psi.dot(&smuyinv));
        let a = rhsdw.clone().into_shape((rhsdw.len(),1))?;
        //todo: cleanup this aux variable
        // //dbg!(uph.t().is_triangular(linfa_linalg::triangular::UPLO::Upper));
        // uph.solve_into(rhsdw);
        let x = uph.t().solve_triangular(&a, linfa_linalg::triangular::UPLO::Lower)?;
        let dw_aux = uph.solve_triangular(&x, linfa_linalg::triangular::UPLO::Upper)?;
        let dw = dw_aux.column(0);
        let dy = - psi.t().dot(&dw);
        let dlam = smuyinv - &lam - inner * &dy;
        let mut alfpri = -1. / ((&dlam/&lam).min().unwrap().min(-0.5));
        alfpri = (0.99995*alfpri).min(1.0);
        let mut alfdual = -1. / ((&dy/&y).min().unwrap().min(-0.5));
        alfdual = alfdual.min(-1./(&dw/&w).min().unwrap().min(-0.5));
        alfdual = (0.99995*alfdual).min(1.0);
        lam = lam + alfpri*dlam;
        w = w + alfdual*&dw;
        y = y + alfdual*&dy;
        mu = lam.t().dot(&y)/col as f64;
        plam = psi.dot(&lam);
        r = &erow - &w*&plam;
        ptw = ptw -alfdual*dy;
        norm_r = norm_inf(r);
        let sum_log_plam = plam.mapv(|x:f64| x.ln()).sum();
        gap = (w.mapv(|x:f64| x.ln()).sum() + sum_log_plam).abs() / (1.+ sum_log_plam);
        if mu<eps && norm_r>eps {
            sig = 1.0;
        } else {
            sig =  array![[(1.-alfpri).powi(2),(1.-alfdual).powi(2),(norm_r-mu)/(norm_r+100.*mu)]].max().unwrap().min(0.3);
        }        
    }
    lam /= row as f64;
    let obj = psi.dot(&lam).mapv(|x| x.ln()).sum();
    lam = &lam/lam.sum();
    Ok((lam,obj))
}

fn norm_inf(a: ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>>) -> f64{
    let zeros:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>> = Array::zeros(a.len());
    a.linf_dist(&zeros).unwrap()
}

// fn divide(dividend: &ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>>, divisor: &ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>>) -> ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>>{
//     //check than dividend.len() == divisor.len()
//     //check than none of the elements of divisor == 0
//     //return a Result
//     let mut res:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>> = Array::zeros(dividend.len());
//     Zip::from(&mut res)
//         .and(dividend)
//         .and(divisor)
//         .for_each(|res,dividend,divisor|{
//             *res = dividend/divisor;
//         });
//     res
// }

