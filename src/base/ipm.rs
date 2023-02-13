use std::error;

use linfa_linalg::{cholesky::{Cholesky}, triangular::{SolveTriangular}};
use ndarray::{ArrayBase, Dim, OwnedRepr, Array, Array2, array};
use ndarray_stats::{QuantileExt, DeviationExt};


pub fn burke(psi: &ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>) -> Result<ArrayBase<OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,Box<dyn error::Error>>{
    // psi.par_mapv_inplace(|x| x.abs());
    // //dbg!(&psi);

    let (row,col) = psi.dim();
    //dbg!((row,col));

    // if row>col {
    //     return Err("The matrix PSI has row>col".into());
    // }

    if psi.min().unwrap() < &0.0 {
        return Err("PSI contains negative elements".into());
    }

    let ecol:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>> = Array::ones(col);
    let mut plam = psi.dot(&ecol);
    // //dbg!(&plam);
    
    if plam.min().unwrap() <= &1e-15 {
        return Err("The vector psi*e has a non-positive entry".into());
    }

    let eps = 1e-8;
    let mut sig = 0.;
    // //dbg!(eps);
    let erow:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>>  = Array::ones(row);
    let mut lam = ecol.clone();
    // //dbg!(lam);
    let mut w = 1./&plam;
    // //dbg!(&w);
    let mut ptw = psi.t().dot(&w);
    // //dbg!(&ptw);
    let shrink = 2.*ptw.max().unwrap().clone();
    lam = lam * shrink;
    plam = plam * shrink;
    w = w/shrink;
    ptw = ptw/shrink;
    let mut y = &ecol - &ptw;
    let mut r = &erow - &w*&plam;
    let mut norm_r = norm_inf(r);

    let sum_log_plam = plam.mapv(|x:f64| x.ln()).sum();

    let mut gap = (w.mapv(|x:f64| x.ln()).sum() + &sum_log_plam).abs() / (1.+ &sum_log_plam);
    let mut  mu = lam.t().dot(&y)/col as f64;

    let mut iter: usize = 0;

    while mu > eps || norm_r > eps || gap > eps {
        iter = iter + 1;
        // dbg!(iter);
        // dbg!(mu);
        // dbg!(gap);
        // dbg!(norm_r);

        let smu = sig * mu;
        //dbg!(&smu);

        let inner = &lam/&y;//divide(&lam, &y);
        //dbg!(&inner);
        let w_plam = &plam/&w;//divide(&plam, &w);
        //dbg!(&w_plam);


        let h = 
            psi.dot(&Array2::from_diag(&inner))
                .dot(&psi.t()) + 
                Array2::from_diag(&w_plam);
        //dbg!(&h);
        
        let uph = h.cholesky()?;
        let uph = uph.t();
        //dbg!(&uph);
        let smuyinv = smu*(&ecol/&y);
        //dbg!(&smuyinv);
        let rhsdw = &erow/&w - (psi.dot(&smuyinv));
        //dbg!(&rhsdw);
        let a = rhsdw.clone().into_shape((rhsdw.len().clone(),1))?;
        //todo: cleanup this aux variable
        // //dbg!(uph.t().is_triangular(linfa_linalg::triangular::UPLO::Upper));

        // uph.solve_into(rhsdw);
        // //dbg!(&rhsdw);
        // //dbg!(&a);
        // // //dbg!(uph.solve_triangular(&a, linfa_linalg::triangular::UPLO::Lower));
        // // //dbg!(uph.solvec(&a));
        let x = uph.t().solve_triangular(&a, linfa_linalg::triangular::UPLO::Lower)?;
        //dbg!(&x);
        // //dbg!(uph.t().dot(&x));
        let dw_aux = uph.solve_triangular(&x, linfa_linalg::triangular::UPLO::Upper)?;
        let dw = dw_aux.column(0);
        //dbg!(&dw);
        // //dbg!(&dw);
        let dy = - psi.t().dot(&dw);
        //dbg!(&dy);
        // //dbg!(&dy);

        let dlam = smuyinv - &lam - inner * &dy;
        //dbg!(&dlam);
        // //dbg!(dlam);
        let mut alfpri = -1. / ((&dlam/&lam).min().unwrap().min(-0.5));
        alfpri = (0.99995*alfpri).min(1.0);
        let mut alfdual = -1. / ((&dy/&y).min().unwrap().min(-0.5));
        alfdual = alfdual.min(-1./(&dw/&w).min().unwrap().min(-0.5));
        alfdual = (0.99995*alfdual).min(1.0);
        //dbg!(&alfpri);
        //dbg!(&alfdual);

        lam = lam + alfpri*dlam;
        //dbg!(&lam);
        w = w + alfdual*&dw;
        //dbg!(&w);
        y = y + alfdual*&dy;
        //dbg!(&y);

        mu = lam.t().dot(&y)/col as f64;
        //dbg!(&mu);
        plam = psi.dot(&lam);
        //dbg!(&plam);
        r = &erow - &w*&plam;
        //dbg!(&r);
        ptw = ptw -alfdual*dy;
        //dbg!(&ptw);
        norm_r = norm_inf(r);
        //dbg!(&norm_r);
        let sum_log_plam = plam.mapv(|x:f64| x.ln()).sum();
        gap = (w.mapv(|x:f64| x.ln()).sum() + &sum_log_plam).abs() / (1.+ &sum_log_plam);
        //dbg!(&gap);

        if mu<eps && norm_r>eps {
            sig = 1.0;
        } else {
            sig =  array![[(1.-alfpri).powi(2),(1.-alfdual).powi(2),(norm_r-mu)/(norm_r+100.*mu)]].max().unwrap().min(0.3);
        }
        //dbg!(&sig);
        
    }
    lam = lam/row as f64;
    // let obj = psi.dot(&lam).mapv(|x| x.ln()).sum();
    lam = &lam/lam.sum();
    // dbg!(lam);
    // dbg!(obj);
    
    Ok(lam)

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

