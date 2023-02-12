use std::error;

use linfa_linalg::{cholesky::{Cholesky, SolveC}, triangular::{SolveTriangular, Triangular}};
use ndarray::{ArrayBase, Dim, OwnedRepr, Array, Zip, Array2, Ix2};
use ndarray_stats::{QuantileExt, DeviationExt};


pub fn burke(mut psi: ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>) -> Result<(),Box<dyn error::Error>>{
    psi.par_mapv_inplace(|x| x.abs());
    // dbg!(&psi);

    let (row,col) = psi.dim();

    if row>col {
        return Err("The matrix PSI has row>col".into());
    }

    if psi.min().unwrap() < &0.0 {
        return Err("PSI contains negative elements".into());
    }

    let ecol:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>> = Array::ones(col);
    let mut plam = psi.dot(&ecol);
    dbg!(&plam);
    
    if plam.min().unwrap() <= &1e-15 {
        return Err("The vector psi*e has a non-positive entry".into());
    }

    let eps = 1e-8;
    let sig = 0.;
    let erow:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>>  = Array::ones(row);
    let mut lam = ecol.clone();
    let w = 1./&plam;
    // dbg!(&w);
    let ptw = psi.t().dot(&w);
    // dbg!(&ptw);
    let shrink = 2.*ptw.max().unwrap().clone();
    lam = lam * shrink;
    plam = plam * shrink;
    let y = &ecol - ptw;
    let R = &erow - &w*&plam;
    let normR = norm_inf(R);

    let sum_log_plam = plam.mapv(|x:f64| x.ln()).sum();

    let gap = (w.mapv(|x:f64| x.ln()).sum() + &sum_log_plam).abs() / (1.+ &sum_log_plam);
    let mu = lam.t().dot(&y)/col as f64;

    let mut iter: usize = 0;

    // while mu > eps || normR > eps || gap > eps {
        iter = iter + 1;

        let smu = sig * mu;

        let inner = &lam/&y;//divide(&lam, &y);
        let w_plam = &plam/&w;//divide(&plam, &w);


        let h = 
            psi.dot(&Array2::from_diag(&inner))
                .dot(&psi.t()) + 
                Array2::from_diag(&w_plam);
        
        let uph = h.cholesky()?;
        // dbg!(uph);
        let smuyinv = smu*(&ecol/&y);
        let rhsdw = &erow/&w - (psi.dot(&smuyinv));
        let a = rhsdw.clone().into_shape((rhsdw.len().clone(),1))?;
        //todo: cleanup this aux variable
        // dbg!(uph.t().is_triangular(linfa_linalg::triangular::UPLO::Upper));

        // uph.solve_into(rhsdw);
        // dbg!(&rhsdw);
        // dbg!(&a);
        // // dbg!(uph.solve_triangular(&a, linfa_linalg::triangular::UPLO::Lower));
        // // dbg!(uph.solvec(&a));
        let x = uph.t().solve_triangular(&a, linfa_linalg::triangular::UPLO::Upper)?;
        // dbg!(uph.t().dot(&x));
        let dw = uph.solve_triangular(&x, linfa_linalg::triangular::UPLO::Lower)?;
        // dbg!(dw);
        let dy = - psi.t().dot(&dw);
        dbg!(dy);

        
    // }
    
    Ok(())

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

