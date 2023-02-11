use std::error;

use linfa_linalg::cholesky::Cholesky;
use ndarray::{ArrayBase, Dim, OwnedRepr, Array, Zip, Array2};
use ndarray_stats::{QuantileExt, DeviationExt};

pub fn burke(mut psi: ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>) -> Result<(),Box<dyn error::Error>>{
    psi.par_mapv_inplace(|x| x.abs());

    let (row,col) = psi.dim();

    if row>col {
        return Err("The matrix PSI has row>col".into());
    }

    if psi.min().unwrap() < &0.0 {
        return Err("PSI contains negative elements".into());
    }

    let ecol:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>> = Array::ones(col);
    let mut plam = psi.dot(&ecol);
    
    if plam.min().unwrap() <= &1e-15 {
        return Err("The matrix PSI has row>col".into());
    }

    let eps = 1e-8;
    let sig = 0.;
    let erow:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>>  = Array::ones(row);
    let mut lam = ecol.clone();
    let w = 1./&plam;
    let ptw = psi.t().dot(&w);
    let shrink = 2.*ptw.max().unwrap().clone();
    lam = lam * shrink;
    plam = plam * shrink;
    let y = ecol - ptw;
    let R = erow - &w*&plam;
    let normR = norm_inf(R);

    let sum_log_plam = plam.mapv(|x:f64| x.ln()).sum();

    let gap = (w.mapv(|x:f64| x.ln()).sum() + &sum_log_plam).abs() / (1.+ &sum_log_plam);
    let mu = lam.t().dot(&y)/col as f64;

    let mut iter: usize = 0;

    while mu > eps || normR > eps || gap > eps {
        iter = iter + 1;

        let smu = sig * mu;

        let mut inner:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>> = Array::zeros(col);
        Zip::from(&mut inner)
            .and(&lam)
            .and(&y)
            .for_each(|inner,lam,y|{
                *inner = lam/y;
            });
        let mut w_plam:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>> = Array::zeros(col);
        Zip::from(&mut w_plam)
            .and(&plam)
            .and(&w)
            .for_each(|w_plam,plam,w|{
                *w_plam = plam/w;
            });

        let h = 
            psi.dot(&Array2::from_diag(&inner))
                .dot(&psi.t()) + 
                Array2::from_diag(&w_plam);
        
        let uph = h.cholesky()?;


    }
    

    Ok(())

}

fn norm_inf(a: ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>>) -> f64{
    let zeros:ArrayBase<OwnedRepr<f64>,Dim<[usize; 1]>> = Array::zeros(a.len());
    a.linf_dist(&zeros).unwrap()
}

