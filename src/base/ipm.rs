use ndarray::{ArrayBase, Dim, OwnedRepr, Array};
use ndarray_stats::{QuantileExt, DeviationExt};

pub fn burke(mut psi: ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>) -> Result<(),String>{
    psi.par_mapv_inplace(|x| x.abs());

    let (row,col) = psi.dim();

    if row>col {
        return Err("The matrix PSI has row>col".to_string());
    }

    if psi.min().unwrap() < &0.0 {
        return Err("PSI contains negative elements".to_string());
    }

    let ecol:ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>> = Array::ones((col,1));
    let mut plam = psi.dot(&ecol);
    
    if plam.min().unwrap() <= &1e-15 {
        return Err("The matrix PSI has row>col".to_string());
    }

    let eps = 1e-8;
    let sig = 0.;
    let erow:ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>  = Array::ones((row,1));
    let mut lam = ecol.clone();
    let w = 1./&plam;
    let ptw = psi.t().dot(&w);
    let shrink = 2.*ptw.max().unwrap().clone();
    lam = lam * shrink;
    plam = plam * shrink;
    let y = ecol - ptw;
    let R = erow - &w*&plam;
    let normR = norm_inf(R);
    //a.mapv(|x: f64| x.ln());
    let sum_log_plam = plam.mapv(|x:f64| x.ln()).sum();

    let gap = (w.mapv(|x:f64| x.ln()).sum() + &sum_log_plam).abs() / (1.+ &sum_log_plam);
    let mu = lam.t().dot(&y)/col as f64;
    

    Ok(())

}

fn norm_inf(a: ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>>) -> f64{
    let zeros:ArrayBase<OwnedRepr<f64>,Dim<[usize; 2]>> = Array::zeros((a.shape()[0],a.shape()[1]));
    a.linf_dist(&zeros).unwrap()
}

