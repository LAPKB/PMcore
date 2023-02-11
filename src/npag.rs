use crate::prelude::*;

const THETA_E: f64 = 1e-4; //convergence Criteria

pub fn npag<S>(sim_eng: Engine<S>, ranges: Vec<(f64,f64)>, settings_path: String, seed: u32)
where
    S: Simulate
{
    let settings = settings::read(settings_path);
    let theta = lds::sobol(settings.config.init_points, ranges, seed);
    let scenarios = datafile::parse(settings.paths.data).unwrap();

    let mut eps  = 0.2;

    // while eps > THETA_E {
        // psi n_sub rows, nspp columns
        let psi = prob(&sim_eng, &scenarios, &theta);
        dbg!(ipm::burke(psi));
        // dbg!(psi);
        
    // }
}