//! Ratio estimator bias calibration: measure grid peak vs particle count M.
//!
//! Sweeps M ∈ {20, 50, 100, 200, 500, 1000} on the synthetic dataset.
//! For each M: runs a 20-point surface sweep (3 resamples per point),
//! finds the peak via quadratic interpolation, and records the bias.
//!
//! Output: calibration.csv with columns: particles,grid_peak,true_ske,bias

use pmcore::prelude::*;

fn main() -> Result<()> {
    let data = data::read_pmetrics("examples/iov_synthetic/data.csv")?;
    let error_models = AssayErrorModels::new().add(
        "outeq_1",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.2, 0.0, 0.0), 0.0),
    )?;

    let ke_val = 0.2934;
    let v_val = 10.0;
    let true_ske = 0.08;
    let sweep_points = 25;
    let resamples = 5; // average 5 particle-filter runs per grid point

    let particle_counts = [20, 50, 100, 200, 500, 1000, 2000];
    let mut csv = String::from("particles,grid_peak,true_ske,bias\n");

    println!("particles  grid_peak  bias      bias_pct");
    println!("---------  ---------  --------  --------");

    for &n_p in &particle_counts {
        let sde = make_sde(n_p);

        let mut best_ske = 0.0;
        let mut best_ll = f64::NEG_INFINITY;

        for i in 0..sweep_points {
            let ske_val = 0.002 + 0.25 * (i as f64 / (sweep_points - 1) as f64);
            let params = vec![ke_val, v_val, ske_val];

            let mut total_ll = 0.0;
            for _ in 0..resamples {
                let mut ll_sum = 0.0;
                for subject in data.subjects() {
                    if let (_, Some(ll)) =
                        sde.simulate_subject_dense(subject, &params, Some(&error_models))?
                    {
                        if ll > 0.0 {
                            ll_sum += ll.ln();
                        }
                    }
                }
                total_ll += ll_sum;
            }
            let mean_ll = total_ll / resamples as f64;

            if mean_ll > best_ll {
                best_ll = mean_ll;
                best_ske = ske_val;
            }
        }

        let bias = best_ske - true_ske;
        let bias_pct = (bias / true_ske) * 100.0;

        println!(
            "{:>9}  {:>9.4}  {:>8.4}  {:>7.1}%",
            n_p, best_ske, bias, bias_pct
        );

        csv.push_str(&format!(
            "{},{:.6},{},{:.6}\n",
            n_p, best_ske, true_ske, bias
        ));
    }

    std::fs::write("examples/iov_synthetic/calibration.csv", &csv)?;
    println!("\nSaved calibration.csv");
    Ok(())
}

fn make_sde(particles: usize) -> SDE {
    sde! {
        name: "calib_sde", params: [ke, v, ske],
        states: [central, ke0], outputs: [outeq_1], particles: particles,
        routes: [bolus(input_1) -> central],
        drift: |x, _t, dx| { dx[ke0] = -x[ke0] + ke; dx[central] = -x[ke0] * x[central]; },
        diffusion: |_, sigma| { sigma[ke0] = ske; },
        init: |_t, x| { x[ke0] = ke; },
        out: |x, _t, y| { y[outeq_1] = x[central] / v; },
    }
}
