//! Head-to-head: evaluate grid peak vs NM optimum on the SAME random surface.
//! Compares ske=0.0576 (grid peak) vs ske=0.0803 (NM optimum) at 500 particles × 20 resamples.

use pmcore::prelude::*;

fn main() -> Result<()> {
    let data = data::read_pmetrics("examples/iov_synthetic/data.csv")?;
    let error_models = AssayErrorModels::new().add(
        "outeq_1",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.2, 0.0, 0.0), 0.0),
    )?;

    // Use 500 particles for high precision
    let sde = sde! {
        name: "compare_sde",
        params: [ke, v, ske],
        states: [central, ke0],
        outputs: [outeq_1],
        particles: 500,
        routes: [bolus(input_1) -> central],
        drift: |x, _t, dx| { dx[ke0] = -x[ke0] + ke; dx[central] = -x[ke0] * x[central]; },
        diffusion: |_, sigma| { sigma[ke0] = ske; },
        init: |_t, x| { x[ke0] = ke; },
        out: |x, _t, y| { y[outeq_1] = x[central] / v; },
    };

    let ke = 0.2934;
    let v = 10.0;
    let n_evals = 30; // many evaluations to average out noise

    for &ske_val in &[0.0576, 0.0803] {
        let params = vec![ke, v, ske_val];
        let mut lls = Vec::with_capacity(n_evals);

        for _ in 0..n_evals {
            let mut ll_sum = 0.0;
            for subject in data.subjects() {
                let (_, likelihood) =
                    sde.simulate_subject_dense(subject, &params, Some(&error_models))?;
                if let Some(ll) = likelihood {
                    if ll > 0.0 {
                        ll_sum += ll.ln();
                    }
                }
            }
            lls.push(ll_sum);
        }

        let mean = lls.iter().sum::<f64>() / n_evals as f64;
        let var = lls.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_evals as f64;
        let se = (var / n_evals as f64).sqrt();

        println!(
            "ske={:.4}: mean LL = {:.4}, SE = {:.4}, range = [{:.4}, {:.4}]",
            ske_val,
            mean,
            se,
            lls.iter().cloned().fold(f64::INFINITY, f64::min),
            lls.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
    }

    Ok(())
}
