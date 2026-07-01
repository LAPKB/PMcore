//! Fair comparison verification with posterior weighting.
//!
//! 1. Stage 1: ODE fit → posterior
//! 2. High-precision surface sweep (500p × 10 resamples)
//! 3. Optimizer runs at [50, 100, 200, 500] particles WITH posterior
//! 4. Each result re-evaluated on the same high-precision surface
//! 5. Output: surface.csv + optimizer.csv

use pmcore::iov::DiffusionConfig;
use pmcore::prelude::*;

fn main() -> Result<()> {
    let data = data::read_pmetrics("examples/iov_synthetic/data.csv")?;
    let error_models = AssayErrorModels::new().add(
        "outeq_1",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.2, 0.0, 0.0), 0.0),
    )?;

    // ── Stage 1: ODE fit to get posterior ──
    let ode = ode! {
        name: "verify_stage1",
        params: [ke, v],
        states: [central],
        outputs: [outeq_1],
        routes: [bolus(input_1) -> central],
        diffeq: |x, _t, dx| { dx[central] = -ke * x[central]; },
        out: |x, _t, y| { y[outeq_1] = x[central] / v; },
    };

    let parameters = ParameterSpace::bounded()
        .add("ke", 0.01, 1.0)
        .add("v", 1.0, 30.0);
    let prior = Theta::sobol(&parameters, 500)?;

    println!("=== Stage 1: ODE fit ===");
    let r_ode = EstimationProblem::nonparametric(ode, data.clone(), prior, error_models.clone())?
        .fit_with(NonParametricAlgorithm::npag())?;
    println!(
        "  SPs: {}, OBJF: {:.2}",
        r_ode.get_theta().nspp(),
        r_ode.objf()
    );

    // Get top support point and its posterior responsibilities
    let ke_val = r_ode.get_theta().matrix()[(0, 0)];
    let v_val = r_ode.get_theta().matrix()[(0, 1)];
    let posterior = r_ode.posterior()?;
    println!("  Top SP: ke={:.4}, v={:.2}", ke_val, v_val);

    // ── Shared config ──
    let true_ske = 0.08;
    let sweep_particles = 500;
    let sweep_resamples = 10;
    let sweep_points = 30;
    let eval_sde = make_sde(sweep_particles);

    // ── 2. High-precision surface sweep ──
    let mut surface_csv = String::from("ske,log_likelihood\n");
    let mut best_ske = 0.0;
    let mut best_ll = f64::NEG_INFINITY;

    for i in 0..sweep_points {
        let ske_val = 0.005 + 0.25 * (i as f64 / (sweep_points - 1) as f64);
        let params = vec![ke_val, v_val, ske_val];
        let mut total_ll = 0.0;
        for _ in 0..sweep_resamples {
            let mut ll_sum = 0.0;
            for subject in data.subjects() {
                if let (_, Some(ll)) =
                    eval_sde.simulate_subject_dense(subject, &params, Some(&error_models))?
                {
                    if ll > 0.0 {
                        ll_sum += ll.ln();
                    }
                }
            }
            total_ll += ll_sum;
        }
        let mean_ll = total_ll / sweep_resamples as f64;
        surface_csv.push_str(&format!("{:.6},{:.4}\n", ske_val, mean_ll));
        if mean_ll > best_ll {
            best_ll = mean_ll;
            best_ske = ske_val;
        }
    }
    std::fs::write("examples/iov_synthetic/surface.csv", &surface_csv)?;
    println!("  Surface peak: ske={:.4}, LL={:.2}", best_ske, best_ll);

    // ── 3. Optimizer runs WITH posterior ──
    println!("\n=== Optimizer runs (posterior-weighted) ===");
    let particles = [50, 100, 200, 500];
    let mut optimizer_csv = String::from("particles,ske_opt,re_eval_ll,converged,iters\n");

    for &n_p in &particles {
        let opt_sde = make_sde(n_p);
        let mut joint = Theta::from_parts(
            faer::Mat::from_fn(1, 3, |_, c| [ke_val, v_val, 0.01][c]),
            ParameterSpace::bounded()
                .add("ke", 0.01, 1.0)
                .add("v", 1.0, 30.0)
                .add("ske", 1e-6, 0.5),
        )?;

        let diff = opt_sde.optimize_diffusion(
            &data,
            &mut joint,
            &["ske".to_string()],
            &error_models,
            None,
            DiffusionConfig {
                max_iter: 40,
                resampling_samples: 3,
                ..DiffusionConfig::default()
            },
        )?;

        let ske_opt = joint.matrix()[(0, 2)];

        // Re-evaluate on high-precision surface
        let params = vec![ke_val, v_val, ske_opt];
        let mut re_eval_ll = 0.0;
        for _ in 0..sweep_resamples {
            let mut ll_sum = 0.0;
            for subject in data.subjects() {
                if let (_, Some(ll)) =
                    eval_sde.simulate_subject_dense(subject, &params, Some(&error_models))?
                {
                    if ll > 0.0 {
                        ll_sum += ll.ln();
                    }
                }
            }
            re_eval_ll += ll_sum;
        }
        re_eval_ll /= sweep_resamples as f64;

        optimizer_csv.push_str(&format!(
            "{},{:.6},{:.4},{},{}\n",
            n_p, ske_opt, re_eval_ll, diff.per_point_converged[0], diff.per_point_iterations[0]
        ));

        println!(
            "  {}p: ske={:.4}  LL={:.2}  ΔLL={:.3}  err={:.3}  conv={}",
            n_p,
            ske_opt,
            re_eval_ll,
            best_ll - re_eval_ll,
            (ske_opt - true_ske).abs(),
            diff.per_point_converged[0]
        );
    }
    std::fs::write("examples/iov_synthetic/optimizer.csv", &optimizer_csv)?;

    // ── 4. Truth evaluation ──
    let mut truth_ll = 0.0;
    for _ in 0..sweep_resamples {
        let mut ll_sum = 0.0;
        for subject in data.subjects() {
            if let (_, Some(ll)) = eval_sde.simulate_subject_dense(
                subject,
                &[ke_val, v_val, true_ske],
                Some(&error_models),
            )? {
                if ll > 0.0 {
                    ll_sum += ll.ln();
                }
            }
        }
        truth_ll += ll_sum;
    }
    truth_ll /= sweep_resamples as f64;
    println!(
        "\n  Truth ske=0.080: LL={:.2}  (grid={:.2}, Δ={:.3})",
        truth_ll,
        best_ll,
        best_ll - truth_ll
    );
    println!("Done.");
    Ok(())
}

fn make_sde(particles: usize) -> SDE {
    sde! {
        name: "verify_sde", params: [ke, v, ske],
        states: [central, ke0], outputs: [outeq_1], particles: particles,
        routes: [bolus(input_1) -> central],
        drift: |x, _t, dx| { dx[ke0] = -x[ke0] + ke; dx[central] = -x[ke0] * x[central]; },
        diffusion: |_, sigma| { sigma[ke0] = ske; },
        init: |_t, x| { x[ke0] = ke; },
        out: |x, _t, y| { y[outeq_1] = x[central] / v; },
    }
}
