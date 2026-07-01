//! SDE Inter-Occasion Variability (IOV) analysis.
//!
//! Stage 1: NPAG fit with ODE to find support points.
//! Stage 2: Optimize SDE diffusion parameter (ske) per support point via the
//!          [`DiffusionOptimize`] trait method on the SDE.
//!
//! Uses small synthetic data and low particle count (50) to keep the example fast.

use pharmsol::SubjectBuilderExt;
use pmcore::iov::DiffusionConfig;
use pmcore::prelude::*;

fn main() -> Result<()> {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, 1)
        .observation(1.0, 10.0, 1)
        .observation(2.0, 6.0, 1)
        .observation(4.0, 2.5, 1)
        .build();
    let data = Data::new(vec![subject]);

    // ── Stage 1: ODE fit ──

    let ode = ode! {
        name: "iov_stage1_ode",
        params: [ke, v],
        states: [central],
        outputs: [outeq_1],
        routes: [bolus(input_1) -> central],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / v;
        },
    };

    let parameters = ParameterSpace::bounded()
        .add("ke", 0.001, 2.0)
        .add("v", 5.0, 50.0);
    let prior = Theta::sobol(&parameters, 50)?;
    let error_models = AssayErrorModels::new().add(
        "outeq_1",
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
    )?;

    println!("Stage 1: Fitting ODE with NPAG...");
    let r_ode = EstimationProblem::nonparametric(ode, data.clone(), prior, error_models.clone())?
        .fit_with(NonParametricAlgorithm::npag())?;

    let n_spp = r_ode.get_theta().nspp();
    println!("  {} support points, OBJF = {:.2}", n_spp, r_ode.objf());

    // ── Stage 2: SDE IOV ──

    let sde = sde! {
        name: "iov_stage2_sde",
        params: [ke, v, ske],
        states: [central, ke0],
        outputs: [outeq_1],
        particles: 10,
        routes: [bolus(input_1) -> central],
        drift: |x, _t, dx| {
            dx[ke0] = -x[ke0] + ke;
            dx[central] = -x[ke0] * x[central];
        },
        diffusion: |_, sigma| {
            sigma[ke0] = ske;
        },
        init: |_t, x| {
            x[ke0] = ke;
        },
        out: |x, _t, y| {
            y[outeq_1] = x[central] / v;
        },
    };

    let mut joint = r_ode
        .get_theta()
        .with_added_parameter("ske", 1e-6, 1.0, 0.01)?;

    println!(
        "Stage 2: Optimizing sigma for {} support points...",
        joint.nspp()
    );

    let posterior = r_ode.posterior()?;
    let diff = sde.optimize_diffusion(
        r_ode.data(),
        &mut joint,
        &["ske".to_string()],
        r_ode.error_models(),
        Some(&posterior),
        DiffusionConfig {
            max_iter: 10,
            resampling_samples: 5,
            ..DiffusionConfig::default()
        },
    )?;

    let n_converged = diff.per_point_converged.iter().filter(|&&c| c).count();
    let mean_ll: f64 =
        diff.per_point_likelihood.iter().sum::<f64>() / diff.per_point_likelihood.len() as f64;

    println!(
        "  Converged: {}/{} points, mean log-likelihood: {:.4}",
        n_converged,
        diff.per_point_converged.len(),
        mean_ll,
    );

    Ok(())
}
