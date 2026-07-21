use pharmsol::equation::{metadata, ModelKind, Route};
use pharmsol::{fa, lag, Parameters, SubjectBuilderExt, SDE};
use pmcore::{AssayErrorModel, AssayErrorModels, ErrorPoly, SdeParticleConfig, SdeParticleFilter};

/// Scientific check that the SDE particle-filter likelihood stays finite.
#[test]
fn test_particle_filter_likelihood() {
    let subject = pharmsol::Subject::builder("id1")
        .bolus(0.0, 20.0, "dose")
        .observation(0.2, 16.6434, "cp")
        .observation(0.4, 14.3233, "cp")
        .observation(0.6, 9.8468, "cp")
        .observation(0.8, 9.4177, "cp")
        .observation(1.0, 7.5170, "cp")
        .build();

    let sde = SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            dx[0] = -x[0] * x[1];
            dx[1] = -x[1] + p[0];
        },
        |_p, diffusion| {
            diffusion[0] = 1.0;
            diffusion[1] = 0.01;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, x| x[1] = 1.0,
        |x, _p, _t, _cov, y| y[0] = x[0],
        10_000,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        metadata::new("particle_filter_test")
            .kind(ModelKind::Sde)
            .parameters(["ke0"])
            .states(["central", "ke_latent"])
            .outputs(["cp"])
            .route(
                Route::bolus("dose")
                    .to_state("central")
                    .inject_input_to_destination(),
            )
            .particles(10_000),
    )
    .expect("particle filter metadata should validate");

    let error_models = AssayErrorModels::new()
        .add(
            "cp",
            AssayErrorModel::additive(ErrorPoly::new(0.5, 0.0, 0.0, 0.0), 0.0),
        )
        .unwrap();

    const NUM_RUNS: usize = 10;
    let mut likelihoods = Vec::with_capacity(NUM_RUNS);
    for run in 0..NUM_RUNS {
        let parameters = Parameters::with_model(&sde, [("ke0", 1.0)]).unwrap();
        let config = SdeParticleConfig::new(10_000)
            .with_process_seed(run as u64)
            .with_resampling_seed(10_000 + run as u64);
        likelihoods.push(
            sde.particle_filter(&subject, &parameters, &error_models, &config)
                .unwrap()
                .log_value
                .exp(),
        );
    }

    let mean = likelihoods.iter().sum::<f64>() / NUM_RUNS as f64;
    assert!(mean.is_finite(), "Mean likelihood should be finite");
}
