use pharmsol::equation::{metadata, ModelKind, Route, SdeSessionError};
use pharmsol::{fa, lag, Censor, Parameters, Subject, SubjectBuilderExt, SDE};
use pmcore::{
    AssayErrorModel, AssayErrorModels, ErrorPoly, SdeParticleConfig, SdeParticleError,
    SdeParticleFilter,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

const N: usize = 128;

fn model() -> SDE {
    SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            dx[0] = -x[0] * x[1];
            dx[1] = -x[1] + p[0];
        },
        |_p, diffusion| {
            diffusion[0] = 1.0;
            diffusion[1] = 0.05;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, x| x[1] = 1.0,
        |x, _p, _t, _cov, y| y[0] = x[0],
        N,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        metadata::new("sde_filter_test")
            .kind(ModelKind::Sde)
            .parameters(["ke0"])
            .states(["central", "ke_latent"])
            .outputs(["cp"])
            .route(
                Route::bolus("dose")
                    .to_state("central")
                    .inject_input_to_destination(),
            )
            .particles(N),
    )
    .unwrap()
}

fn parameters(model: &SDE) -> Parameters {
    Parameters::with_model(model, [("ke0", 1.0)]).unwrap()
}

fn assay(sigma: f64) -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            "cp",
            AssayErrorModel::additive(ErrorPoly::new(sigma, 0.0, 0.0, 0.0), 0.0),
        )
        .unwrap()
}

fn config(threshold: f64) -> SdeParticleConfig {
    SdeParticleConfig::new(N)
        .with_ess_threshold(threshold)
        .with_process_seed(7123)
        .with_resampling_seed(991)
}

fn observed_subject() -> Subject {
    Subject::builder("id1")
        .bolus(0.0, 20.0, "dose")
        .observation(0.2, 16.6434, "cp")
        .observation(0.4, 14.3233, "cp")
        .observation(0.6, 9.8468, "cp")
        .observation(0.8, 9.4177, "cp")
        .observation(1.0, 7.5170, "cp")
        .build()
}

#[test]
fn original_particle_filter_scientific_intent_remains_finite() {
    let model = model();
    let result = model
        .particle_filter(
            &observed_subject(),
            &parameters(&model),
            &assay(0.5),
            &config(0.5),
        )
        .unwrap();

    assert!(result.log_value.is_finite());
    assert_eq!(result.records.len(), 5);
}

#[test]
fn same_seeds_are_exactly_reproducible() {
    let model = model();
    let parameters = parameters(&model);
    let subject = observed_subject();
    let first = model
        .particle_filter(&subject, &parameters, &assay(0.5), &config(0.8))
        .unwrap();
    let second = model
        .particle_filter(&subject, &parameters, &assay(0.5), &config(0.8))
        .unwrap();

    assert_eq!(first, second);
}

#[test]
fn session_enforces_boundary_and_validates_ancestors() {
    let model = model();
    let parameters = parameters(&model);
    let subject = Subject::builder("barrier")
        .missing_observation(0.2, "cp")
        .missing_observation(0.4, "cp")
        .build();
    let mut rng = StdRng::seed_from_u64(9);
    let mut session = model
        .particle_session(&subject, &parameters, N, &mut rng)
        .unwrap();

    session.next_observation().unwrap().unwrap();
    assert!(matches!(
        session.next_observation(),
        Err(SdeSessionError::BoundaryPending)
    ));
    assert!(matches!(
        session.select_ancestors(&[0]),
        Err(SdeSessionError::AncestorCount { .. })
    ));
    let mut invalid = vec![0; N];
    invalid[N - 1] = N;
    assert!(matches!(
        session.select_ancestors(&invalid),
        Err(SdeSessionError::AncestorOutOfRange { .. })
    ));
    session.select_ancestors(&vec![0; N]).unwrap();
    assert!(session.next_observation().unwrap().is_some());
}

#[test]
fn ancestry_selection_changes_later_particle_states() {
    let model = model();
    let parameters = parameters(&model);
    let subject = Subject::builder("state")
        .bolus(0.0, 20.0, "dose")
        .missing_observation(0.2, "cp")
        .missing_observation(0.8, "cp")
        .build();
    let mut left_rng = StdRng::seed_from_u64(44);
    let mut right_rng = StdRng::seed_from_u64(44);
    let mut selected = model
        .particle_session(&subject, &parameters, N, &mut left_rng)
        .unwrap();
    let mut retained = model
        .particle_session(&subject, &parameters, N, &mut right_rng)
        .unwrap();

    let first = selected.next_observation().unwrap().unwrap();
    let ancestor = first
        .predictions()
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.prediction().total_cmp(&b.prediction()))
        .unwrap()
        .0;
    selected.select_ancestors(&vec![ancestor; N]).unwrap();
    retained.next_observation().unwrap();
    retained.retain_particles().unwrap();

    let selected_mean = selected
        .next_observation()
        .unwrap()
        .unwrap()
        .predictions()
        .iter()
        .map(|prediction| prediction.prediction())
        .sum::<f64>()
        / N as f64;
    let retained_mean = retained
        .next_observation()
        .unwrap()
        .unwrap()
        .predictions()
        .iter()
        .map(|prediction| prediction.prediction())
        .sum::<f64>()
        / N as f64;
    assert_ne!(selected_mean, retained_mean);
}

#[test]
fn tiny_densities_remain_finite_and_all_impossible_is_typed() {
    let model = model();
    let parameters = parameters(&model);
    let tiny = Subject::builder("tiny")
        .bolus(0.0, 20.0, "dose")
        .observation(0.2, 100.0, "cp")
        .build();
    let result = model
        .particle_filter(&tiny, &parameters, &assay(0.01), &config(0.01))
        .unwrap();
    assert!(result.log_value.is_finite());

    let impossible = Subject::builder("impossible")
        .bolus(0.0, 20.0, "dose")
        .observation(0.2, 1e308, "cp")
        .build();
    assert!(matches!(
        model.particle_filter(&impossible, &parameters, &assay(0.5), &config(0.01)),
        Err(SdeParticleError::ImpossibleObservation { .. })
    ));
}

#[test]
fn uncensored_zero_sigma_is_typed() {
    let model = model();
    let parameters = parameters(&model);
    let subject = Subject::builder("zero-sigma")
        .bolus(0.0, 20.0, "dose")
        .observation(0.2, 16.0, "cp")
        .build();

    assert!(matches!(
        model.particle_filter(&subject, &parameters, &assay(0.0), &config(0.01)),
        Err(SdeParticleError::InvalidSigma { sigma: 0.0, .. })
    ));
}

#[test]
fn ess_threshold_controls_resampling_and_no_resampling_preserves_weights() {
    let model = model();
    let parameters = parameters(&model);
    let subject = observed_subject();
    let never = model
        .particle_filter(&subject, &parameters, &assay(0.2), &config(1e-9))
        .unwrap();
    assert!(never.records.iter().all(|record| !record.resampled));
    assert!(never.records[0]
        .normalized_weights
        .windows(2)
        .any(|pair| pair[0] != pair[1]));
    assert_eq!(
        never.final_normalized_weights,
        never.records.last().unwrap().normalized_weights
    );

    let aggressive = model
        .particle_filter(&subject, &parameters, &assay(0.2), &config(1.0))
        .unwrap();
    assert!(aggressive.records.iter().any(|record| record.resampled));
    assert!(aggressive
        .records
        .iter()
        .filter(|record| record.resampled)
        .all(|record| record.ancestors.as_ref().unwrap().len() == N));

    for record in aggressive.records.iter().filter(|record| record.resampled) {
        let recorded_ess = 1.0
            / record
                .normalized_weights
                .iter()
                .map(|weight| weight * weight)
                .sum::<f64>();
        assert!((recorded_ess - record.effective_sample_size).abs() < 1e-10);
        assert!(record.effective_sample_size <= N as f64);
    }
    assert!(aggressive
        .final_normalized_weights
        .iter()
        .all(|weight| (*weight - 1.0 / N as f64).abs() < 1e-12));
}

#[test]
fn censoring_and_missing_observations_follow_sequential_rules() {
    let model = model();
    let parameters = parameters(&model);
    for censor in [Censor::BLOQ, Censor::ALOQ] {
        let subject = Subject::builder("censored")
            .bolus(0.0, 20.0, "dose")
            .censored_observation(0.2, 18.0, "cp", censor)
            .missing_observation(0.4, "cp")
            .build();
        let result = model
            .particle_filter(&subject, &parameters, &assay(0.5), &config(0.01))
            .unwrap();
        assert!(result.log_value.is_finite());
        assert_eq!(result.records[1].log_increment, 0.0);
        assert_eq!(
            result.records[0].normalized_weights,
            result.records[1].normalized_weights
        );
    }
}

#[test]
fn result_uses_predictive_mixture_not_final_mean_particle() {
    let model = model();
    let parameters = parameters(&model);
    let subject = Subject::builder("mixture")
        .bolus(0.0, 20.0, "dose")
        .observation(0.5, 10.0, "cp")
        .build();
    let result = model
        .particle_filter(&subject, &parameters, &assay(0.5), &config(1e-9))
        .unwrap();

    let mut rng = StdRng::seed_from_u64(7123);
    let mut session = model
        .particle_session(&subject, &parameters, N, &mut rng)
        .unwrap();
    let boundary = session.next_observation().unwrap().unwrap();
    let mean = boundary
        .predictions()
        .iter()
        .map(|prediction| prediction.prediction())
        .sum::<f64>()
        / N as f64;
    let sigma: f64 = 0.5;
    let final_mean_approximation = -0.5 * (2.0 * std::f64::consts::PI).ln()
        - sigma.ln()
        - (10.0 - mean).powi(2) / (2.0 * sigma * sigma);

    assert_ne!(result.log_value, final_mean_approximation);
}
