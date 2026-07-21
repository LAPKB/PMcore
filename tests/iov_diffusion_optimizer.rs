use faer::Mat;
use pharmsol::equation::{metadata, ModelKind, Route};
use pharmsol::{fa, lag, Data, SubjectBuilderExt, SDE};
use pmcore::iov::{DiffusionConfig, DiffusionOptimize};
use pmcore::prelude::{BoundedParameter, ParameterSpace, Posterior, Theta};
use pmcore::{AssayErrorModel, AssayErrorModels, ErrorPoly};

fn model() -> SDE {
    SDE::new(
        |x, p, _t, dx, _rateiv, _cov| dx[0] = -p[0] * x[0],
        |p, diffusion| diffusion[0] = p[1],
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, _p, _t, _cov, y| y[0] = x[0],
        64,
    )
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        metadata::new("iov_diffusion_optimizer")
            .kind(ModelKind::Sde)
            .parameters(["ke", "diff"])
            .states(["central"])
            .outputs(["cp"])
            .route(
                Route::bolus("dose")
                    .to_state("central")
                    .inject_input_to_destination(),
            )
            .particles(64),
    )
    .unwrap()
}

fn data() -> Data {
    Data::new(vec![pharmsol::Subject::builder("id")
        .bolus(0.0, 10.0, "dose")
        .observation(0.5, 6.0, "cp")
        .observation(1.0, 3.7, "cp")
        .build()])
}

fn theta() -> Theta {
    theta_with_diffusion(0.001, 0.1, &[0.01])
}

fn theta_with_diffusion(lower: f64, upper: f64, values: &[f64]) -> Theta {
    let parameters = ParameterSpace::<BoundedParameter>::new()
        .add("ke", 0.5, 1.5)
        .add("diff", lower, upper);
    Theta::from_parts(
        Mat::from_fn(
            values.len(),
            2,
            |row, column| {
                if column == 0 {
                    1.0
                } else {
                    values[row]
                }
            },
        ),
        parameters,
    )
    .unwrap()
}

fn error_models() -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            "cp",
            AssayErrorModel::additive(ErrorPoly::new(0.5, 0.0, 0.0, 0.0), 0.0),
        )
        .unwrap()
}

#[test]
fn diffusion_optimizer_mutates_selected_column_and_returns_per_point_results() {
    let sde = model();
    let data = data();
    let mut theta = theta();
    let errors = error_models();
    let result = sde
        .optimize_diffusion(
            &data,
            &mut theta,
            &["diff".to_string()],
            &errors,
            None,
            DiffusionConfig {
                max_iter: 2,
                resampling_samples: 1,
                ..DiffusionConfig::default()
            },
        )
        .unwrap();

    assert_eq!(result.per_point_likelihood.len(), 1);
    assert_eq!(result.per_point_iterations.len(), 1);
    assert_eq!(result.per_point_converged.len(), 1);
    assert_eq!(theta.matrix()[(0, 0)], 1.0);
    assert!((0.001..=0.1).contains(&theta.matrix()[(0, 1)]));
    assert!(result.per_point_likelihood[0].is_finite());
}

#[test]
fn diffusion_optimizer_rejects_invalid_public_config_without_mutating_theta() {
    let sde = model();
    let data = data();
    let errors = error_models();
    let invalid_configs = [
        DiffusionConfig {
            max_iter: 0,
            ..DiffusionConfig::default()
        },
        DiffusionConfig {
            sd_tolerance: f64::NAN,
            ..DiffusionConfig::default()
        },
        DiffusionConfig {
            sd_tolerance: 0.0,
            ..DiffusionConfig::default()
        },
        DiffusionConfig {
            initial_perturbation: 0.0,
            ..DiffusionConfig::default()
        },
        DiffusionConfig {
            initial_perturbation: 1.01,
            ..DiffusionConfig::default()
        },
        DiffusionConfig {
            resampling_samples: 0,
            ..DiffusionConfig::default()
        },
    ];

    for config in invalid_configs {
        let mut theta = theta();
        let before = theta.matrix().clone();
        assert!(sde
            .optimize_diffusion(
                &data,
                &mut theta,
                &["diff".to_string()],
                &errors,
                None,
                config,
            )
            .is_err());
        assert_eq!(theta.matrix(), &before);
    }
}

#[test]
fn diffusion_optimizer_rejects_duplicate_diffusion_parameter() {
    let sde = model();
    let data = data();
    let mut theta = theta();
    let before = theta.matrix().clone();
    let error = sde
        .optimize_diffusion(
            &data,
            &mut theta,
            &["diff".to_string(), "diff".to_string()],
            &error_models(),
            None,
            DiffusionConfig::default(),
        )
        .unwrap_err();

    assert!(error.to_string().contains("duplicate diffusion parameter"));
    assert_eq!(theta.matrix(), &before);
}

#[test]
fn diffusion_optimizer_rejects_negative_diffusion_bounds_without_mutating_theta() {
    let sde = model();
    for (lower, upper, initial) in [(-1.0, -0.1, -0.5), (-0.1, 0.1, 0.0)] {
        let mut theta = theta_with_diffusion(lower, upper, &[initial]);
        let before = theta.matrix().clone();
        let error = sde
            .optimize_diffusion(
                &data(),
                &mut theta,
                &["diff".to_string()],
                &error_models(),
                None,
                DiffusionConfig::default(),
            )
            .unwrap_err();

        assert!(error.to_string().contains("nonnegative inclusive bounds"));
        assert_eq!(theta.matrix(), &before);
    }
}

#[test]
fn diffusion_optimizer_rejects_invalid_initial_values_without_mutating_theta() {
    let sde = model();
    for initial in [f64::NAN, -0.001, 0.101] {
        let mut theta = theta_with_diffusion(0.0, 0.1, &[initial]);
        let before = theta.matrix().clone();
        let error = sde
            .optimize_diffusion(
                &data(),
                &mut theta,
                &["diff".to_string()],
                &error_models(),
                None,
                DiffusionConfig::default(),
            )
            .unwrap_err();

        assert!(error.to_string().contains("initial diffusion parameter"));
        assert_eq!(theta.matrix()[(0, 0)].to_bits(), before[(0, 0)].to_bits());
        assert_eq!(theta.matrix()[(0, 1)].to_bits(), before[(0, 1)].to_bits());
    }
}

#[test]
fn diffusion_optimizer_rejects_posterior_row_mismatch_without_mutating_theta() {
    let sde = model();
    let mut theta = theta();
    let before = theta.matrix().clone();
    let posterior = Posterior::from(Mat::from_fn(2, 1, |_, _| 1.0));
    let error = sde
        .optimize_diffusion(
            &data(),
            &mut theta,
            &["diff".to_string()],
            &error_models(),
            Some(&posterior),
            DiffusionConfig::default(),
        )
        .unwrap_err();

    assert!(error.to_string().contains("posterior row count"));
    assert_eq!(theta.matrix(), &before);
}

#[test]
fn diffusion_optimizer_rejects_posterior_column_mismatch_without_mutating_theta() {
    let sde = model();
    let mut theta = theta();
    let before = theta.matrix().clone();
    let posterior = Posterior::from(Mat::from_fn(1, 2, |_, _| 0.5));
    let error = sde
        .optimize_diffusion(
            &data(),
            &mut theta,
            &["diff".to_string()],
            &error_models(),
            Some(&posterior),
            DiffusionConfig::default(),
        )
        .unwrap_err();

    assert!(error.to_string().contains("posterior column count"));
    assert_eq!(theta.matrix(), &before);
}

#[test]
fn diffusion_optimizer_rejects_non_finite_posterior_without_mutating_theta() {
    let sde = model();
    let mut theta = theta();
    let before = theta.matrix().clone();
    let posterior = Posterior::from(Mat::from_fn(1, 1, |_, _| f64::NAN));
    let error = sde
        .optimize_diffusion(
            &data(),
            &mut theta,
            &["diff".to_string()],
            &error_models(),
            Some(&posterior),
            DiffusionConfig::default(),
        )
        .unwrap_err();

    assert!(error.to_string().contains("posterior value"));
    assert_eq!(theta.matrix(), &before);
}

#[test]
fn diffusion_optimizer_propagates_particle_filter_failure_without_mutating_theta() {
    let sde = model();
    let data = data();
    let mut theta = theta();
    let before = theta.matrix().clone();
    let error = sde
        .optimize_diffusion(
            &data,
            &mut theta,
            &["diff".to_string()],
            &AssayErrorModels::new(),
            None,
            DiffusionConfig {
                max_iter: 1,
                resampling_samples: 1,
                ..DiffusionConfig::default()
            },
        )
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("diffusion optimization failed for support point 0"));
    assert_eq!(theta.matrix(), &before);
}
