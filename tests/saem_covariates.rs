use pmcore::prelude::*;

fn analytical_model() -> pharmsol::equation::Analytical {
    analytical! {
        name: "saem_covariate_integration",
        params: [ke, v, bio, frac],
        states: [central],
        outputs: [cp],
        routes: [bolus(iv) -> central],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| { y[cp] = bio * frac * x[central] / v; },
    }
}

fn ode_model() -> pharmsol::equation::ODE {
    ode! {
        name: "saem_covariate_integration",
        params: [ke, v, bio, frac],
        states: [central],
        outputs: [cp],
        routes: [bolus(iv) -> central],
        diffeq: |x, _t, dx| { dx[central] = -ke * x[central]; },
        out: |x, _t, y| { y[cp] = bio * frac * x[central] / v; },
    }
}

fn data() -> Data {
    let weights = [55.0, 74.0, 83.0, 61.0, 79.0, 68.0];
    Data::new(
        (0..6)
            .map(|index| {
                let wt = weights[index];
                let group = (index % 3) as f64;
                let group_effect = if group == 1.0 {
                    0.18
                } else if group == 2.0 {
                    -0.14
                } else {
                    0.0
                };
                let ke = (0.12_f64.ln() + 0.012 * (wt - 70.0) + group_effect).exp();
                let frac_phi = ((0.65_f64 - 0.2) / (0.9 - 0.65)).ln() + 0.004 * (wt - 70.0);
                let frac = 0.2 + 0.7 / (1.0 + (-frac_phi).exp());
                let mut subject = Subject::builder(format!("covariate_{index}"))
                    .covariate("wt", 0.0, wt)
                    .covariate("group", 0.0, group)
                    .bolus(0.0, 100.0, "iv");
                for time in [0.5, 1.0, 2.0, 4.0, 8.0, 12.0] {
                    subject =
                        subject.observation(time, frac * 100.0 * (-ke * time).exp() / 45.0, "cp");
                }
                subject.build()
            })
            .collect(),
    )
}

fn problem<E>(model: E) -> anyhow::Result<EstimationProblem<E, Parametric>>
where
    E: pharmsol::Equation + EquationMetadataSource,
{
    EstimationProblem::parametric(model, data())
        .parameter(Parameter::log("ke").with_initial(0.14))
        .parameter(Parameter::log("v").with_initial(45.0).fixed())
        .parameter(Parameter::probit("bio", 0.5, 1.5).with_initial(1.0).fixed())
        .parameter(
            Parameter::logit("frac", 0.2, 0.9)
                .with_initial(0.65)
                .fixed()
                .without_random_effect(),
        )
        .omega(
            Omega::new()
                .variance("ke", 0.08)
                .fixed_variance("v", 0.04)
                .fixed_variance("bio", 0.03)
                .fixed_covariance("ke", "v", 0.012),
        )
        .covariate_effect(CovariateEffect::continuous("ke", "wt", 70.0).with_initial(0.006))
        .covariate_effect(CovariateEffect::categorical("ke", "group", 0.0, 1.0).with_initial(0.10))
        .covariate_effect(CovariateEffect::categorical("ke", "group", 0.0, 2.0).with_initial(-0.08))
        .covariate_effect(
            CovariateEffect::continuous("frac", "wt", 70.0)
                .with_initial(0.004)
                .fixed(),
        )
        .error_model("cp", ResidualErrorModel::constant(0.12))
        .build()
}

fn signed_zero_builder(
    effects: impl IntoIterator<Item = CovariateEffect>,
    observed: f64,
) -> anyhow::Result<EstimationProblem<pharmsol::equation::Analytical, Parametric>> {
    let data = Data::new(vec![Subject::builder("signed_zero")
        .covariate("group", 0.0, observed)
        .bolus(0.0, 100.0, "iv")
        .observation(1.0, 1.0, "cp")
        .build()]);
    let mut builder = EstimationProblem::parametric(analytical_model(), data)
        .parameter(Parameter::log("ke").with_initial(0.12).fixed())
        .parameter(
            Parameter::log("v")
                .with_initial(45.0)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::probit("bio", 0.5, 1.5)
                .with_initial(1.0)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::logit("frac", 0.2, 0.9)
                .with_initial(0.65)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::new().fixed_variance("ke", 0.08));
    for effect in effects {
        builder = builder.covariate_effect(effect);
    }
    builder
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.12)).fixed(),
        )
        .build()
}

fn config(seed: u64) -> SaemConfig {
    SaemConfig::new()
        .seed(seed)
        .n_chains(1)
        .mcmc_iterations(1)
        .eta_block_iterations(0)
        .burn_in(1)
        // Include one actual exploration cycle: k1 counts burn-in plus
        // exploration, so k1 == burn_in would skip exploration entirely.
        .k1_iterations(2)
        .k2_iterations(1)
        .compute_map(false)
}

fn assert_result<E: pharmsol::Equation>(result: &ParametricResult<E>) {
    let cycles = result.cycle_diagnostics();
    assert_eq!(cycles.len(), 3);
    assert_eq!(cycles[0].phase, SaemPhase::BurnIn);
    assert_eq!(cycles[1].phase, SaemPhase::Exploration);
    assert_eq!(cycles[2].phase, SaemPhase::Smoothing);
    assert_eq!(cycles[0].stochastic_approximation_step, 0.0);
    assert_eq!(cycles[1].stochastic_approximation_step, 1.0);
    assert_eq!(cycles[2].stochastic_approximation_step, 1.0);
    assert!(matches!(
        cycles[0].omega_update.outcome,
        CovarianceCycleUpdateOutcome::NotAttempted {
            reason: CovarianceUpdateNotAttemptedReason::BurnIn
        }
    ));
    assert!(cycles[0].omega_update.proposal.is_none());
    assert!(cycles[0].omega_update.solved_target.is_none());
    for cycle in cycles {
        assert!(matches!(
            cycle.omega_iov_update.outcome,
            CovarianceCycleUpdateOutcome::NotAttempted {
                reason: CovarianceUpdateNotAttemptedReason::NotConfigured
            }
        ));
        assert!(cycle.omega_iov_update.proposal.is_none());
        assert!(cycle.omega_iov_update.solved_target.is_none());
    }
    for cycle in &cycles[1..] {
        assert!(cycle.omega_update.proposal.is_some());
        assert!(cycle.omega_update.solved_target.is_some());
        assert!(!matches!(
            cycle.omega_update.outcome,
            CovarianceCycleUpdateOutcome::NotAttempted { .. }
        ));
    }
    assert!(cycles[1]
        .omega_update
        .attempted_fractions
        .first()
        .is_some_and(|fraction| *fraction <= 0.1));
    assert_eq!(
        cycles[2].omega_update.attempted_fractions.first(),
        Some(&1.0)
    );

    for (actual, expected) in result.population_parameters()[1..]
        .iter()
        .zip([45.0, 1.0, 0.65])
    {
        assert!((actual - expected).abs() <= 1e-12);
    }
    let effects = result.covariate_estimates().unwrap();
    assert_eq!(effects.len(), 4);
    assert_eq!(effects[0].name(), "beta:ke:wt");
    assert_eq!(effects[1].name(), "beta:ke:group:1");
    assert_eq!(effects[2].name(), "beta:ke:group:2");
    assert_eq!(effects[3].name(), "beta:frac:wt");
    assert_eq!(effects[3].estimate(), 0.004);
    assert!(result.population_parameters().iter().all(|x| x.is_finite()));
    assert!(result
        .covariate_subject_population_parameters()
        .unwrap()
        .unwrap()
        .iter()
        .all(|row| {
            row.phi().iter().all(|x| x.is_finite())
                && row.psi()[2] > 0.5
                && row.psi()[2] < 1.5
                && row.psi()[3] > 0.2
                && row.psi()[3] < 0.9
        }));
    assert_eq!(result.omega()[[0, 1]], 0.012);
    assert_eq!(result.omega()[[0, 2]], 0.0);
    assert_eq!(result.omega()[[1, 2]], 0.0);
}

#[test]
fn analytical_model_exercises_covariate_and_covariance_declarations() {
    let result = problem(analytical_model())
        .unwrap()
        .fit_with(config(9001))
        .unwrap();
    assert_result(&result);
}

#[test]
fn ode_model_exercises_covariate_and_covariance_declarations() {
    let result = problem(ode_model())
        .unwrap()
        .fit_with(config(9002))
        .unwrap();
    assert_result(&result);
}

#[test]
fn signed_zero_category_levels_use_one_canonical_identity() {
    let duplicate = signed_zero_builder(
        [
            CovariateEffect::categorical("ke", "group", 1.0, 0.0)
                .with_initial(0.1)
                .fixed(),
            CovariateEffect::categorical("ke", "group", 1.0, -0.0)
                .with_initial(0.2)
                .fixed(),
        ],
        -0.0,
    )
    .unwrap_err()
    .to_string();
    assert!(duplicate.contains("duplicate"), "{duplicate}");

    let collision = signed_zero_builder(
        [CovariateEffect::categorical("ke", "group", -0.0, 0.0)
            .with_initial(0.1)
            .fixed()],
        0.0,
    )
    .unwrap_err()
    .to_string();
    assert!(collision.contains("reference"), "{collision}");

    let accepted = signed_zero_builder(
        [CovariateEffect::categorical("ke", "group", 1.0, 0.0)
            .with_initial(0.1)
            .fixed()],
        -0.0,
    )
    .unwrap();
    let model = accepted.covariates().unwrap();
    assert_eq!(
        model.subject_values()[0].value().to_bits(),
        0.0f64.to_bits()
    );
    assert_eq!(model.subject_design()[0].values(), &[1.0]);
}

#[test]
fn nonlinear_constraint_is_explicitly_rejected() {
    let error = EstimationProblem::parametric(analytical_model(), data())
        .parameter(Parameter::log("ke").with_initial(0.12))
        .parameter(Parameter::log("v").with_initial(45.0))
        .parameter(Parameter::probit("bio", 0.5, 1.5).with_initial(1.0))
        .parameter(Parameter::logit("frac", 0.2, 0.9).with_initial(0.65))
        .constraint(ParametricConstraint::nonlinear("ke * v <= 10"))
        .error_model("cp", ResidualErrorModel::constant(0.12))
        .build()
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("unsupported nonlinear parametric constraint"));
}
