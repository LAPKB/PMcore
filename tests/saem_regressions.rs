use pharmsol::Predictions;
use pmcore::prelude::*;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use std::collections::BTreeMap;

fn analytical_one_compartment() -> pharmsol::equation::Analytical {
    analytical! {
        name: "saem_reproducible_analytical_one_cmt",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
        },
    }
}

fn analytical_one_compartment_with_scale() -> pharmsol::equation::Analytical {
    analytical! {
        name: "saem_correlated_subset_iiv",
        params: [ke, v, scale],
        states: [central],
        outputs: [cp],
        routes: [
            infusion(iv) -> central,
        ],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = scale * x[central] / v;
        },
    }
}

fn analytical_one_compartment_two_outputs() -> pharmsol::equation::Analytical {
    analytical! {
        name: "saem_multi_output_residuals",
        params: [ke, v],
        states: [central],
        outputs: [cp, doubled],
        routes: [
            infusion(iv) -> central,
        ],
        structure: one_compartment,
        out: |x, _p, _t, _cov, y| {
            y[cp] = x[central] / v;
            y[doubled] = 2.0 * x[central] / v;
        },
    }
}

fn validation_data() -> Data {
    Data::new(vec![
        Subject::builder("s1")
            .infusion(0.0, 100.0, "iv", 0.5)
            .observation(0.5, 4.70, "cp")
            .observation(1.0, 4.15, "cp")
            .observation(2.0, 3.15, "cp")
            .observation(4.0, 1.75, "cp")
            .build(),
        Subject::builder("s2")
            .infusion(0.0, 120.0, "iv", 0.5)
            .observation(0.5, 4.65, "cp")
            .observation(1.0, 4.25, "cp")
            .observation(2.0, 3.45, "cp")
            .observation(4.0, 2.15, "cp")
            .build(),
        Subject::builder("s3")
            .infusion(0.0, 80.0, "iv", 0.5)
            .observation(0.5, 4.45, "cp")
            .observation(1.0, 3.75, "cp")
            .observation(2.0, 2.65, "cp")
            .observation(4.0, 1.20, "cp")
            .build(),
        Subject::builder("s4")
            .infusion(0.0, 110.0, "iv", 0.5)
            .observation(0.5, 4.55, "cp")
            .observation(1.0, 4.10, "cp")
            .observation(2.0, 3.25, "cp")
            .observation(4.0, 1.95, "cp")
            .build(),
    ])
}

fn validation_problem() -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    EstimationProblem::parametric(analytical_one_compartment(), validation_data())
        .parameter(Parameter::log("ke").with_initial(0.30))
        .parameter(Parameter::log("v").with_initial(20.0))
        .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
        )
        .build()
        .expect("V01 analytical validation problem should build")
}

fn validation_config(seed: u64) -> SaemConfig {
    SaemConfig::new()
        .seed(seed)
        .n_chains(3)
        .mcmc_iterations(2)
        .burn_in(2)
        .k1_iterations(8)
        .k2_iterations(4)
        .map_max_iterations(100)
}

#[test]
fn analytical_same_seed_is_exactly_reproducible() {
    let first = validation_problem()
        .fit_with(validation_config(20_260_710))
        .expect("first V01 fit should complete");
    let second = validation_problem()
        .fit_with(validation_config(20_260_710))
        .expect("second V01 fit should complete");

    assert_eq!(first.objf().to_bits(), second.objf().to_bits());
    assert_eq!(first.iterations(), second.iterations());
    assert_eq!(
        first.population_parameters(),
        second.population_parameters()
    );
    assert_eq!(first.omega(), second.omega());
    assert_eq!(first.residual_sigmas(), second.residual_sigmas());
    assert_eq!(first.residual_sigmas(), &[0.25]);
    assert_eq!(first.residual_error_estimates().len(), 1);
    assert_eq!(first.residual_error_estimates()[0].output, "cp");
    assert_eq!(first.residual_error_estimates()[0].output_index, 0);
    assert_eq!(
        first.residual_error_estimates()[0].model,
        ResidualErrorModel::constant(0.25)
    );
    assert!(!first.residual_error_estimates()[0].estimated);
    assert_eq!(first.eta_chain_means(), second.eta_chain_means());
    assert_eq!(first.kappa_chain_means(), second.kappa_chain_means());
    assert_eq!(first.conditional_modes(), second.conditional_modes());
    assert_eq!(first.individual_summaries(), second.individual_summaries());

    assert!(first.objf().is_finite());
    assert!(first
        .population_parameters()
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
    assert!(first.omega().iter().all(|value| value.is_finite()));
}

fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn one_compartment_infusion_concentration(
    time: f64,
    dose: f64,
    duration: f64,
    ke: f64,
    volume: f64,
) -> f64 {
    let rate = dose / duration;
    let amount_at_end = rate * (-ke * duration).exp_m1().abs() / ke;
    let amount = if time <= duration {
        rate * (-ke * time).exp_m1().abs() / ke
    } else {
        amount_at_end * (-ke * (time - duration)).exp()
    };
    amount / volume
}

#[test]
fn closed_form_generator_matches_model_predictions() {
    const TIMES: [f64; 6] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
    let dose = 100.0;
    let duration = 0.5;
    let ke = 0.3;
    let volume = 20.0;
    let mut builder = Subject::builder("formula-check").infusion(0.0, dose, "iv", duration);
    for time in TIMES {
        builder = builder.observation(time, 0.0, "cp");
    }
    let subject = builder.build();
    let predictions = analytical_one_compartment()
        .estimate_predictions_dense(&subject, &[ke, volume])
        .expect("analytical prediction should succeed")
        .get_predictions();

    assert_eq!(predictions.len(), TIMES.len());
    for (prediction, time) in predictions.iter().zip(TIMES) {
        let expected = one_compartment_infusion_concentration(time, dose, duration, ke, volume);
        assert!((prediction.prediction() - expected).abs() <= 1e-12);
    }
}

#[test]
fn constant_sigma_fixture_matches_seeded_generator() {
    const TIMES: [f64; 6] = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0];
    let mut lines = include_str!("fixtures/constant_sigma.csv").lines();
    assert_eq!(lines.next(), Some("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV"));
    let mut rng = StdRng::seed_from_u64(20_260_711);
    for index in 0..48 {
        let dose = 80.0 + 10.0 * (index % 5) as f64;
        let ke = 0.30 * (0.20 * standard_normal(&mut rng)).exp();
        let volume = 20.0 * (0.20 * standard_normal(&mut rng)).exp();
        let dose_record = lines.next().expect("dose record should be present");
        assert!(dose_record.starts_with(&format!("v02_{index:03},0,.,1,{dose}")));
        for time in TIMES {
            let expected = one_compartment_infusion_concentration(time, dose, 0.5, ke, volume)
                + 0.25 * standard_normal(&mut rng);
            let record = lines.next().expect("observation record should be present");
            let fields = record.split(',').collect::<Vec<_>>();
            assert_eq!(fields[0], format!("v02_{index:03}"));
            assert_eq!(fields[1].parse::<f64>().unwrap(), time);
            assert_eq!(
                fields[2].parse::<f64>().unwrap().to_bits(),
                expected.to_bits()
            );
        }
    }
    assert_eq!(lines.next(), None);
}

fn seeded_constant_sigma_data(seed: u64, subject_count: usize) -> Data {
    const POPULATION_KE: f64 = 0.30;
    const POPULATION_V: f64 = 20.0;
    const ETA_SD: f64 = 0.20;
    const SIGMA: f64 = 0.25;
    const DURATION: f64 = 0.5;
    const TIMES: [f64; 6] = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0];

    let mut rng = StdRng::seed_from_u64(seed);
    let subjects = (0..subject_count)
        .map(|index| {
            let dose = 80.0 + 10.0 * (index % 5) as f64;
            let ke = POPULATION_KE * (ETA_SD * standard_normal(&mut rng)).exp();
            let volume = POPULATION_V * (ETA_SD * standard_normal(&mut rng)).exp();
            let mut builder =
                Subject::builder(format!("v02_{index:03}")).infusion(0.0, dose, "iv", DURATION);
            for time in TIMES {
                let prediction =
                    one_compartment_infusion_concentration(time, dose, DURATION, ke, volume);
                let observation = prediction + SIGMA * standard_normal(&mut rng);
                builder = builder.observation(time, observation, "cp");
            }
            builder.build()
        })
        .collect();
    Data::new(subjects)
}

fn constant_sigma_problem(
    data_seed: u64,
) -> EstimationProblem<pharmsol::equation::Analytical, Parametric> {
    EstimationProblem::parametric(
        analytical_one_compartment(),
        seeded_constant_sigma_data(data_seed, 48),
    )
    .parameter(Parameter::log("ke").with_initial(0.22))
    .parameter(Parameter::log("v").with_initial(25.0))
    .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
    .error_model("cp", ResidualErrorModel::constant(0.50))
    .build()
    .expect("V02 analytical known-truth problem should build")
}

fn constant_sigma_config(seed: u64) -> SaemConfig {
    SaemConfig::new()
        .seed(seed)
        .n_chains(3)
        .mcmc_iterations(3)
        .burn_in(20)
        .k1_iterations(100)
        .k2_iterations(80)
        .compute_map(false)
}

#[test]
fn estimated_constant_sigma_recovers_seeded_known_truth() {
    const TRUE_THETA: [f64; 2] = [0.30, 20.0];
    const TRUE_OMEGA_DIAGONAL: [f64; 2] = [0.04, 0.04];
    const TRUE_SIGMA: f64 = 0.25;

    let first = constant_sigma_problem(20_260_711)
        .fit_with(constant_sigma_config(20_260_712))
        .expect("first V02 known-truth fit should complete");
    let second = constant_sigma_problem(20_260_711)
        .fit_with(constant_sigma_config(20_260_712))
        .expect("second V02 known-truth fit should complete");

    assert_eq!(
        first.population_parameters(),
        second.population_parameters()
    );
    assert_eq!(first.omega(), second.omega());
    assert_eq!(first.residual_sigmas(), second.residual_sigmas());
    assert_eq!(
        first.residual_error_estimates(),
        second.residual_error_estimates()
    );
    assert!(first.objf().is_finite());
    assert_eq!(first.conditional_n2ll(), first.objf());
    assert_eq!(
        first.conditional_negative_log_likelihood() * 2.0,
        first.conditional_n2ll()
    );
    let final_cycle = first
        .cycle_diagnostics()
        .last()
        .expect("completed V02 fit should retain cycle diagnostics");
    assert_eq!(
        final_cycle.conditional_negative_log_likelihood,
        first.conditional_negative_log_likelihood()
    );
    assert_eq!(
        final_cycle.population_parameters,
        first.population_parameters()
    );
    assert_eq!(&final_cycle.omega, first.omega());
    assert_eq!(
        final_cycle.residual_error_estimates,
        first.residual_error_estimates()
    );
    assert_eq!(first.residual_sigmas().len(), 1);
    assert_eq!(first.residual_error_estimates().len(), 1);
    assert_eq!(first.residual_error_estimates()[0].output, "cp");
    assert_eq!(first.residual_error_estimates()[0].output_index, 0);
    assert_eq!(
        first.residual_error_estimates()[0].model,
        ResidualErrorModel::constant(first.residual_sigmas()[0])
    );
    assert!(first.residual_error_estimates()[0].estimated);

    let relative_error = |estimate: f64, truth: f64| (estimate - truth).abs() / truth;
    assert!(relative_error(first.population_parameters()[0], TRUE_THETA[0]) <= 0.10);
    assert!(relative_error(first.population_parameters()[1], TRUE_THETA[1]) <= 0.10);
    assert!(relative_error(first.omega()[[0, 0]], TRUE_OMEGA_DIAGONAL[0]) <= 0.50);
    assert!(relative_error(first.omega()[[1, 1]], TRUE_OMEGA_DIAGONAL[1]) <= 0.50);
    assert!(relative_error(first.residual_sigmas()[0], TRUE_SIGMA) <= 0.15);
}

fn residual_fixture(path: &str) -> Data {
    data::read_pmetrics(path).expect("residual validation fixture should parse")
}

fn iov_fixture(path: &str) -> Data {
    let mut rows = BTreeMap::<String, Vec<(usize, f64, Option<f64>, Option<f64>)>>::new();
    for line in std::fs::read_to_string(path)
        .expect("IOV validation fixture should be readable")
        .lines()
        .skip(1)
    {
        let columns = line.split(',').collect::<Vec<_>>();
        let optional = |value: &str| {
            (value != ".")
                .then(|| value.parse())
                .transpose()
                .expect("IOV numeric field should parse")
        };
        rows.entry(columns[0].to_owned()).or_default().push((
            columns[1].parse().expect("occasion should parse"),
            columns[2].parse().expect("time should parse"),
            optional(columns[3]),
            optional(columns[4]),
        ));
    }
    Data::new(
        rows.into_iter()
            .map(|(id, rows)| {
                let mut builder = Subject::builder(id);
                let mut current_occasion = None;
                for (occasion, time, observation, dose) in rows {
                    if current_occasion.is_some_and(|current| current != occasion) {
                        builder = builder.reset();
                    }
                    current_occasion = Some(occasion);
                    if let Some(dose) = dose {
                        builder = builder.infusion(time, dose, "iv", 0.5);
                    }
                    if let Some(observation) = observation {
                        builder = builder.observation(time, observation, "cp");
                    }
                }
                builder.build()
            })
            .collect(),
    )
}

fn relative_error(estimate: f64, truth: f64) -> f64 {
    (estimate - truth).abs() / truth
}

#[test]
fn proportional_fit_recovers_population_and_residual_scales() {
    const TRUE_KE: f64 = 0.30;
    const TRUE_V: f64 = 20.0;
    const TRUE_OMEGA: f64 = 0.04;
    const TRUE_PROPORTIONAL_SD: f64 = 0.10;

    for seed in [20_260_741, 20_260_742, 20_260_743] {
        let result = EstimationProblem::parametric(
            analytical_one_compartment(),
            residual_fixture("tests/fixtures/proportional_residual.csv"),
        )
        .parameter(Parameter::log("ke").with_initial(0.24))
        .parameter(Parameter::log("v").with_initial(24.0))
        .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
        .error_model("cp", ResidualErrorModel::proportional(0.20))
        .build()
        .expect("V03 proportional problem should build")
        .fit_with(
            SaemConfig::new()
                .seed(seed)
                .n_chains(4)
                .mcmc_iterations(4)
                .eta_block_iterations(1)
                .burn_in(100)
                .k1_iterations(120)
                .k2_iterations(80)
                .compute_map(false),
        )
        .expect("V03 proportional fit should complete");

        assert!(relative_error(result.population_parameters()[0], TRUE_KE) < 0.10);
        assert!(relative_error(result.population_parameters()[1], TRUE_V) < 0.10);
        assert!(relative_error(result.omega()[[0, 0]], TRUE_OMEGA) < 0.50);
        assert!(relative_error(result.omega()[[1, 1]], TRUE_OMEGA) < 0.50);
        assert!(relative_error(result.residual_sigmas()[0], TRUE_PROPORTIONAL_SD) < 0.15);
        assert_eq!(
            result.residual_error_estimates()[0].model,
            ResidualErrorModel::proportional(result.residual_sigmas()[0])
        );
        assert!(result.residual_error_estimates()[0].estimated);
        assert!(result.cycle_diagnostics()[..100]
            .iter()
            .all(|cycle| cycle.residual_diagnostics.is_empty()));
        assert!(result.cycle_diagnostics()[100..].iter().all(|cycle| {
            cycle.residual_diagnostics.len() == 1
                && cycle.residual_diagnostics[0].output == "cp"
                && cycle.residual_diagnostics[0].prediction_evaluation_count == 64 * 4 * 4
                && cycle.residual_diagnostics[0].proportional_floor_count == 0
                && cycle.residual_diagnostics[0].non_finite_prediction_count == 0
                && !cycle.residual_diagnostics[0].update_rejected
        }));
        assert_eq!(
            result.termination_reason(),
            Some(&pmcore::algorithms::StopReason::MaxCycles)
        );
    }
}

fn exponential_residual_data(seed: u64, subject_count: usize) -> Data {
    const KE: f64 = 0.30;
    const V: f64 = 20.0;
    const EXPONENTIAL_SD: f64 = 0.15;
    let error_model = ResidualErrorModel::exponential(EXPONENTIAL_SD);
    let mut rng = StdRng::seed_from_u64(seed);
    let times = [0.5, 1.0, 2.0, 4.0];

    Data::new(
        (0..subject_count)
            .map(|subject_index| {
                let individual_ke = KE;
                let individual_v = V;
                let mut subject = Subject::builder(format!("v03-exp-{}", subject_index + 1))
                    .infusion(0.0, 100.0, "iv", 0.5);
                for time in times {
                    let prediction = one_compartment_infusion_concentration(
                        time,
                        100.0,
                        0.5,
                        individual_ke,
                        individual_v,
                    );
                    let observation = error_model
                        .simulate_with_standard_normal(prediction, standard_normal(&mut rng))
                        .expect(
                            "positive analytical prediction should support lognormal simulation",
                        );
                    subject = subject.observation(time, observation, "cp");
                }
                subject.build()
            })
            .collect(),
    )
}

#[test]
fn exponential_sigma_recovers_log_scale_coefficient() {
    const TRUE_EXPONENTIAL_SD: f64 = 0.15;
    let problem = EstimationProblem::parametric(
        analytical_one_compartment(),
        exponential_residual_data(20_260_713, 64),
    )
    .parameter(
        Parameter::log("ke")
            .with_initial(0.30)
            .fixed()
            .without_random_effect(),
    )
    .parameter(
        Parameter::log("v")
            .with_initial(20.0)
            .fixed()
            .without_random_effect(),
    )
    .error_model("cp", ResidualErrorModel::exponential(0.30))
    .build()
    .expect("V03 exponential problem should build");
    let result = problem
        .fit_with(
            SaemConfig::new()
                .seed(20_260_714)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(20)
                .k1_iterations(120)
                .k2_iterations(80)
                .compute_map(false),
        )
        .expect("V03 exponential fit should complete");

    let estimate = result
        .residual_error_estimate("cp")
        .expect("named exponential estimate should exist");
    let ResidualErrorModel::Exponential { sigma } = estimate.model else {
        panic!("expected exponential residual estimate");
    };
    assert!(
        (sigma - TRUE_EXPONENTIAL_SD).abs() / TRUE_EXPONENTIAL_SD < 0.30,
        "estimated exponential sigma {sigma} should recover truth {TRUE_EXPONENTIAL_SD}"
    );
    assert!(estimate.estimated);
    assert!(result
        .population_parameters()
        .iter()
        .all(|value| value.is_finite()));
    assert!(result.omega().is_empty());
    assert!(result.conditional_negative_log_likelihood().is_finite());
    assert!(result.cycle_diagnostics()[20..].iter().all(|cycle| {
        cycle.residual_diagnostics.len() == 1
            && cycle.residual_diagnostics[0].output == "cp"
            && !cycle.residual_diagnostics[0].update_rejected
            && cycle.residual_diagnostics[0].non_finite_prediction_count == 0
            && cycle.residual_diagnostics[0].exponential_domain_violation_count == 0
    }));
}

#[test]
fn exponential_fit_recovers_population_and_residual_scales() {
    const TRUE_KE: f64 = 0.30;
    const TRUE_V: f64 = 20.0;
    const TRUE_OMEGA: f64 = 0.04;
    const TRUE_EXPONENTIAL_SD: f64 = 0.15;

    for seed in [20_260_714, 20_260_741, 20_260_742] {
        let result = EstimationProblem::parametric(
            analytical_one_compartment(),
            residual_fixture("tests/fixtures/exponential_residual.csv"),
        )
        .parameter(Parameter::log("ke").with_initial(0.24))
        .parameter(Parameter::log("v").with_initial(24.0))
        .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
        .error_model("cp", ResidualErrorModel::exponential(0.30))
        .build()
        .expect("V03 exponential full-fit problem should build")
        .fit_with(
            SaemConfig::new()
                .seed(seed)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(100)
                .k1_iterations(120)
                .k2_iterations(80)
                .compute_map(false),
        )
        .expect("V03 exponential full fit should complete");

        assert!(relative_error(result.population_parameters()[0], TRUE_KE) < 0.10);
        assert!(relative_error(result.population_parameters()[1], TRUE_V) < 0.10);
        assert!(relative_error(result.omega()[[0, 0]], TRUE_OMEGA) < 0.35);
        assert!(relative_error(result.omega()[[1, 1]], TRUE_OMEGA) < 0.35);
        assert!(relative_error(result.residual_sigmas()[0], TRUE_EXPONENTIAL_SD) < 0.10);
        assert!(result.conditional_negative_log_likelihood().is_finite());
        assert!(result.cycle_diagnostics()[100..].iter().all(|cycle| {
            cycle.residual_diagnostics.len() == 1
                && !cycle.residual_diagnostics[0].update_rejected
                && cycle.residual_diagnostics[0].non_finite_prediction_count == 0
                && cycle.residual_diagnostics[0].exponential_domain_violation_count == 0
        }));
        assert_eq!(
            result.termination_reason(),
            Some(&pmcore::algorithms::StopReason::MaxCycles)
        );
    }
}

#[test]
fn exponential_residual_fit_rejects_nonpositive_observation_domain() {
    let data = Data::new(vec![Subject::builder("invalid-exp")
        .infusion(0.0, 100.0, "iv", 0.5)
        .observation(1.0, 0.0, "cp")
        .build()]);
    let problem = EstimationProblem::parametric(analytical_one_compartment(), data)
        .parameter(
            Parameter::log("ke")
                .with_initial(0.30)
                .fixed()
                .without_random_effect(),
        )
        .parameter(
            Parameter::log("v")
                .with_initial(20.0)
                .fixed()
                .without_random_effect(),
        )
        .error_model("cp", ResidualErrorModel::exponential(0.15))
        .build()
        .expect("declaration validation should precede prediction-domain evaluation");

    let error = problem
        .fit_with(SaemConfig::new().compute_map(false))
        .expect_err("zero observation is outside the exponential residual domain");
    let message = error.to_string();
    assert!(message.contains("initial conditional likelihood is non-finite"));
    assert!(message.contains("invalid-exp"));
    assert!(message.contains("exponential residual model output 'cp'"));
    assert!(message.contains("1 non-positive or non-finite observation/prediction pair"));
    assert!(message.contains("positive finite observations and predictions"));
}

fn multi_output_residual_data(seed: u64, subject_count: usize) -> Data {
    const KE: f64 = 0.30;
    const V: f64 = 20.0;
    const ETA_SD: f64 = 0.20;
    const CONSTANT_SD: f64 = 0.25;
    const PROPORTIONAL_SD: f64 = 0.10;
    let mut rng = StdRng::seed_from_u64(seed);
    let times = [0.5, 1.0, 2.0, 4.0];

    Data::new(
        (0..subject_count)
            .map(|subject_index| {
                let individual_ke = KE * (ETA_SD * standard_normal(&mut rng)).exp();
                let individual_v = V * (ETA_SD * standard_normal(&mut rng)).exp();
                let mut subject = Subject::builder(format!("multi-{}", subject_index + 1))
                    .infusion(0.0, 100.0, "iv", 0.5);
                for time in times {
                    let cp = one_compartment_infusion_concentration(
                        time,
                        100.0,
                        0.5,
                        individual_ke,
                        individual_v,
                    );
                    let doubled = 2.0 * cp;
                    subject = subject
                        .observation(time, cp + CONSTANT_SD * standard_normal(&mut rng), "cp")
                        .observation(
                            time,
                            doubled + PROPORTIONAL_SD * doubled.abs() * standard_normal(&mut rng),
                            "doubled",
                        );
                }
                subject.build()
            })
            .collect(),
    )
}

#[test]
fn multi_output_constant_and_proportional_sigmas_update_independently() {
    const TRUE_CONSTANT_SD: f64 = 0.25;
    const TRUE_PROPORTIONAL_SD: f64 = 0.10;
    let problem = EstimationProblem::parametric(
        analytical_one_compartment_two_outputs(),
        multi_output_residual_data(20_260_720, 64),
    )
    .parameter(Parameter::log("ke").with_initial(0.24))
    .parameter(Parameter::log("v").with_initial(24.0))
    .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
    .error_model("cp", ResidualErrorModel::constant(0.50))
    .error_model("doubled", ResidualErrorModel::proportional(0.20))
    .build()
    .expect("multi-output residual problem should build");
    let result = problem
        .fit_with(
            SaemConfig::new()
                .seed(20_260_721)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(20)
                .k1_iterations(120)
                .k2_iterations(80)
                .compute_map(false),
        )
        .expect("multi-output residual fit should complete");

    assert_eq!(result.residual_error_estimates().len(), 2);
    assert_eq!(result.residual_error_estimates()[0].output, "cp");
    assert_eq!(result.residual_error_estimates()[1].output, "doubled");
    assert_eq!(
        result.residual_error_estimate("cp"),
        Some(&result.residual_error_estimates()[0])
    );
    assert_eq!(
        result.residual_error_estimate("doubled"),
        Some(&result.residual_error_estimates()[1])
    );
    assert_eq!(result.residual_error_estimate("missing"), None);
    assert!((result.residual_sigmas()[0] - TRUE_CONSTANT_SD).abs() / TRUE_CONSTANT_SD < 0.30);
    assert!(
        (result.residual_sigmas()[1] - TRUE_PROPORTIONAL_SD).abs() / TRUE_PROPORTIONAL_SD < 0.30
    );
    assert_eq!(
        result.residual_error_estimates()[0].model,
        ResidualErrorModel::constant(result.residual_sigmas()[0])
    );
    assert_eq!(
        result.residual_error_estimates()[1].model,
        ResidualErrorModel::proportional(result.residual_sigmas()[1])
    );
    assert!(result.cycle_diagnostics()[20..].iter().all(|cycle| {
        cycle.residual_diagnostic("cp") == Some(&cycle.residual_diagnostics[0])
            && cycle.residual_diagnostic("doubled") == Some(&cycle.residual_diagnostics[1])
            && cycle.residual_diagnostics.len() == 2
            && cycle.residual_diagnostics[0].output == "cp"
            && cycle.residual_diagnostics[1].output == "doubled"
            && cycle.residual_diagnostics[0].prediction_evaluation_count == 64 * 4 * 4
            && cycle.residual_diagnostics[1].prediction_evaluation_count == 64 * 4 * 4
            && cycle.residual_diagnostics[0].proportional_floor_count == 0
            && cycle.residual_diagnostics[1].proportional_floor_count == 0
            && !cycle.residual_diagnostics[0].update_rejected
            && !cycle.residual_diagnostics[1].update_rejected
    }));
}

fn combined_residual_data(seed: u64, subject_count: usize) -> Data {
    const KE: f64 = 0.30;
    const V: f64 = 20.0;
    const ETA_SD: f64 = 0.20;
    const ADDITIVE_SD: f64 = 0.20;
    const PROPORTIONAL_SD: f64 = 0.08;
    let mut rng = StdRng::seed_from_u64(seed);
    let times = [0.5, 1.0, 2.0, 4.0];

    Data::new(
        (0..subject_count)
            .map(|subject_index| {
                let dose = 50.0 * (1 + subject_index % 4) as f64;
                let individual_ke = KE * (ETA_SD * standard_normal(&mut rng)).exp();
                let individual_v = V * (ETA_SD * standard_normal(&mut rng)).exp();
                let mut subject = Subject::builder(format!("v04-{}", subject_index + 1))
                    .infusion(0.0, dose, "iv", 0.5);
                for time in times {
                    let prediction = one_compartment_infusion_concentration(
                        time,
                        dose,
                        0.5,
                        individual_ke,
                        individual_v,
                    );
                    let residual_sd =
                        (ADDITIVE_SD.powi(2) + PROPORTIONAL_SD.powi(2) * prediction.powi(2)).sqrt();
                    subject = subject.observation(
                        time,
                        prediction + residual_sd * standard_normal(&mut rng),
                        "cp",
                    );
                }
                subject.build()
            })
            .collect(),
    )
}

#[test]
fn combined_error_jointly_estimates_additive_and_proportional_scales() {
    const TRUE_ADDITIVE_SD: f64 = 0.20;
    const TRUE_PROPORTIONAL_SD: f64 = 0.08;
    let problem = EstimationProblem::parametric(
        analytical_one_compartment(),
        combined_residual_data(20_260_704, 80),
    )
    .parameter(Parameter::log("ke").with_initial(0.24))
    .parameter(Parameter::log("v").with_initial(24.0))
    .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
    .error_model("cp", ResidualErrorModel::combined(0.40, 0.15))
    .build()
    .expect("V04 combined residual problem should build");
    let result = problem
        .fit_with(
            SaemConfig::new()
                .seed(20_260_705)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(20)
                .k1_iterations(100)
                .k2_iterations(50)
                .residual_optimizer_max_iterations(100)
                .compute_map(false),
        )
        .expect("V04 combined residual fit should complete");

    let ResidualErrorModel::Combined { a, b } = result.residual_error_estimates()[0].model else {
        panic!("V04 should retain a combined residual model");
    };
    assert!(
        (a - TRUE_ADDITIVE_SD).abs() / TRUE_ADDITIVE_SD < 0.40,
        "combined additive estimate {a}, proportional estimate {b}"
    );
    assert!(
        (b - TRUE_PROPORTIONAL_SD).abs() / TRUE_PROPORTIONAL_SD < 0.40,
        "combined additive estimate {a}, proportional estimate {b}"
    );
    assert_eq!(
        result.residual_error_estimates()[0].combined_additive_estimated,
        Some(true)
    );
    assert_eq!(
        result.residual_error_estimates()[0].combined_proportional_estimated,
        Some(true)
    );
    assert!(result.cycle_diagnostics()[20..].iter().all(|cycle| {
        let diagnostics = &cycle.residual_diagnostics[0];
        !diagnostics.update_rejected
            && diagnostics.optimizer_objective.is_some_and(f64::is_finite)
            && diagnostics.optimizer_converged.is_some()
            && diagnostics.optimizer_iterations.is_some()
            && diagnostics.optimizer_termination.is_some()
            && !diagnostics.combined_additive_collapse_warning
    }));
    assert!(result.cycle_diagnostics()[20..]
        .iter()
        .any(|cycle| cycle.residual_diagnostics[0].optimizer_converged == Some(true)));
    assert!(!result.warnings().iter().any(|warning| matches!(
        warning,
        ParametricWarning::CombinedAdditiveCollapse { output, .. }
            | ParametricWarning::ResidualUpdateRejected { output, .. }
            if output == "cp"
    )));
}

#[test]
fn combined_fit_recovers_population_and_residual_scales() {
    const TRUE_KE: f64 = 0.30;
    const TRUE_V: f64 = 20.0;
    const TRUE_OMEGA: f64 = 0.04;
    const TRUE_ADDITIVE_SD: f64 = 0.20;
    const TRUE_PROPORTIONAL_SD: f64 = 0.08;

    for seed in [20_260_741, 20_260_742, 20_260_743] {
        let result = EstimationProblem::parametric(
            analytical_one_compartment(),
            residual_fixture("tests/fixtures/combined_residual.csv"),
        )
        .parameter(Parameter::log("ke").with_initial(0.24))
        .parameter(Parameter::log("v").with_initial(24.0))
        .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
        .error_model("cp", ResidualErrorModel::combined(0.40, 0.15))
        .build()
        .expect("V04 combined panel problem should build")
        .fit_with(
            SaemConfig::new()
                .seed(seed)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(100)
                .k1_iterations(100)
                .k2_iterations(50)
                .residual_optimizer_max_iterations(100)
                .compute_map(false),
        )
        .expect("V04 combined panel fit should complete");

        let ResidualErrorModel::Combined { a, b } = result.residual_error_estimates()[0].model
        else {
            panic!("V04 should retain a combined residual model");
        };
        assert!(
            relative_error(result.population_parameters()[0], TRUE_KE) < 0.10,
            "seed {seed}: KE {}",
            result.population_parameters()[0]
        );
        assert!(
            relative_error(result.population_parameters()[1], TRUE_V) < 0.10,
            "seed {seed}: V {}",
            result.population_parameters()[1]
        );
        assert!(
            relative_error(result.omega()[[0, 0]], TRUE_OMEGA) < 0.25,
            "seed {seed}: omega KE {}",
            result.omega()[[0, 0]]
        );
        assert!(
            relative_error(result.omega()[[1, 1]], TRUE_OMEGA) < 0.25,
            "seed {seed}: omega V {}",
            result.omega()[[1, 1]]
        );
        assert!(
            relative_error(a, TRUE_ADDITIVE_SD) < 0.35,
            "seed {seed}: additive {a}"
        );
        assert!(
            relative_error(b, TRUE_PROPORTIONAL_SD) < 0.10,
            "seed {seed}: proportional {b}"
        );
        assert!(result.cycle_diagnostics()[100..].iter().all(|cycle| {
            let diagnostics = &cycle.residual_diagnostics[0];
            !diagnostics.update_rejected
                && diagnostics.optimizer_objective.is_some_and(f64::is_finite)
                && !diagnostics.combined_additive_collapse_warning
        }));
        assert_eq!(
            result.termination_reason(),
            Some(&pmcore::algorithms::StopReason::MaxCycles)
        );
    }
}

#[test]
fn eta_block_mixture_preserves_combined_residual_behavior() {
    const TRUE_ADDITIVE_SD: f64 = 0.20;
    const TRUE_PROPORTIONAL_SD: f64 = 0.08;
    let result = EstimationProblem::parametric(
        analytical_one_compartment(),
        combined_residual_data(20_260_704, 80),
    )
    .parameter(Parameter::log("ke").with_initial(0.24))
    .parameter(Parameter::log("v").with_initial(24.0))
    .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
    .error_model("cp", ResidualErrorModel::combined(0.40, 0.15))
    .build()
    .expect("V04 block-mixture problem should build")
    .fit_with(
        SaemConfig::new()
            .seed(20_260_705)
            .n_chains(4)
            .mcmc_iterations(4)
            .eta_block_iterations(1)
            .burn_in(20)
            .k1_iterations(100)
            .k2_iterations(50)
            .residual_optimizer_max_iterations(100)
            .compute_map(false),
    )
    .expect("V04 block-mixture fit should complete");

    let ResidualErrorModel::Combined { a, b } = result.residual_error_estimates()[0].model else {
        panic!("V04 should retain a combined residual model");
    };
    assert!(
        (a - TRUE_ADDITIVE_SD).abs() / TRUE_ADDITIVE_SD < 0.40,
        "block-mixture additive estimate {a}, proportional estimate {b}"
    );
    assert!(
        (b - TRUE_PROPORTIONAL_SD).abs() / TRUE_PROPORTIONAL_SD < 0.40,
        "block-mixture additive estimate {a}, proportional estimate {b}"
    );
    assert!(result.cycle_diagnostics().iter().all(|cycle| {
        cycle.eta_block_proposals == 80 * 4
            && cycle.eta_block_accepted + cycle.eta_block_rejected == cycle.eta_block_proposals
            && cycle.eta_block_subject_acceptance_rates.len() == 80
    }));
}

#[test]
fn combined_error_can_fix_additive_and_estimate_proportional_component() {
    const FIXED_ADDITIVE_SD: f64 = 0.20;
    const TRUE_PROPORTIONAL_SD: f64 = 0.08;
    let problem = EstimationProblem::parametric(
        analytical_one_compartment(),
        combined_residual_data(20_260_704, 80),
    )
    .parameter(Parameter::log("ke").with_initial(0.24))
    .parameter(Parameter::log("v").with_initial(24.0))
    .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
    .error_model(
        "cp",
        ParametricErrorModel::new(ResidualErrorModel::combined(FIXED_ADDITIVE_SD, 0.15))
            .fixed_combined_additive(),
    )
    .build()
    .expect("V04 partially fixed combined problem should build");
    let result = problem
        .fit_with(
            SaemConfig::new()
                .seed(20_260_705)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(20)
                .k1_iterations(100)
                .k2_iterations(50)
                .residual_optimizer_max_iterations(100)
                .compute_map(false),
        )
        .expect("V04 partially fixed combined fit should complete");

    let estimate = &result.residual_error_estimates()[0];
    let ResidualErrorModel::Combined { a, b } = estimate.model else {
        panic!("V04 should retain a combined residual model");
    };
    assert_eq!(a, FIXED_ADDITIVE_SD);
    assert!((b - TRUE_PROPORTIONAL_SD).abs() / TRUE_PROPORTIONAL_SD < 0.40);
    assert_eq!(estimate.combined_additive_estimated, Some(false));
    assert_eq!(estimate.combined_proportional_estimated, Some(true));
    assert!(estimate.estimated);
    assert!(result.cycle_diagnostics()[20..]
        .iter()
        .all(|cycle| { !cycle.residual_diagnostics[0].combined_additive_collapse_warning }));
}

#[test]
fn correlated_subset_iiv_fit_preserves_structure() {
    const TRUE_KE: f64 = 0.30;
    const TRUE_V: f64 = 20.0;
    const TRUE_OMEGA: f64 = 0.04;
    const TRUE_SIGMA: f64 = 0.25;

    for seed in [20_260_741, 20_260_742, 20_260_743] {
        let result = EstimationProblem::parametric(
            analytical_one_compartment_with_scale(),
            residual_fixture("tests/fixtures/correlated_iiv.csv"),
        )
        .parameter(Parameter::log("ke").with_initial(0.24))
        .parameter(Parameter::log("v").with_initial(24.0))
        .parameter(
            Parameter::log("scale")
                .with_initial(1.0)
                .fixed()
                .without_random_effect(),
        )
        .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]).covariance("ke", "v", 0.03))
        .error_model("cp", ResidualErrorModel::constant(0.50))
        .build()
        .expect("V05 correlated subset-IIV problem should build")
        .fit_with(
            SaemConfig::new()
                .seed(seed)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(100)
                .k1_iterations(120)
                .k2_iterations(80)
                .compute_map(false),
        )
        .expect("V05 correlated subset-IIV fit should complete");

        assert_eq!(result.population_parameters().len(), 3);
        assert_eq!(result.population_parameters()[2], 1.0);
        assert_eq!(result.random_effect_names(), ["ke", "v"]);
        assert_eq!(result.omega().dim(), (2, 2));
        assert!(relative_error(result.population_parameters()[0], TRUE_KE) < 0.10);
        assert!(relative_error(result.population_parameters()[1], TRUE_V) < 0.10);
        assert!(relative_error(result.omega()[[0, 0]], TRUE_OMEGA) < 0.35);
        assert!(relative_error(result.omega()[[1, 1]], TRUE_OMEGA) < 0.35);
        // This finite fixture consistently estimates covariance near 0.029
        // across independent implementations rather than recovering the
        // generating-population value exactly.
        assert!((0.027..=0.030).contains(&result.omega()[[0, 1]]));
        assert_eq!(result.omega()[[0, 1]], result.omega()[[1, 0]]);
        assert!(
            result.omega()[[0, 0]] * result.omega()[[1, 1]] - result.omega()[[0, 1]].powi(2) > 0.0
        );
        assert!(relative_error(result.residual_sigmas()[0], TRUE_SIGMA) < 0.10);
        assert_eq!(
            result.termination_reason(),
            Some(&pmcore::algorithms::StopReason::MaxCycles)
        );
    }
}

fn two_occasion_iov_data(seed: u64, subject_count: usize) -> Data {
    const KE: f64 = 0.30;
    const V: f64 = 20.0;
    const ETA_SD: f64 = 0.15;
    const KAPPA_SD: f64 = 0.20;
    const SIGMA: f64 = 0.25;
    let mut rng = StdRng::seed_from_u64(seed);
    let times = [0.5, 1.0, 2.0];

    Data::new(
        (0..subject_count)
            .map(|subject_index| {
                let eta_ke = ETA_SD * standard_normal(&mut rng);
                let eta_v = ETA_SD * standard_normal(&mut rng);
                let individual_v = V * eta_v.exp();
                let mut subject = Subject::builder(format!("v06-{}", subject_index + 1));
                for occasion in 0..2 {
                    if occasion > 0 {
                        subject = subject.reset();
                    }
                    subject = subject.infusion(0.0, 100.0, "iv", 0.5);
                    let kappa_ke = KAPPA_SD * standard_normal(&mut rng);
                    let occasion_ke = KE * (eta_ke + kappa_ke).exp();
                    for time in times {
                        let prediction = one_compartment_infusion_concentration(
                            time,
                            100.0,
                            0.5,
                            occasion_ke,
                            individual_v,
                        );
                        subject = subject.observation(
                            time,
                            prediction + SIGMA * standard_normal(&mut rng),
                            "cp",
                        );
                    }
                }
                subject.build()
            })
            .collect(),
    )
}

#[test]
fn two_occasion_iov_recovers_distinct_eta_and_kappa_covariances() {
    const TRUE_OMEGA_IOV: f64 = 0.04;
    let problem = EstimationProblem::parametric(
        analytical_one_compartment(),
        two_occasion_iov_data(20_260_706, 48),
    )
    .parameter(Parameter::log("ke").with_initial(0.24))
    .parameter(Parameter::log("v").with_initial(24.0))
    .omega(Omega::diagonal([("ke", 0.05), ("v", 0.05)]))
    .iov(Iov::diagonal([("ke", 0.08)]))
    .error_model("cp", ResidualErrorModel::constant(0.50))
    .build()
    .expect("V06 two-occasion IOV problem should build");
    let result = problem
        .fit_with(
            SaemConfig::new()
                .seed(20_260_707)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(20)
                .k1_iterations(120)
                .k2_iterations(80)
                .compute_map(false),
        )
        .expect("V06 two-occasion IOV fit should complete");

    let omega_iov = result
        .omega_iov()
        .expect("V06 result should retain Omega_IOV");
    assert_eq!(result.iov_effect_names(), ["ke"]);
    assert_eq!(omega_iov.dim(), (1, 1));
    assert!(omega_iov[[0, 0]].is_finite());
    assert!(omega_iov[[0, 0]] > 0.0);
    assert!((omega_iov[[0, 0]] - TRUE_OMEGA_IOV).abs() / TRUE_OMEGA_IOV < 0.75);
    assert_eq!(result.kappa_chain_means().len(), 96);
    assert!(result.kappa_chain_means().chunks_exact(2).all(|occasions| {
        occasions[0].subject_id == occasions[1].subject_id
            && occasions[0].occasion_index == 0
            && occasions[1].occasion_index == 1
    }));
    assert_eq!(
        result.termination_reason(),
        Some(&pmcore::algorithms::StopReason::MaxCycles)
    );
    assert!(result
        .cycle_diagnostics()
        .iter()
        .all(|cycle| cycle.kappa_proposals == 48 * 4 * 4 * 2));
    assert!(result
        .cycle_diagnostics()
        .iter()
        .all(|cycle| cycle.kappa_accepted + cycle.kappa_rejected == cycle.kappa_proposals));
    assert!(result.omega()[[0, 0]].is_finite());
    assert!(result.omega()[[1, 1]].is_finite());
    assert!(result.residual_sigmas()[0].is_finite());
}

#[test]
fn iov_fit_recovers_occasion_variance_and_finite_iiv() {
    const TRUE_KE: f64 = 0.30;
    const TRUE_V: f64 = 20.0;
    const TRUE_OMEGA_V: f64 = 0.0225;
    const TRUE_OMEGA_IOV: f64 = 0.04;
    const TRUE_SIGMA: f64 = 0.25;

    for seed in [20_260_741, 20_260_742, 20_260_743] {
        let result = EstimationProblem::parametric(
            analytical_one_compartment(),
            iov_fixture("tests/fixtures/two_occasion_iov.csv"),
        )
        .parameter(Parameter::log("ke").with_initial(0.24))
        .parameter(Parameter::log("v").with_initial(24.0))
        .omega(Omega::diagonal([("ke", 0.05), ("v", 0.05)]))
        .iov(Iov::diagonal([("ke", 0.08)]))
        .error_model("cp", ResidualErrorModel::constant(0.50))
        .build()
        .expect("V06 shared IOV problem should build")
        .fit_with(
            SaemConfig::new()
                .seed(seed)
                .n_chains(4)
                .mcmc_iterations(20)
                .burn_in(100)
                .k1_iterations(120)
                .k2_iterations(80)
                .compute_map(false),
        )
        .expect("V06 shared IOV fit should complete");

        let omega_iov = result.omega_iov().expect("V06 should retain Omega_IOV");
        let eta_ke_variance = result.omega()[[0, 0]];
        let total_ke_variance = eta_ke_variance + omega_iov[[0, 0]];
        assert!(relative_error(result.population_parameters()[0], TRUE_KE) < 0.10);
        assert!(relative_error(result.population_parameters()[1], TRUE_V) < 0.10);
        assert!(eta_ke_variance.is_finite() && (0.01..0.08).contains(&eta_ke_variance));
        assert!(relative_error(result.omega()[[1, 1]], TRUE_OMEGA_V) < 0.25);
        assert!(relative_error(omega_iov[[0, 0]], TRUE_OMEGA_IOV) < 0.20);
        assert!((0.05..0.10).contains(&total_ke_variance));
        assert!(relative_error(result.residual_sigmas()[0], TRUE_SIGMA) < 0.10);
        assert_eq!(result.kappa_chain_means().len(), 96);
        assert!(result.cycle_diagnostics().iter().all(|cycle| {
            cycle.kappa_proposals == 48 * 4 * 20 * 2
                && cycle.kappa_accepted + cycle.kappa_rejected == cycle.kappa_proposals
        }));
        assert_eq!(
            result.termination_reason(),
            Some(&pmcore::algorithms::StopReason::MaxCycles)
        );
    }
}

fn uneven_correlated_iov_data(seed: u64) -> Data {
    const KE: f64 = 0.30;
    const V: f64 = 20.0;
    const KAPPA_KE_SD: f64 = 0.15;
    const KAPPA_V_SD: f64 = 0.10;
    const KAPPA_CORRELATION: f64 = 0.40;
    const SIGMA: f64 = 0.25;
    let mut rng = StdRng::seed_from_u64(seed);
    let times = [0.5, 2.0];

    Data::new(
        (0..12)
            .map(|subject_index| {
                let occasion_count = 1 + subject_index % 3;
                let mut subject = Subject::builder(format!("v06-uneven-{}", subject_index + 1));
                for occasion_index in 0..occasion_count {
                    if occasion_index > 0 {
                        subject = subject.reset();
                    }
                    subject = subject.infusion(0.0, 100.0, "iv", 0.5);
                    let z1 = standard_normal(&mut rng);
                    let z2 = standard_normal(&mut rng);
                    let kappa_ke = KAPPA_KE_SD * z1;
                    let kappa_v = KAPPA_V_SD
                        * (KAPPA_CORRELATION * z1 + (1.0 - KAPPA_CORRELATION.powi(2)).sqrt() * z2);
                    let occasion_ke = KE * kappa_ke.exp();
                    let occasion_v = V * kappa_v.exp();
                    for time in times {
                        let prediction = one_compartment_infusion_concentration(
                            time,
                            100.0,
                            0.5,
                            occasion_ke,
                            occasion_v,
                        );
                        subject = subject.observation(
                            time,
                            prediction + SIGMA * standard_normal(&mut rng),
                            "cp",
                        );
                    }
                }
                subject.build()
            })
            .collect(),
    )
}

#[test]
fn two_dimensional_iov_supports_correlation_and_uneven_occasion_counts() {
    let problem = EstimationProblem::parametric(
        analytical_one_compartment(),
        uneven_correlated_iov_data(20_260_724),
    )
    .parameter(Parameter::log("ke").with_initial(0.30).fixed())
    .parameter(Parameter::log("v").with_initial(20.0).fixed())
    .omega(
        Omega::new()
            .fixed_variance("ke", 0.01)
            .fixed_variance("v", 0.01),
    )
    .iov(
        Iov::new()
            .fixed_variance("ke", 0.0225)
            .fixed_variance("v", 0.01)
            .fixed_covariance("ke", "v", 0.006),
    )
    .error_model(
        "cp",
        ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
    )
    .build()
    .expect("correlated two-dimensional IOV problem should build");
    let result = problem
        .fit_with(
            SaemConfig::new()
                .seed(20_260_725)
                .n_chains(2)
                .mcmc_iterations(2)
                .burn_in(2)
                .k1_iterations(8)
                .k2_iterations(4)
                .compute_map(false),
        )
        .expect("uneven two-dimensional IOV fit should complete");

    assert_eq!(result.iov_effect_names(), ["ke", "v"]);
    assert_eq!(
        result.omega_iov(),
        Some(&ndarray::array![[0.0225, 0.006], [0.006, 0.01]])
    );
    assert_eq!(result.kappa_chain_means().len(), 24);
    assert!(result
        .kappa_chain_means()
        .iter()
        .all(|estimate| estimate.values.len() == 2));
    for subject_index in 0..12 {
        let subject_id = format!("v06-uneven-{}", subject_index + 1);
        let occasion_count = 1 + subject_index % 3;
        for occasion_index in 0..occasion_count {
            assert!(result
                .kappa_chain_mean(&subject_id, occasion_index)
                .is_some());
        }
        assert!(result
            .kappa_chain_mean(&subject_id, occasion_count)
            .is_none());
    }
    for cycle in result.cycle_diagnostics() {
        assert_eq!(cycle.kappa_proposals, 24 * 2 * 2, "cycle {cycle:?}");
        assert_eq!(
            cycle.kappa_accepted + cycle.kappa_rejected,
            cycle.kappa_proposals,
            "cycle {cycle:?}"
        );
        assert!(!cycle.omega_update_rejected, "cycle {cycle:?}");
        assert!(!cycle.omega_iov_update_rejected, "cycle {cycle:?}");
    }
    assert!(!result.warnings().iter().any(|warning| matches!(
        warning,
        ParametricWarning::OmegaUpdateRejected { .. }
            | ParametricWarning::OmegaIovUpdateRejected { .. }
    )));
}

fn combined_iov_data(seed: u64, subject_count: usize) -> Data {
    const KE: f64 = 0.30;
    const V: f64 = 20.0;
    const ETA_SD: f64 = 0.15;
    const KAPPA_SD: f64 = 0.20;
    const ADDITIVE_SD: f64 = 0.50;
    const PROPORTIONAL_SD: f64 = 0.08;
    let mut rng = StdRng::seed_from_u64(seed);
    let times = [0.5, 2.0, 8.0];

    Data::new(
        (0..subject_count)
            .map(|subject_index| {
                let eta_ke = ETA_SD * standard_normal(&mut rng);
                let eta_v = ETA_SD * standard_normal(&mut rng);
                let individual_v = V * eta_v.exp();
                let mut subject = Subject::builder(format!("combined-iov-{}", subject_index + 1));
                for occasion in 0..2 {
                    if occasion > 0 {
                        subject = subject.reset();
                    }
                    let dose = if occasion == 0 { 25.0 } else { 200.0 };
                    subject = subject.infusion(0.0, dose, "iv", 0.5);
                    let kappa_ke = KAPPA_SD * standard_normal(&mut rng);
                    let occasion_ke = KE * (eta_ke + kappa_ke).exp();
                    for time in times {
                        let prediction = one_compartment_infusion_concentration(
                            time,
                            dose,
                            0.5,
                            occasion_ke,
                            individual_v,
                        );
                        let residual_sd = (ADDITIVE_SD.powi(2)
                            + PROPORTIONAL_SD.powi(2) * prediction.powi(2))
                        .sqrt();
                        subject = subject.observation(
                            time,
                            prediction + residual_sd * standard_normal(&mut rng),
                            "cp",
                        );
                    }
                }
                subject.build()
            })
            .collect(),
    )
}

#[test]
fn combined_error_updates_from_iiv_and_iov_prediction_pairs() {
    const TRUE_ADDITIVE_SD: f64 = 0.50;
    const TRUE_PROPORTIONAL_SD: f64 = 0.08;
    let problem = EstimationProblem::parametric(
        analytical_one_compartment(),
        combined_iov_data(20_260_722, 40),
    )
    .parameter(Parameter::log("ke").with_initial(0.24))
    .parameter(Parameter::log("v").with_initial(24.0))
    .omega(Omega::diagonal([("ke", 0.05), ("v", 0.05)]))
    .iov(Iov::diagonal([("ke", 0.08)]))
    .error_model("cp", ResidualErrorModel::combined(0.40, 0.15))
    .build()
    .expect("combined IOV problem should build");
    let result = problem
        .fit_with(
            SaemConfig::new()
                .seed(20_260_723)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(20)
                .k1_iterations(100)
                .k2_iterations(50)
                .residual_optimizer_max_iterations(100)
                .compute_map(false),
        )
        .expect("combined IOV fit should complete");

    let ResidualErrorModel::Combined { a, b } = result.residual_error_estimates()[0].model else {
        panic!("combined IOV fit should retain its residual family");
    };
    assert!((a - TRUE_ADDITIVE_SD).abs() / TRUE_ADDITIVE_SD < 0.50);
    assert!(b.is_finite() && b > 0.0 && b < 0.30);
    assert!(b > TRUE_PROPORTIONAL_SD * 0.25);
    assert!(result.omega_iov().is_some_and(|omega| omega[[0, 0]] > 0.0));
    assert_eq!(result.kappa_chain_means().len(), 80);
    assert!(result.cycle_diagnostics()[20..].iter().all(|cycle| {
        let residual = &cycle.residual_diagnostics[0];
        cycle.kappa_proposals == 40 * 4 * 4 * 2
            && residual.prediction_evaluation_count == 40 * 4 * 6
            && residual.optimizer_objective.is_some_and(f64::is_finite)
            && !residual.update_rejected
    }));
}

fn sparse_iiv_data() -> Data {
    Data::new(
        [8.5, 10.0, 11.5, 9.25]
            .into_iter()
            .enumerate()
            .map(|(index, observation)| {
                Subject::builder(format!("sparse-{}", index + 1))
                    .infusion(0.0, 100.0, "iv", 0.5)
                    .observation(1.0, observation, "cp")
                    .build()
            })
            .collect(),
    )
}

#[test]
fn sparse_correlated_iiv_fit_remains_positive_definite() {
    for seed in [20_260_708, 20_260_742, 20_260_743] {
        let result = EstimationProblem::parametric(
            analytical_one_compartment(),
            residual_fixture("tests/fixtures/sparse_iiv.csv"),
        )
        .parameter(Parameter::log("ke").with_initial(0.25))
        .parameter(Parameter::log("v").with_initial(10.0))
        .omega(Omega::diagonal([("ke", 0.25), ("v", 0.25)]).covariance("ke", "v", 0.20))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed(),
        )
        .build()
        .expect("V08 sparse correlated-IIV problem should build")
        .fit_with(
            SaemConfig::new()
                .seed(seed)
                .n_chains(4)
                .mcmc_iterations(4)
                .burn_in(20)
                .k1_iterations(80)
                .k2_iterations(40)
                .omega_sa_max_step(0.1)
                .compute_map(false),
        )
        .expect("V08 sparse fit should complete without covariance collapse");

        let omega = result.omega();
        let determinant = omega[[0, 0]] * omega[[1, 1]] - omega[[0, 1]].powi(2);
        let correlation = omega[[0, 1]] / (omega[[0, 0]] * omega[[1, 1]]).sqrt();
        assert!(result
            .population_parameters()
            .iter()
            .all(|value| value.is_finite()));
        assert!(omega.iter().all(|value| value.is_finite()));
        assert!(omega[[0, 0]] >= 1e-10);
        assert!(omega[[1, 1]] >= 1e-10);
        assert!(determinant > 0.0);
        assert_eq!(result.cycle_diagnostics().len(), 120);
        assert!(result.cycle_diagnostics().iter().all(|cycle| {
            cycle.eta_proposals == 128
                && cycle.eta_accepted + cycle.eta_rejected == cycle.eta_proposals
                && cycle.kappa_proposals == 0
                && cycle.kappa_subject_acceptance_rates.is_empty()
        }));
        let rejected_omega_cycles = result
            .cycle_diagnostics()
            .iter()
            .filter(|cycle| cycle.omega_update_rejected)
            .count();
        let rejection_warning = result.warnings().iter().find_map(|warning| match warning {
            ParametricWarning::OmegaUpdateRejected { cycles, .. } => Some(*cycles),
            _ => None,
        });
        assert_eq!(
            rejection_warning,
            (rejected_omega_cycles > 0).then_some(rejected_omega_cycles)
        );
        // One observation cannot identify two correlated random effects, so a
        // near-boundary correlation is statistically possible. Robustness here
        // means finite, strictly positive-definite covariance rather than an
        // arbitrary correlation shrinkage target.
        assert!(correlation.abs() < 1.0);
        assert_eq!(
            result.termination_reason(),
            Some(&pmcore::algorithms::StopReason::MaxCycles)
        );
    }
}

#[test]
fn eta_block_mixture_remains_finite_and_positive_definite() {
    let result = EstimationProblem::parametric(analytical_one_compartment(), sparse_iiv_data())
        .parameter(Parameter::log("ke").with_initial(0.25))
        .parameter(Parameter::log("v").with_initial(10.0))
        .omega(Omega::diagonal([("ke", 0.25), ("v", 0.25)]).covariance("ke", "v", 0.20))
        .error_model(
            "cp",
            ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed(),
        )
        .build()
        .expect("V08 block-mixture problem should build")
        .fit_with(
            SaemConfig::new()
                .seed(20_260_708)
                .n_chains(4)
                .mcmc_iterations(4)
                .eta_block_iterations(1)
                .burn_in(20)
                .k1_iterations(80)
                .k2_iterations(40)
                .omega_sa_max_step(0.1)
                .compute_map(false),
        )
        .expect("V08 block-mixture fit should complete");

    let omega = result.omega();
    let determinant = omega[[0, 0]] * omega[[1, 1]] - omega[[0, 1]].powi(2);
    assert!(omega.iter().all(|value| value.is_finite()));
    assert!(determinant > 0.0);
    assert!(result.cycle_diagnostics().iter().all(|cycle| {
        cycle.eta_block_proposals == 4 * 4
            && cycle.eta_block_accepted + cycle.eta_block_rejected == cycle.eta_block_proposals
            && cycle.eta_proposals == 128 + 16
            && cycle.eta_block_subject_acceptance_rates.len() == 4
    }));
}

#[test]
fn joint_eta_kappa_conditional_modes_match_fixed_fixture() {
    const EXPECTED: [[f64; 5]; 4] = [
        [
            0.133_062_784_399_915_76,
            -0.067_013_580_960_008_37,
            0.001_491_987_806_696_499_3,
            0.235_068_327_315_337_67,
            -3.593_568_882_829_558_7,
        ],
        [
            -0.065_327_837_183_284_7,
            -0.020_090_278_178_073_37,
            -0.011_648_729_042_780_788,
            -0.104_521_845_434_551_87,
            -4.100_406_935_081_131,
        ],
        [
            -0.004_356_546_702_307_954,
            0.068_298_914_413_273_82,
            0.125_753_711_642_636_2,
            -0.133_475_798_701_328_86,
            -1.584_497_158_120_898_4,
        ],
        [
            0.069_515_500_467_549_06,
            0.112_919_976_382_365_78,
            0.017_704_794_919_049_423,
            0.105_901_941_785_214_2,
            -2.100_335_072_625_273_6,
        ],
    ];

    let result = EstimationProblem::parametric(
        analytical_one_compartment(),
        iov_fixture("tests/fixtures/conditional_modes.csv"),
    )
    .parameter(Parameter::log("ke").with_initial(0.30).fixed())
    .parameter(Parameter::log("v").with_initial(20.0).fixed())
    .omega(
        Omega::new()
            .fixed_variance("ke", 0.0225)
            .fixed_variance("v", 0.0225),
    )
    .iov(Iov::new().fixed_variance("ke", 0.04))
    .error_model(
        "cp",
        ParametricErrorModel::new(ResidualErrorModel::constant(0.25)).fixed(),
    )
    .build()
    .expect("V10 fixed conditional-mode problem should build")
    .fit_with(
        SaemConfig::new()
            .seed(20_260_741)
            .n_chains(4)
            .mcmc_iterations(4)
            .burn_in(20)
            .k1_iterations(20)
            .k2_iterations(10)
            .map_max_iterations(500),
    )
    .expect("V10 conditional modes should complete");

    assert_eq!(result.conditional_modes().len(), EXPECTED.len());
    for (mode, expected) in result.conditional_modes().iter().zip(EXPECTED) {
        assert_eq!(mode.eta.len(), 2);
        assert_eq!(mode.kappas.len(), 2);
        assert_eq!(mode.kappas[0].occasion_index, 0);
        assert_eq!(mode.kappas[1].occasion_index, 1);
        assert!((mode.eta[0] - expected[0]).abs() < 5e-5);
        assert!((mode.eta[1] - expected[1]).abs() < 5e-5);
        assert!((mode.kappas[0].values[0] - expected[2]).abs() < 5e-5);
        assert!((mode.kappas[1].values[0] - expected[3]).abs() < 5e-5);
        assert!((mode.objective - expected[4]).abs() < 1e-8);
        assert!(mode.converged);
    }
    assert_eq!(
        result.termination_reason(),
        Some(&pmcore::algorithms::StopReason::MaxCycles)
    );
}

fn build_with_residual_model(
    model: ParametricErrorModel,
) -> anyhow::Result<EstimationProblem<pharmsol::equation::Analytical, Parametric>> {
    EstimationProblem::parametric(analytical_one_compartment(), validation_data())
        .parameter(Parameter::log("ke").with_initial(0.30))
        .parameter(Parameter::log("v").with_initial(20.0))
        .omega(Omega::diagonal([("ke", 0.09), ("v", 0.09)]))
        .error_model("cp", model)
        .build()
}

#[test]
fn residual_model_declarations_validate_supported_parameter_domains() {
    let zero_proportional = build_with_residual_model(ResidualErrorModel::proportional(0.0).into())
        .expect_err("zero proportional SD must fail closed");
    assert!(zero_proportional
        .to_string()
        .contains("proportional residual SD coefficient"));

    let non_finite_constant =
        build_with_residual_model(ResidualErrorModel::constant(f64::NAN).into())
            .expect_err("non-finite constant SD must fail closed");
    assert!(non_finite_constant
        .to_string()
        .contains("constant residual SD"));

    build_with_residual_model(
        ParametricErrorModel::new(ResidualErrorModel::exponential(0.25)).fixed(),
    )
    .expect("positive fixed exponential log-scale SD should build");
    let invalid_exponential =
        build_with_residual_model(ResidualErrorModel::exponential(0.0).into())
            .expect_err("zero exponential log-scale SD must fail closed");
    assert!(invalid_exponential
        .to_string()
        .contains("exponential residual log-scale SD"));

    build_with_residual_model(ResidualErrorModel::combined(0.25, 0.10).into())
        .expect("positive estimated combined residual coefficients should build");
    let zero_combined = build_with_residual_model(ResidualErrorModel::combined(0.25, 0.0).into())
        .expect_err("estimated combined components must both start above zero");
    assert!(zero_combined
        .to_string()
        .contains("estimated combined proportional SD"));

    build_with_residual_model(
        ParametricErrorModel::new(ResidualErrorModel::combined(0.25, 0.0))
            .fixed_combined_proportional(),
    )
    .expect("a fixed zero combined component should remain valid");

    build_with_residual_model(
        ParametricErrorModel::new(ResidualErrorModel::combined(0.25, 0.0)).fixed(),
    )
    .expect("fixed combined residual scoring may fix one component at zero");
}

#[test]
fn exponential_residual_fit_uses_positive_domain_and_updates_log_scale_sigma() {
    let result = build_with_residual_model(ResidualErrorModel::exponential(0.40).into())
        .expect("positive exponential declaration should build")
        .fit_with(
            SaemConfig::new()
                .seed(20_260_713)
                .n_chains(2)
                .mcmc_iterations(2)
                .burn_in(1)
                .k1_iterations(6)
                .k2_iterations(2)
                .compute_map(false),
        )
        .expect("positive-domain exponential fit should complete");

    let estimate = result
        .residual_error_estimate("cp")
        .expect("named exponential residual estimate should exist");
    let ResidualErrorModel::Exponential { sigma } = estimate.model else {
        panic!("expected exponential residual estimate");
    };
    assert!(estimate.estimated);
    assert!(sigma.is_finite() && sigma > 0.0);
    assert!(result.conditional_negative_log_likelihood().is_finite());
    assert_eq!(result.cycle_diagnostics().len(), 8);
}
