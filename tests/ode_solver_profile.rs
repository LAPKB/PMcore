use pharmsol::prelude::*;
use pharmsol::Predictions;

const RELEASE_RTOL: f64 = 1e-8;
const RELEASE_ATOL: f64 = 1e-10;

fn configure(model: equation::ODE) -> equation::ODE {
    model
        .with_solver(OdeSolver::Bdf)
        .with_tolerances(RELEASE_RTOL, RELEASE_ATOL)
}

fn one_compartment() -> equation::ODE {
    configure(ode! {
        name: "ode_solver_profile_one_compartment",
        params: [ke],
        states: [central],
        outputs: [amount],
        routes: [
            bolus(iv_bolus) -> central,
            infusion(iv_infusion) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[amount] = x[central];
        },
    })
}

fn two_compartment() -> equation::ODE {
    configure(ode! {
        name: "ode_solver_profile_two_compartment",
        params: [k10, k12, k21],
        states: [central, peripheral],
        outputs: [central_amount, peripheral_amount],
        routes: [
            bolus(iv_bolus) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -(k10 + k12) * x[central] + k21 * x[peripheral];
            dx[peripheral] = k12 * x[central] - k21 * x[peripheral];
        },
        out: |x, _t, y| {
            y[central_amount] = x[central];
            y[peripheral_amount] = x[peripheral];
        },
    })
}

fn within_d1(actual: f64, expected: f64) -> bool {
    let absolute = (actual - expected).abs();
    let relative = absolute / expected.abs().max(f64::MIN_POSITIVE);
    absolute <= 1e-6 || relative <= 1e-4
}

fn assert_d1(label: &str, actual: f64, expected: f64) {
    assert!(actual.is_finite(), "{label}: actual is non-finite");
    assert!(expected.is_finite(), "{label}: oracle is non-finite");
    assert!(
        within_d1(actual, expected),
        "{label}: actual={actual:.16e}, expected={expected:.16e}, absolute={:.6e}, relative={:.6e}",
        (actual - expected).abs(),
        (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE),
    );
}

fn predictions(
    label: &str,
    model: &equation::ODE,
    subject: &Subject,
    parameters: &[f64],
) -> Vec<Prediction> {
    model
        .estimate_predictions_dense(subject, parameters)
        .unwrap_or_else(|error| panic!("{label}: solver-profile prediction failed: {error}"))
        .get_predictions()
}

#[test]
fn one_compartment_scale_and_time_panel_meets_d1() {
    let cases: &[(f64, f64, &[f64])] = &[
        (1e-4, 1e-4, &[0.01, 1.0, 100.0, 10_000.0]),
        (100.0, 0.1, &[0.001, 0.1, 1.0, 10.0, 100.0]),
        (1e6, 20.0, &[1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0]),
    ];

    for (case_index, &(dose, ke, times)) in cases.iter().enumerate() {
        let mut builder = Subject::builder(format!("s1-{case_index}")).bolus(0.0, dose, "iv_bolus");
        for &time in times {
            builder = builder.observation(time, 0.0, "amount");
        }
        let subject = builder.build();
        let release = predictions(
            &format!("S1 case {case_index} release"),
            &one_compartment(),
            &subject,
            &[ke],
        );
        assert_eq!(release.len(), times.len());

        for (index, (release_point, &time)) in release.iter().zip(times).enumerate() {
            let expected = dose * (-ke * time).exp();
            assert_eq!(release_point.time().to_bits(), time.to_bits());
            assert_eq!(release_point.outeq(), 0);
            assert_eq!(release_point.state().len(), 1);
            assert_d1(
                &format!("S1 case {case_index} point {index} release output"),
                release_point.prediction(),
                expected,
            );
            assert_d1(
                &format!("S1 case {case_index} point {index} release state"),
                release_point.state()[0],
                expected,
            );
        }
    }
}

fn bolus_contribution(amount: f64, event_time: f64, time: f64, ke: f64) -> f64 {
    if time <= event_time {
        0.0
    } else {
        amount * (-ke * (time - event_time)).exp()
    }
}

fn infusion_contribution(amount: f64, start: f64, duration: f64, time: f64, ke: f64) -> f64 {
    if time <= start {
        return 0.0;
    }
    let rate = amount / duration;
    let elapsed_infusion = (time - start).min(duration);
    let amount_at_elapsed = rate / ke * (1.0 - (-ke * elapsed_infusion).exp());
    if time <= start + duration {
        amount_at_elapsed
    } else {
        amount_at_elapsed * (-ke * (time - start - duration)).exp()
    }
}

#[test]
fn event_driven_bolus_and_infusion_panel_meets_d1() {
    let ke = 0.17;
    let times = [0.5, 3.5, 6.5, 7.5, 9.5, 20.0];
    let mut builder = Subject::builder("s2")
        .bolus(0.0, 100.0, "iv_bolus")
        .infusion(3.0, 80.0, "iv_infusion", 4.0)
        .bolus(9.0, 25.0, "iv_bolus");
    for time in times {
        builder = builder.observation(time, 0.0, "amount");
    }
    let subject = builder.build();
    let release = predictions("S2 release", &one_compartment(), &subject, &[ke]);
    assert_eq!(release.len(), times.len());

    for (index, (release_point, time)) in release.iter().zip(times).enumerate() {
        let expected = bolus_contribution(100.0, 0.0, time, ke)
            + infusion_contribution(80.0, 3.0, 4.0, time, ke)
            + bolus_contribution(25.0, 9.0, time, ke);
        assert_eq!(release_point.time().to_bits(), time.to_bits());
        assert_eq!(release_point.outeq(), 0);
        assert_eq!(release_point.state().len(), 1);
        assert_d1(
            &format!("S2 point {index} release output"),
            release_point.prediction(),
            expected,
        );
        assert_d1(
            &format!("S2 point {index} release state"),
            release_point.state()[0],
            expected,
        );
    }
}

fn two_compartment_closed_form(dose: f64, k10: f64, k12: f64, k21: f64, time: f64) -> [f64; 2] {
    let sum = k10 + k12 + k21;
    let discriminant = (sum * sum - 4.0 * k10 * k21).sqrt();
    let alpha = 0.5 * (sum + discriminant);
    let beta = 0.5 * (sum - discriminant);
    let denominator = alpha - beta;
    let fast = (-alpha * time).exp();
    let slow = (-beta * time).exp();
    let central = dose * ((alpha - k21) * fast + (k21 - beta) * slow) / denominator;
    let peripheral = dose * k12 * (slow - fast) / denominator;
    [central, peripheral]
}

#[test]
fn two_compartment_stiffness_panel_meets_d1() {
    let cases: &[(f64, f64, f64, &[f64])] = &[
        (0.1, 0.2, 0.15, &[0.001, 0.1, 1.0, 5.0, 20.0, 100.0]),
        (0.05, 25.0, 0.1, &[1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 20.0]),
        (2.0, 0.02, 30.0, &[1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 5.0]),
    ];
    let dose = 100.0;

    for (case_index, &(k10, k12, k21, times)) in cases.iter().enumerate() {
        let mut builder = Subject::builder(format!("s3-{case_index}")).bolus(0.0, dose, "iv_bolus");
        for &time in times {
            builder = builder
                .observation(time, 0.0, "central_amount")
                .observation(time, 0.0, "peripheral_amount");
        }
        let subject = builder.build();
        let parameters = [k10, k12, k21];
        let release = predictions(
            &format!("S3 case {case_index} release"),
            &two_compartment(),
            &subject,
            &parameters,
        );
        assert_eq!(release.len(), 2 * times.len());

        for (index, release_point) in release.iter().enumerate() {
            let time_index = index / 2;
            let output_index = index % 2;
            let time = times[time_index];
            let expected = two_compartment_closed_form(dose, k10, k12, k21, time);
            assert_eq!(release_point.time().to_bits(), time.to_bits());
            assert_eq!(release_point.outeq(), output_index);
            assert_eq!(release_point.state().len(), 2);
            assert_d1(
                &format!("S3 case {case_index} point {index} release output"),
                release_point.prediction(),
                expected[output_index],
            );
            for (state_index, expected_state) in expected.iter().copied().enumerate() {
                assert_d1(
                    &format!("S3 case {case_index} point {index} release state {state_index}"),
                    release_point.state()[state_index],
                    expected_state,
                );
            }
        }
    }
}
