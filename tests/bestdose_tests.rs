//! Integration tests for the BestDose API.
//!
//! Covers the two pieces of the refactored flow:
//! 1. Computing a patient-specific parameter distribution with the NCNPAG and
//!    NPMAP algorithms.
//! 2. Optimizing doses against a distribution with [`BestDoseProblem`].

use anyhow::Result;
use pmcore::bestdose::{BestDoseOptions, BestDoseProblem, DoseRange, Target};
use pmcore::prelude::*;

// ── Shared fixtures ─────────────────────────────────────────────────────────

/// One-compartment model with a bolus input: `C(t) = dose·exp(-ke·t) / v`.
fn bolus_model() -> ODE {
    ode! {
        name: "one_compartment_bolus",
        params: [ke, v],
        states: [central],
        outputs: [outeq_0],
        routes: [
            bolus(input_0) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[outeq_0] = x[central] / v;
        },
    }
}

/// One-compartment model with an infusion input.
fn infusion_model() -> ODE {
    ode! {
        name: "one_compartment_infusion",
        params: [ke, v],
        states: [central],
        outputs: [outeq_0],
        routes: [
            infusion(input_0) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[outeq_0] = x[central] / v;
        },
    }
}

fn parameter_space() -> ParameterSpace<BoundedParameter> {
    ParameterSpace::<BoundedParameter>::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0)
}

fn error_models() -> AssayErrorModels {
    AssayErrorModels::new()
        .add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.2, 0.0, 0.0), 0.0),
        )
        .unwrap()
}

/// Build a [`Theta`] from explicit `[ke, v]` support points.
fn theta(points: &[[f64; 2]]) -> Theta {
    let mat = faer::Mat::from_fn(points.len(), 2, |r, c| points[r][c]);
    Theta::from_parts(mat, parameter_space()).unwrap()
}

/// Analytic bolus concentration for a single support point.
fn conc(dose: f64, ke: f64, v: f64, t: f64) -> f64 {
    dose * (-ke * t).exp() / v
}

/// The parameter values of the highest-weight support point.
fn best_point(theta: &Theta, weights: &Weights) -> Vec<f64> {
    let m = theta.matrix();
    let mut best = 0usize;
    let mut best_w = f64::NEG_INFINITY;
    for i in 0..m.nrows() {
        if weights[i] > best_w {
            best_w = weights[i];
            best = i;
        }
    }
    (0..m.ncols()).map(|c| m[(best, c)]).collect()
}

// ── BestDoseProblem construction ────────────────────────────────────────────

#[test]
fn new_rejects_mismatched_weights() {
    let theta = theta(&[[0.3, 50.0], [0.5, 60.0]]);
    let weights = Weights::uniform(1); // one weight for two support points
    assert!(BestDoseProblem::new(bolus_model(), theta, weights).is_err());
}

// ── Concentration targeting ─────────────────────────────────────────────────

#[test]
fn concentration_single_point_hits_target() -> Result<()> {
    let (ke, v) = (0.3, 50.0);
    let problem = BestDoseProblem::new(bolus_model(), theta(&[[ke, v]]), Weights::uniform(1))?;

    let target_conc = 2.0;
    let target = Subject::builder("p")
        .bolus(0.0, 0.0, 0) // optimizable
        .observation(2.0, target_conc, 0)
        .build();

    let result = problem.optimize(
        target,
        Target::Concentration,
        DoseRange::new(0.0, 1000.0),
        0.0,
        BestDoseOptions::default(),
    )?;

    // With a single support point the target is hit exactly.
    let expected_dose = target_conc * v / (-ke * 2.0).exp();
    assert!(
        (result.doses()[0] - expected_dose).abs() < 1.0,
        "dose {} vs expected {}",
        result.doses()[0],
        expected_dose
    );

    let achievement = &result.achievements()[0];
    assert!((achievement.achieved - target_conc).abs() < 1e-3);
    assert!(result.cost() < 1e-6);
    Ok(())
}

#[test]
fn fixed_doses_are_preserved() -> Result<()> {
    let problem = BestDoseProblem::new(bolus_model(), theta(&[[0.3, 50.0]]), Weights::uniform(1))?;

    let target = Subject::builder("p")
        .bolus(0.0, 500.0, 0) // fixed
        .bolus(12.0, 0.0, 0) // optimizable
        .observation(14.0, 2.0, 0)
        .build();

    let result = problem.optimize(
        target,
        Target::Concentration,
        DoseRange::new(0.0, 1000.0),
        0.0,
        BestDoseOptions::default(),
    )?;

    let doses = result.doses();
    assert_eq!(doses.len(), 2);
    assert_eq!(doses[0], 500.0, "fixed dose must be preserved");
    assert!(doses[1] > 0.0 && doses[1].is_finite());
    Ok(())
}

#[test]
fn dose_range_is_respected() -> Result<()> {
    let problem = BestDoseProblem::new(bolus_model(), theta(&[[0.3, 50.0]]), Weights::uniform(1))?;

    // The unconstrained optimum (~182 mg) lies above the allowed range.
    let target = Subject::builder("p")
        .bolus(0.0, 0.0, 0)
        .observation(2.0, 2.0, 0)
        .build();

    let result = problem.optimize(
        target,
        Target::Concentration,
        DoseRange::new(50.0, 150.0),
        0.0,
        BestDoseOptions::default(),
    )?;

    let dose = result.doses()[0];
    assert!(
        dose > 140.0 && dose <= 150.0 + 1e-6,
        "dose {} should be clamped to the upper bound",
        dose
    );
    Ok(())
}

#[test]
fn all_fixed_doses_return_unchanged() -> Result<()> {
    let problem = BestDoseProblem::new(bolus_model(), theta(&[[0.3, 50.0]]), Weights::uniform(1))?;

    let target = Subject::builder("p")
        .bolus(0.0, 100.0, 0) // fixed, nothing to optimize
        .observation(2.0, 1.0, 0)
        .build();

    let result = problem.optimize(
        target,
        Target::Concentration,
        DoseRange::new(0.0, 1000.0),
        0.0,
        BestDoseOptions::default(),
    )?;

    assert_eq!(result.doses(), vec![100.0]);
    assert_eq!(result.achievements().len(), 1);
    assert!(result.cost().is_finite());
    Ok(())
}

#[test]
fn infusions_are_optimizable() -> Result<()> {
    let problem =
        BestDoseProblem::new(infusion_model(), theta(&[[0.3, 50.0]]), Weights::uniform(1))?;

    let target = Subject::builder("p")
        .infusion(0.0, 0.0, 0, 1.0) // optimizable one-hour infusion
        .observation(2.0, 2.0, 0)
        .build();

    let result = problem.optimize(
        target,
        Target::Concentration,
        DoseRange::new(0.0, 2000.0),
        0.0,
        BestDoseOptions::default(),
    )?;

    let doses = result.doses();
    assert_eq!(doses.len(), 1);
    assert!(doses[0] > 0.0 && doses[0].is_finite());
    Ok(())
}

#[test]
fn achievements_cover_every_observation() -> Result<()> {
    let problem = BestDoseProblem::new(bolus_model(), theta(&[[0.3, 50.0]]), Weights::uniform(1))?;

    let target = Subject::builder("p")
        .bolus(0.0, 0.0, 0)
        .observation(2.0, 2.0, 0)
        .observation(4.0, 1.5, 0)
        .observation(6.0, 1.0, 0)
        .build();

    let result = problem.optimize(
        target,
        Target::Concentration,
        DoseRange::new(0.0, 1000.0),
        0.0,
        BestDoseOptions::default(),
    )?;

    let achievements = result.achievements();
    assert_eq!(achievements.len(), 3);
    assert_eq!(achievements[0].time, 2.0);
    assert_eq!(achievements[1].time, 4.0);
    assert_eq!(achievements[2].time, 6.0);
    for a in achievements {
        assert!(a.achieved.is_finite());
    }
    Ok(())
}

// ── AUC targeting ───────────────────────────────────────────────────────────

#[test]
fn auc_from_zero_hits_target() -> Result<()> {
    let problem = BestDoseProblem::new(bolus_model(), theta(&[[0.3, 50.0]]), Weights::uniform(1))?;

    let target_auc = 100.0;
    let target = Subject::builder("p")
        .bolus(0.0, 0.0, 0)
        .observation(12.0, target_auc, 0)
        .build();

    let result = problem.optimize(
        target,
        Target::AUCFromZero,
        DoseRange::new(0.0, 5000.0),
        0.0,
        BestDoseOptions {
            prediction_interval: 0.05,
        },
    )?;

    let achievement = &result.achievements()[0];
    let rel_error = ((achievement.achieved - target_auc) / target_auc).abs();
    assert!(
        rel_error < 0.02,
        "achieved AUC {} vs target {} (rel error {})",
        achievement.achieved,
        target_auc,
        rel_error
    );
    assert!(result.doses()[0] > 0.0);
    Ok(())
}

#[test]
fn auc_from_last_dose_optimizes_maintenance_dose() -> Result<()> {
    let problem = BestDoseProblem::new(bolus_model(), theta(&[[0.3, 50.0]]), Weights::uniform(1))?;

    let target = Subject::builder("p")
        .bolus(0.0, 200.0, 0) // fixed loading dose
        .bolus(12.0, 0.0, 0) // optimizable maintenance dose
        .observation(24.0, 40.0, 0) // target interval AUC (12–24 h)
        .build();

    let result = problem.optimize(
        target,
        Target::AUCFromLastDose,
        DoseRange::new(0.0, 2000.0),
        0.0,
        BestDoseOptions {
            prediction_interval: 0.05,
        },
    )?;

    let doses = result.doses();
    assert_eq!(doses.len(), 2);
    assert_eq!(doses[0], 200.0, "loading dose must be preserved");
    assert!(doses[1] > 0.0 && doses[1].is_finite());

    let achievement = &result.achievements()[0];
    assert_eq!(achievement.time, 24.0);
    assert!(achievement.achieved.is_finite() && achievement.achieved > 0.0);
    Ok(())
}

// ── Patient-specific posteriors (NCNPAG / NPMAP) ─────────────────────────────

/// Past observations generated from a known support point, used to individualize.
fn history_from(point: [f64; 2]) -> Subject {
    let [ke, v] = point;
    Subject::builder("history")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, conc(100.0, ke, v, 1.0), 0)
        .observation(3.0, conc(100.0, ke, v, 3.0), 0)
        .observation(6.0, conc(100.0, ke, v, 6.0), 0)
        .build()
}

#[test]
fn ncnpag_individualizes_toward_matching_point() -> Result<()> {
    let matching = [0.3, 50.0];
    let other = [1.5, 150.0];
    let prior = theta(&[matching, other]);

    let posterior = EstimationProblem::nonparametric(
        bolus_model(),
        Data::new(vec![history_from(matching)]),
        prior,
        error_models(),
    )?
    .fit_with(NcnpagConfig::default())?;

    // Weights remain a normalized distribution.
    let weight_sum: f64 = (0..posterior.weights().len())
        .map(|i| posterior.weights()[i])
        .sum();
    assert!((weight_sum - 1.0).abs() < 1e-6);

    // The dominant support point matches the data-generating parameters.
    let point = best_point(posterior.get_theta(), posterior.weights());
    assert!(
        (point[0] - matching[0]).abs() < 0.05,
        "expected ke ≈ {}, got {}",
        matching[0],
        point[0]
    );
    Ok(())
}

#[test]
fn npmap_reweights_toward_matching_point() -> Result<()> {
    let matching = [0.3, 50.0];
    let other = [1.5, 150.0];
    let prior = theta(&[matching, other]);

    let posterior = EstimationProblem::nonparametric(
        bolus_model(),
        Data::new(vec![history_from(matching)]),
        prior,
        error_models(),
    )?
    .fit_with(NpmapConfig::default())?;

    let point = best_point(posterior.get_theta(), posterior.weights());
    assert!(
        (point[0] - matching[0]).abs() < 0.05,
        "expected ke ≈ {}, got {}",
        matching[0],
        point[0]
    );
    Ok(())
}

/// End-to-end: individualize with NCNPAG, then optimize doses against the
/// resulting posterior.
#[test]
fn ncnpag_posterior_feeds_dose_optimization() -> Result<()> {
    let matching = [0.3, 50.0];
    let other = [1.2, 120.0];
    let prior = theta(&[matching, other]);

    let posterior = EstimationProblem::nonparametric(
        bolus_model(),
        Data::new(vec![history_from(matching)]),
        prior,
        error_models(),
    )?
    .fit_with(NcnpagConfig::default())?;

    let problem = BestDoseProblem::new(
        bolus_model(),
        posterior.get_theta().clone(),
        posterior.weights().clone(),
    )?;

    // Target the same profile the matching patient produced at 150 mg.
    let target = Subject::builder("target")
        .bolus(0.0, 0.0, 0)
        .observation(2.0, conc(150.0, matching[0], matching[1], 2.0), 0)
        .build();

    let result = problem.optimize(
        target,
        Target::Concentration,
        DoseRange::new(0.0, 1000.0),
        0.0,
        BestDoseOptions::default(),
    )?;

    // The optimizer should recover the 150 mg dose.
    assert!(
        (result.doses()[0] - 150.0).abs() < 5.0,
        "expected ~150 mg, got {}",
        result.doses()[0]
    );
    Ok(())
}
