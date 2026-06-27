//! Example: Compute Bayes risk for a given sampling design
//!
//! Uses the same PK model and support points as Section 6 of Bayard & Neely (2017).
//! Instead of optimizing sample times, this calculates the Bayes risk for
//! user-specified observation times.

use anyhow::Result;
use pmcore::mmopt::bayes_risk;
use pmcore::prelude::*;
use pmcore::structs::theta::Theta;
use pmcore::structs::weights::Weights;

/// One-compartment model: dx/dt = -K*x + input, y = x/V
fn one_comp_model() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0] + rateiv[0];
        },
        |_p, _, _| lag! {},
        |_p, _, _| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    )
}

fn main() -> Result<()> {
    let eq = one_comp_model();
    let params = Parameters::new().add("ke", 0.01, 0.2).add("v", 80.0, 120.0);

    // Table 6.1 support points [K, V]
    let support_points: [(f64, f64); 10] = [
        (0.090088, 113.7451),
        (0.111611, 93.4326),
        (0.066074, 90.2832),
        (0.108604, 89.2334),
        (0.103047, 112.1093),
        (0.033965, 94.3847),
        (0.100859, 109.8633),
        (0.023174, 111.7920),
        (0.087041, 108.6670),
        (0.095996, 100.3418),
    ];

    let mat = faer::Mat::from_fn(10, 2, |r, c| match c {
        0 => support_points[r].0,
        1 => support_points[r].1,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params)?;

    let errormodel = ErrorModel::additive(ErrorPoly::new(0.1, 0.0, 0.0, 0.0), 0.0);
    let weights = Weights::uniform(10);

    // --- Design A: Two observations at the MMopt-optimal times ---
    let subject_a = Subject::builder("design_a")
        .infusion(0.0, 300.0, 0, 1.0)
        .missing_observation(1.0, 0)
        .missing_observation(9.5, 0)
        .build();

    let risk_a = bayes_risk(
        &theta,
        &subject_a,
        eq.clone(),
        errormodel.clone(),
        0,
        &weights,
    )?;
    println!("Design A  t = {{1.0, 9.5}}      Bayes risk = {:.6}", risk_a);

    // --- Design B: Two observations at sub-optimal times ---
    let subject_b = Subject::builder("design_b")
        .infusion(0.0, 300.0, 0, 1.0)
        .missing_observation(2.0, 0)
        .missing_observation(6.0, 0)
        .build();

    let risk_b = bayes_risk(
        &theta,
        &subject_b,
        eq.clone(),
        errormodel.clone(),
        0,
        &weights,
    )?;
    println!(
        "Design B  t = {{2.0, 6.0}}          Bayes risk = {:.6}",
        risk_b
    );

    // --- Design C: B + one more sample ---
    let subject_c = Subject::builder("design_c")
        .infusion(0.0, 300.0, 0, 1.0)
        .missing_observation(2.0, 0)
        .missing_observation(6.0, 0)
        .missing_observation(12.0, 0)
        .build();

    let risk_c = bayes_risk(
        &theta,
        &subject_c,
        eq.clone(),
        errormodel.clone(),
        0,
        &weights,
    )?;
    println!(
        "Design C  t = {{2.0, 6.0, 12.0}}     Bayes risk = {:.6}",
        risk_c
    );

    // --- Design D: C + one more sample ---
    let subject_d = Subject::builder("design_d")
        .infusion(0.0, 300.0, 0, 1.0)
        .missing_observation(2.0, 0)
        .missing_observation(6.0, 0)
        .missing_observation(12.0, 0)
        .missing_observation(18.0, 0)
        .build();

    let risk_d = bayes_risk(&theta, &subject_d, eq, errormodel, 0, &weights)?;
    println!(
        "Design D  t = {{2.0, 6.0, 12.0, 18.0}} Bayes risk = {:.6}",
        risk_d
    );

    println!(
        "\nDesign A vs B: {:.1}% lower risk with optimal times",
        (1.0 - risk_a / risk_b) * 100.0
    );
    println!(
        "B → C (add 1 sample): {:.1}% risk reduction",
        (1.0 - risk_c / risk_b) * 100.0
    );
    println!(
        "C → D (add 1 sample): {:.1}% risk reduction",
        (1.0 - risk_d / risk_c) * 100.0
    );

    Ok(())
}
