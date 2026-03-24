//! Replication of the experiments in Bayard & Neely (2017)
//! "Experiment Design for Nonparametric Models Based On Minimizing Bayes Risk"
//! J Pharmacokinet Pharmacodyn. 2017;44(2):95-111. PMCID: PMC5376526

use anyhow::Result;
use pmcore::mmopt::mmopt;
use pmcore::prelude::*;
use pmcore::structs::theta::Theta;

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
    section4()?;
    println!();
    section6()?;
    Ok(())
}

/// Paper Section 4: Two-support-point exponential decay example
///
/// Model: μ(t,a) = e^{-at}  (implemented as 1-compartment with D=V=1)
/// Support points: a1 = 1.5 (fast), a2 = 0.25 (slow)
/// Uniform priors: p1 = p2 = 0.5
/// Error: σ = 0.3 (constant additive)
/// Candidate times: 0.1 to 5.0 hours at 0.1-hour intervals
///
/// Analytical optimum: t* = ln(6)/1.25 ≈ 1.4334 hours
fn section4() -> Result<()> {
    println!("=== Section 4: Two-support-point example ===\n");

    let eq = one_comp_model();
    let params = Parameters::new().add("ke", 0.1, 5.0).add("v", 0.5, 2.0);

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 1.5,  // a1 (fast)
        (0, 1) => 1.0,  // V = 1
        (1, 0) => 0.25, // a2 (slow)
        (1, 1) => 1.0,  // V = 1
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params)?;

    let errormodel = ErrorModel::additive(ErrorPoly::new(0.3, 0.0, 0.0, 0.0), 0.0);

    // Candidate times: 0.1 to 5.0 at 0.1h steps
    let mut builder = Subject::builder("section4");
    builder = builder.bolus(0.0, 1.0, 0);
    for i in 1..=50 {
        builder = builder.missing_observation(i as f64 * 0.1, 0);
    }
    let subject = builder.build();

    let weights = vec![0.5, 0.5];
    let analytical = (6.0_f64).ln() / 1.25;

    let result = mmopt(&theta, &subject, eq, errormodel, 0, 1, weights)?;

    println!(
        "  Analytical optimum:     t* = ln(6)/1.25 = {:.4} h",
        analytical
    );
    println!("  MMopt optimal time:     t  = {:.4} h", result.times[0]);
    println!("  Bayes risk (overbound): {:.6}", result.risk);

    Ok(())
}

/// Paper Section 6: PK example with 10 support points
///
/// Model: one-compartment, dx/dt = d(t) - K*x, y = x/V
/// Dose: 300 units infused over 1 hour (rate = 300/hr)
/// Error: σ = 0.1 (constant additive)
/// 10 support points from Table 6.1 with equal priors (p_i = 0.1)
/// Candidate times: 0.25 to 24.0 hours at 0.25-hour intervals
///
/// Paper results (Table 6.2):
///   n=1: t* = {4.25},       Bayes Risk = 0.5474
///   n=2: t* = {1.0, 9.5},   Bayes Risk = 0.2947
///   n=3: t* = {1.0, 1.0, 10.5}, Bayes Risk = 0.2325
fn section6() -> Result<()> {
    println!("=== Section 6: PK example (10 support points, Table 6.1) ===\n");

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

    // 1-hour infusion of 300 units; candidate times 0.25 to 24h at 0.25h steps
    let mut builder = Subject::builder("section6");
    builder = builder.infusion(0.0, 300.0, 0, 1.0);
    for i in 1..=96 {
        builder = builder.missing_observation(i as f64 * 0.25, 0);
    }
    let subject = builder.build();

    let weights = vec![0.1; 10];

    // --- 1-sample design ---
    let r1 = mmopt(
        &theta,
        &subject,
        eq.clone(),
        errormodel.clone(),
        0,
        1,
        weights.clone(),
    )?;
    println!("  1-sample design:");
    println!("    Paper:  t* = {{4.25}},       Bayes Risk = 0.5474");
    println!(
        "    MMopt:  t* = {{{:.2}}},       Bayes risk = {:.6}",
        r1.times[0], r1.risk
    );

    // --- 2-sample design ---
    let r2 = mmopt(
        &theta,
        &subject,
        eq.clone(),
        errormodel.clone(),
        0,
        2,
        weights.clone(),
    )?;
    println!("\n  2-sample design:");
    println!("    Paper:  t* = {{1.0, 9.5}},   Bayes Risk = 0.2947");
    println!(
        "    MMopt:  t* = {{{:.2}, {:.2}}},  Bayes risk = {:.6}",
        r2.times[0], r2.times[1], r2.risk
    );

    // --- 3-sample design ---
    let r3 = mmopt(&theta, &subject, eq, errormodel, 0, 3, weights)?;
    println!("\n  3-sample design:");
    println!("    Paper:  t* = {{1.0, 1.0, 10.5}}, Bayes Risk = 0.2325");
    println!(
        "    MMopt:  t* = {{{:.2}, {:.2}, {:.2}}}, Bayes risk = {:.6}",
        r3.times[0], r3.times[1], r3.times[2], r3.risk
    );

    println!(
        "\n  Risk reduction: 1→2 samples: {:.1}%, 2→3 samples: {:.1}%",
        (1.0 - r2.risk / r1.risk) * 100.0,
        (1.0 - r3.risk / r2.risk) * 100.0,
    );

    Ok(())
}
