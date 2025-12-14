//! Compare NPSAH vs NPSAH2 on multiple scenarios
//!
//! This example runs both algorithms on several test cases to evaluate improvements:
//! 1. Bimodal distribution (bimodal_ke) - Tests ability to find multiple modes (2 params)
//! 2. Two-compartment with lag (two_eq_lag) - Tests convergence with lag time (4 params)
//! 3. Theophylline (theophylline) - Standard PK model (3 params)
//! 4. Neely model (neely) - Complex multi-output model (10 params)
//!
//! Run with: cargo run --release --example compare_npsah

use anyhow::Result;
use pmcore::prelude::*;
use std::time::Instant;

// ============================================================================
// TEST CASE 1: Bimodal Distribution (2 parameters)
// ============================================================================

fn create_bimodal_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    )
}

fn create_bimodal_settings(algorithm: Algorithm) -> Settings {
    let params = Parameters::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);

    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )
        .unwrap()
        .add(1, ErrorModel::None)
        .unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(algorithm)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(500);
    settings.set_prior(Prior::sobol(2028, 22));
    settings.set_write_logs(false);
    settings
}

// ============================================================================
// TEST CASE 2: Two-compartment with lag (4 parameters)
// ============================================================================

fn create_two_eq_lag_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ka, ke, _tlag, _v);
            dx[0] = -ka * x[0] + b[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0 => tlag}
        },
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    )
}

fn create_two_eq_lag_settings(algorithm: Algorithm) -> Settings {
    let params = Parameters::new()
        .add("ka", 0.1, 0.9)
        .add("ke", 0.001, 0.1)
        .add("tlag", 0.0, 4.0)
        .add("v", 30.0, 120.0);

    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::additive(ErrorPoly::new(-0.00119, 0.44379, -0.45864, 0.16537), 0.0),
        )
        .unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(algorithm)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(500);
    settings.set_prior(Prior::sobol(1234, 30));
    settings.set_write_logs(false);
    settings
}

// ============================================================================
// TEST CASE 3: Theophylline (3 parameters)
// ============================================================================

fn create_theo_equation() -> equation::Analytical {
    equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] * 1000.0 / v;
        },
        (2, 1),
    )
}

fn create_theo_settings(algorithm: Algorithm) -> Settings {
    let params = Parameters::new()
        .add("ka", 0.001, 3.0)
        .add("ke", 0.001, 3.0)
        .add("v", 0.001, 50.0);

    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::proportional(ErrorPoly::new(0.1, 0.1, 0.0, 0.0), 2.0),
        )
        .unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(algorithm)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(500);
    settings.set_prior(Prior::sobol(1234, 30));
    settings.set_write_logs(false);
    settings
}

// ============================================================================
// TEST CASE 4: Neely model (10 parameters, complex)
// ============================================================================

fn create_neely_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, t, dx, b, rateiv, cov| {
            fetch_params!(p, cls, k30, k40, qs, vps, vs, fm1, fm2, theta1, theta2);
            fetch_cov!(cov, t, wt, pkvisit);

            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let q = qs * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let vp = vps * (wt / 70.0);
            let ke = cl / v;
            let k12 = q / v;
            let k21 = q / vp;

            dx[0] = rateiv[0] - ke * x[0] * (1.0 - fm1 - fm2) - (fm1 + fm2) * x[0] - k12 * x[0]
                + k21 * x[1]
                + b[0];
            dx[1] = k12 * x[0] - k21 * x[1];
            dx[2] = fm1 * x[0] - k30 * x[2];
            dx[3] = fm2 * x[0] - k40 * x[3];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_params!(p, cls, _k30, _k40, qs, vps, vs, _fm1, _fm2, theta1, theta2);
            fetch_cov!(cov, t, wt, pkvisit);

            let vfrac1 = 0.068202;
            let vfrac2 = 0.022569;
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let vm1 = vfrac1 * v;
            let vm2 = vfrac2 * v;

            y[0] = x[0] / v;
            y[1] = x[2] / vm1;
            y[2] = x[3] / vm2;
        },
        (4, 3),
    )
}

fn create_neely_settings(algorithm: Algorithm) -> Settings {
    let params = Parameters::new()
        .add("cls", 0.0, 0.4)
        .add("k30", 0.0, 0.5)
        .add("k40", 0.3, 1.5)
        .add("qs", 0.0, 0.5)
        .add("vps", 0.0, 5.0)
        .add("vs", 0.0, 2.0)
        .add("fm1", 0.0, 0.2)
        .add("fm2", 0.0, 0.1)
        .add("theta1", -4.0, 2.0)
        .add("theta2", -2.0, 0.5);

    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap()
        .add(
            1,
            ErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap()
        .add(
            2,
            ErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(algorithm)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(500);
    settings.set_prior(Prior::sobol(2028, 22));
    settings.set_write_logs(false);
    settings
}

// ============================================================================
// RUN RESULT STRUCT
// ============================================================================

struct RunResult {
    name: String,
    scenario: String,
    n_params: usize,
    objf: f64,
    n_spp: usize,
    cycles: usize,
    duration: std::time::Duration,
}

// ============================================================================
// GENERIC RUNNER
// ============================================================================

fn run_scenario<E: pharmsol::prelude::simulator::Equation + Send + 'static>(
    name: &str,
    scenario: &str,
    n_params: usize,
    equation: E,
    settings: Settings,
    data: Data,
) -> Result<RunResult> {
    println!("  Running {} on {}...", name, scenario);
    let start = Instant::now();
    let mut alg = dispatch_algorithm(settings, equation, data)?;
    let result = match alg.fit() {
        Ok(r) => r,
        Err(e) => {
            println!("  [ERROR] {} on {} failed: {:?}", name, scenario, e);
            return Err(e);
        }
    };
    let duration = start.elapsed();

    Ok(RunResult {
        name: name.to_string(),
        scenario: scenario.to_string(),
        n_params,
        objf: result.objf(),
        n_spp: result.get_theta().nspp(),
        cycles: result.cycles(),
        duration,
    })
}

fn print_divider() {
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════"
    );
}

fn print_section(title: &str) {
    println!();
    print_divider();
    println!("  {}", title);
    print_divider();
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!();
    println!(
        "╔═══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                         NPSAH vs NPSAH2 COMPREHENSIVE BENCHMARK                       ║"
    );
    println!(
        "║                                                                                       ║"
    );
    println!(
        "║  Testing algorithm improvements across models of varying complexity                   ║"
    );
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    let mut all_results: Vec<RunResult> = Vec::new();

    // ========================================================================
    // SCENARIO 1: Bimodal Distribution (2 params)
    // ========================================================================
    print_section("SCENARIO 1: Bimodal Distribution (2 params)");
    println!("  Challenge: Find two distinct modes in ke parameter");
    println!();

    let data_path = "examples/bimodal_ke/bimodal_ke.csv";
    if std::path::Path::new(data_path).exists() {
        let data = data::read_pmetrics(data_path)?;

        if let Ok(r) = run_scenario(
            "NPSAH",
            "bimodal_ke",
            2,
            create_bimodal_equation(),
            create_bimodal_settings(Algorithm::NPSAH),
            data.clone(),
        ) {
            all_results.push(r);
        }
        if let Ok(r) = run_scenario(
            "NPSAH2",
            "bimodal_ke",
            2,
            create_bimodal_equation(),
            create_bimodal_settings(Algorithm::NPSAH2),
            data,
        ) {
            all_results.push(r);
        }
    } else {
        println!("  [SKIPPED] Data file not found: {}", data_path);
    }

    // ========================================================================
    // SCENARIO 2: Two-compartment with lag (4 params)
    // ========================================================================
    print_section("SCENARIO 2: Two-compartment with Lag (4 params)");
    println!("  Challenge: Handle absorption lag time parameter");
    println!();

    let data_path = "examples/two_eq_lag/two_eq_lag.csv";
    if std::path::Path::new(data_path).exists() {
        let data = data::read_pmetrics(data_path)?;

        if let Ok(r) = run_scenario(
            "NPSAH",
            "two_eq_lag",
            4,
            create_two_eq_lag_equation(),
            create_two_eq_lag_settings(Algorithm::NPSAH),
            data.clone(),
        ) {
            all_results.push(r);
        }
        if let Ok(r) = run_scenario(
            "NPSAH2",
            "two_eq_lag",
            4,
            create_two_eq_lag_equation(),
            create_two_eq_lag_settings(Algorithm::NPSAH2),
            data,
        ) {
            all_results.push(r);
        }
    } else {
        println!("  [SKIPPED] Data file not found: {}", data_path);
    }

    // ========================================================================
    // SCENARIO 3: Theophylline (3 params)
    // ========================================================================
    print_section("SCENARIO 3: Theophylline (3 params)");
    println!("  Challenge: Standard PK with analytical solution");
    println!();

    let data_path = "examples/theophylline/theophylline.csv";
    if std::path::Path::new(data_path).exists() {
        let data = data::read_pmetrics(data_path)?;

        if let Ok(r) = run_scenario(
            "NPSAH",
            "theophylline",
            3,
            create_theo_equation(),
            create_theo_settings(Algorithm::NPSAH),
            data.clone(),
        ) {
            all_results.push(r);
        }
        if let Ok(r) = run_scenario(
            "NPSAH2",
            "theophylline",
            3,
            create_theo_equation(),
            create_theo_settings(Algorithm::NPSAH2),
            data,
        ) {
            all_results.push(r);
        }
    } else {
        println!("  [SKIPPED] Data file not found: {}", data_path);
    }

    // ========================================================================
    // SCENARIO 4: Neely model (10 params)
    // ========================================================================
    print_section("SCENARIO 4: Neely Model (10 params)");
    println!("  Challenge: High-dimensional parameter space, multiple outputs");
    println!();

    let data_path = "examples/neely/data.csv";
    if std::path::Path::new(data_path).exists() {
        let data = data::read_pmetrics(data_path)?;

        if let Ok(r) = run_scenario(
            "NPSAH",
            "neely",
            10,
            create_neely_equation(),
            create_neely_settings(Algorithm::NPSAH),
            data.clone(),
        ) {
            all_results.push(r);
        }
        if let Ok(r) = run_scenario(
            "NPSAH2",
            "neely",
            10,
            create_neely_equation(),
            create_neely_settings(Algorithm::NPSAH2),
            data,
        ) {
            all_results.push(r);
        }
    } else {
        println!("  [SKIPPED] Data file not found: {}", data_path);
    }

    // ========================================================================
    // SUMMARY TABLE
    // ========================================================================
    print_section("SUMMARY RESULTS");

    println!();
    println!(
        "┌───────────┬──────────────┬────────┬──────────────┬────────────┬────────┬──────────────┐"
    );
    println!(
        "│ Algorithm │ Scenario     │ Params │     -2LL     │ Support Pts│ Cycles │    Time      │"
    );
    println!(
        "├───────────┼──────────────┼────────┼──────────────┼────────────┼────────┼──────────────┤"
    );

    for result in &all_results {
        println!(
            "│ {:9} │ {:12} │ {:6} │ {:12.4} │ {:10} │ {:6} │ {:10.2?} │",
            result.name,
            result.scenario,
            result.n_params,
            result.objf,
            result.n_spp,
            result.cycles,
            result.duration
        );
    }
    println!(
        "└───────────┴──────────────┴────────┴──────────────┴────────────┴────────┴──────────────┘"
    );

    // ========================================================================
    // COMPARATIVE ANALYSIS BY SCENARIO
    // ========================================================================
    print_section("COMPARATIVE ANALYSIS");

    let scenarios: Vec<&str> = all_results
        .iter()
        .map(|r| r.scenario.as_str())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut npsah_wins = 0;
    let mut npsah2_wins = 0;
    let mut total_speedup = 0.0;
    let mut total_objf_improvement = 0.0;
    let mut comparison_count = 0;

    for scenario in &scenarios {
        let npsah = all_results
            .iter()
            .find(|r| r.name == "NPSAH" && r.scenario == *scenario);
        let npsah2 = all_results
            .iter()
            .find(|r| r.name == "NPSAH2" && r.scenario == *scenario);

        if let (Some(r1), Some(r2)) = (npsah, npsah2) {
            println!();
            println!("  {} ({} params):", scenario, r1.n_params);
            println!("  ─────────────────────────────────────────────────────");

            let objf_diff = r1.objf - r2.objf;
            let time_ratio = r1.duration.as_secs_f64() / r2.duration.as_secs_f64();

            let better_objf = if objf_diff > 0.001 {
                "NPSAH2"
            } else if objf_diff < -0.001 {
                "NPSAH"
            } else {
                "TIE"
            };

            println!(
                "    -2LL:    NPSAH={:.4}, NPSAH2={:.4} → {} wins",
                r1.objf, r2.objf, better_objf
            );

            if time_ratio > 1.0 {
                println!("    Speed:   NPSAH2 is {:.2}x FASTER", time_ratio);
            } else {
                println!("    Speed:   NPSAH2 is {:.2}x slower", 1.0 / time_ratio);
            }

            println!("    Cycles:  NPSAH={}, NPSAH2={}", r1.cycles, r2.cycles);
            println!("    SPPs:    NPSAH={}, NPSAH2={}", r1.n_spp, r2.n_spp);

            // Track wins
            if objf_diff > 0.001 {
                npsah2_wins += 1;
            } else if objf_diff < -0.001 {
                npsah_wins += 1;
            }

            total_speedup += time_ratio;
            total_objf_improvement += objf_diff;
            comparison_count += 1;
        }
    }

    // ========================================================================
    // OVERALL SUMMARY
    // ========================================================================
    print_section("OVERALL SUMMARY");

    if comparison_count > 0 {
        let avg_speedup = total_speedup / comparison_count as f64;
        let avg_objf_improvement = total_objf_improvement / comparison_count as f64;

        println!();
        println!("  Scenarios compared: {}", comparison_count);
        println!();
        println!(
            "  -2LL Wins:     NPSAH={}, NPSAH2={}, Ties={}",
            npsah_wins,
            npsah2_wins,
            comparison_count - npsah_wins - npsah2_wins
        );
        println!();
        if avg_speedup > 1.0 {
            println!(
                "  Avg Speed:     NPSAH2 is {:.2}x FASTER on average",
                avg_speedup
            );
        } else {
            println!(
                "  Avg Speed:     NPSAH2 is {:.2}x slower on average",
                1.0 / avg_speedup
            );
        }
        println!();
        if avg_objf_improvement > 0.0 {
            println!(
                "  Avg -2LL:      NPSAH2 finds {:.4} BETTER solutions on average",
                avg_objf_improvement
            );
        } else {
            println!(
                "  Avg -2LL:      NPSAH finds {:.4} better solutions on average",
                -avg_objf_improvement
            );
        }
        println!();

        // Final verdict
        let speed_verdict = if avg_speedup > 1.1 {
            "faster"
        } else if avg_speedup < 0.9 {
            "slower"
        } else {
            "similar speed"
        };
        let quality_verdict = if avg_objf_improvement > 0.01 {
            "better solutions"
        } else if avg_objf_improvement < -0.01 {
            "worse solutions"
        } else {
            "similar quality"
        };

        println!(
            "  ╔═════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "  ║ VERDICT: NPSAH2 is {} and finds {}              ║",
            speed_verdict, quality_verdict
        );
        println!(
            "  ╚═════════════════════════════════════════════════════════════════════════════╝"
        );
    }

    println!();
    Ok(())
}
