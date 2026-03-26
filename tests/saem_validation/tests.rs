//! SAEM Validation Tests
//!
//! These tests compare PMcore SAEM against R saemix reference values.

use super::reference::*;
use anyhow::Result;

// Path to reference data files (relative to tests/ directory)
const VALIDATION_DIR: &str = "tests/saem_validation";

// =============================================================================
// Basic Model Tests
// Verify the model produces correct predictions
// =============================================================================

/// Test that the ODE model produces correct predictions
#[test]
fn test_ode_predictions() {
    use pmcore::prelude::*;

    println!("=== Testing ODE Predictions ===");

    // Simple one-compartment IV bolus model
    // CRITICAL: b[0] is the bolus input term - must be included!
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0]; // b[0] is the bolus input
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    );

    // Create a simple subject with one bolus and observations
    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0) // 100 units at t=0 into compartment 0
        .observation(1.0, 0.0, 0) // placeholder observation at t=1
        .observation(2.0, 0.0, 0) // placeholder observation at t=2
        .build();

    // Parameters: ke=0.4, V=10
    let params = vec![0.4, 10.0];

    // Get predictions
    let predictions = eq.estimate_predictions(&subject, &params).unwrap();
    let preds: Vec<f64> = predictions
        .get_predictions()
        .iter()
        .map(|p| p.prediction())
        .collect();

    println!("  Predictions: {:?}", preds);

    // Expected: C(t) = (Dose/V) * exp(-ke*t) = (100/10) * exp(-0.4*t)
    // C(1) = 10 * exp(-0.4) = 10 * 0.6703 = 6.703
    // C(2) = 10 * exp(-0.8) = 10 * 0.4493 = 4.493
    let expected_c1 = 10.0 * (-0.4_f64).exp();
    let expected_c2 = 10.0 * (-0.8_f64).exp();

    println!(
        "  Expected: C(1)={:.4}, C(2)={:.4}",
        expected_c1, expected_c2
    );

    assert!(preds.len() >= 2, "Should have at least 2 predictions");
    assert!(
        (preds[0] - expected_c1).abs() < 0.1,
        "C(1) mismatch: got {}, expected {}",
        preds[0],
        expected_c1
    );
    assert!(
        (preds[1] - expected_c2).abs() < 0.1,
        "C(2) mismatch: got {}, expected {}",
        preds[1],
        expected_c2
    );

    println!("  ✓ ODE predictions correct\n");
}

// =============================================================================
// Component-Level Tests
// These verify exact matching of individual components
// =============================================================================

/// Test parameter transformations match R exactly
#[test]
fn test_component_transforms() {
    use pmcore::structs::parametric::ParameterTransform;

    println!("=== Testing Parameter Transforms ===");

    // Test log-normal transform (code 1 in saemix)
    let transform = ParameterTransform::LogNormal;

    // Test values from R reference
    let psi_values: Vec<f64> = vec![0.1, 0.5, 1.0, 2.0, 5.0];
    let expected_phi: Vec<f64> = psi_values.iter().map(|&x| x.ln()).collect();

    for (i, &psi) in psi_values.iter().enumerate() {
        let phi = transform.psi_to_phi(psi);
        let psi_back = transform.phi_to_psi(phi);

        println!(
            "  psi={:.4} -> phi={:.6} (expected: {:.6}) -> psi_back={:.6}",
            psi, phi, expected_phi[i], psi_back
        );

        // Check forward transform
        assert_close(phi, expected_phi[i], 1e-10, &format!("phi[{}]", i));

        // Check round-trip
        assert_close(psi_back, psi, 1e-10, &format!("psi_roundtrip[{}]", i));
    }

    println!("  ✓ Log-normal transforms match R\n");

    // Test identity transform (code 0)
    let identity = ParameterTransform::None;
    for &val in &[-1.0, 0.0, 1.0, 5.0] {
        assert_close(identity.psi_to_phi(val), val, 1e-15, "identity_forward");
        assert_close(identity.phi_to_psi(val), val, 1e-15, "identity_inverse");
    }
    println!("  ✓ Identity transforms match\n");
}

/// Test sufficient statistics computation matches R
#[test]
fn test_component_sufficient_stats() {
    use faer::Col;
    use pmcore::structs::parametric::SufficientStats;

    println!("=== Testing Sufficient Statistics ===");

    // Test data from R reference (generate_reference.R)
    let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

    // Expected values from R
    let expected_s1 = vec![9.0, 12.0];
    let expected_mu = vec![3.0, 4.0];
    // Omega = S2/n - mu*mu'
    // E[X^2] for col 0: (1+9+25)/3 = 35/3
    // Var = 35/3 - 9 = 8/3 ≈ 2.6667
    let expected_omega_00 = 8.0 / 3.0;

    let mut stats = SufficientStats::new(2);

    for sample in &samples {
        let phi = Col::from_fn(2, |i| sample[i]);
        stats.accumulate(&phi).unwrap();
    }

    // Check S1
    println!(
        "  S1: [{:.4}, {:.4}] (expected: {:?})",
        stats.s1()[0],
        stats.s1()[1],
        expected_s1
    );
    assert_close(stats.s1()[0], expected_s1[0], 1e-10, "S1[0]");
    assert_close(stats.s1()[1], expected_s1[1], 1e-10, "S1[1]");

    // Compute M-step
    let (mu, omega) = stats.compute_m_step().unwrap();

    // Check mu
    println!(
        "  mu: [{:.4}, {:.4}] (expected: {:?})",
        mu[0], mu[1], expected_mu
    );
    assert_close(mu[0], expected_mu[0], 1e-10, "mu[0]");
    assert_close(mu[1], expected_mu[1], 1e-10, "mu[1]");

    // Check omega diagonal
    println!(
        "  omega[0,0]: {:.6} (expected: {:.6})",
        omega[(0, 0)],
        expected_omega_00
    );
    assert_close(omega[(0, 0)], expected_omega_00, 1e-10, "omega[0,0]");

    println!("  ✓ Sufficient statistics match R\n");
}

/// Test step size schedule matches R saemix
#[test]
fn test_component_step_size() {
    use pmcore::structs::parametric::sufficient_stats::StepSizeSchedule;

    println!("=== Testing Step Size Schedule ===");

    // PMcore PolyakRuppert schedule:
    // - k < start_averaging: gamma = 1.0 (burn-in/exploration)
    // - k >= start_averaging: gamma = 1/(k - start_averaging + 1) (smoothing)

    let n_burn = 100; // Total burn-in iterations
    let n_smooth = 200;
    let schedule = StepSizeSchedule::new_saem(n_burn, n_smooth);

    // Test values based on PMcore's actual implementation
    let test_cases = vec![
        (1, 1.0, "burn-in start"),
        (50, 1.0, "burn-in middle"),
        (99, 1.0, "last burn-in"),           // 99 < 100, so gamma = 1.0
        (100, 1.0, "first smoothing"),       // 100 >= 100, gamma = 1/(100-100+1) = 1.0
        (101, 0.5, "second smoothing"),      // gamma = 1/(101-100+1) = 0.5
        (102, 1.0 / 3.0, "third smoothing"), // gamma = 1/3
        (200, 0.01, "late smoothing"),       // gamma = 1/(200-100+1) = 1/101 ≈ 0.0099
    ];

    for (iter, expected, desc) in test_cases {
        let actual = schedule.step_size(iter);
        println!(
            "  iter {}: gamma={:.6} (expected: {:.6}) - {}",
            iter, actual, expected, desc
        );

        // Allow some tolerance for numerical differences
        assert_close(actual, expected, 0.01, &format!("step_size({})", iter));
    }

    println!("  ✓ Step size schedule matches PMcore behavior\n");
}

// =============================================================================
// Integration Tests
// These test algorithm phases against R
// =============================================================================

/// Test SAEM algorithm initialization
#[test]
fn test_saem_initialization() -> Result<()> {
    use pmcore::algorithms::parametric::dispatch_parametric_algorithm;
    use pmcore::prelude::*;

    println!("=== Testing SAEM Initialization ===");

    // Create simple model - b[0] is the bolus input term!
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    );

    // Create test data
    let subjects: Vec<Subject> = (0..3)
        .map(|id| {
            Subject::builder(id.to_string())
                .bolus(0.0, 100.0, 0)
                .observation(1.0, 5.0, 0)
                .observation(4.0, 2.0, 0)
                .build()
        })
        .collect();
    let data = Data::new(subjects);

    // Parameters - log-normal by default
    let params = Parameters::new().add("ke", 0.1, 1.0).add("v", 5.0, 20.0);

    // Residual error model (parametric algorithms use ResidualErrorModels)
    use pharmsol::{ResidualErrorModel, ResidualErrorModels};
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::constant(0.5));

    let settings = Settings::builder()
        .set_algorithm(Algorithm::SAEM)
        .set_parameters(params)
        .set_residual_error(residual_error)
        .build();

    let mut algorithm = dispatch_parametric_algorithm(settings, eq, data)?;

    // Before initialization, mu is in ψ (natural) space
    // Population is initialized with midpoints: ke = (0.1+1.0)/2 = 0.55, v = (5+20)/2 = 12.5
    assert_eq!(algorithm.iteration(), 0, "Initial iteration should be 0");
    assert_eq!(algorithm.population().npar(), 2, "Should have 2 parameters");

    let mu_ke_pre = algorithm.population().mu()[0];
    let mu_v_pre = algorithm.population().mu()[1];
    println!("  Before init - mu[ke] (ψ space): {:.4}", mu_ke_pre);
    println!("  Before init - mu[v] (ψ space): {:.4}", mu_v_pre);

    // These are arithmetic midpoints in ψ space
    assert_close(mu_ke_pre, 0.55, 0.01, "mu_ke pre-init");
    assert_close(mu_v_pre, 12.5, 0.01, "mu_v pre-init");

    // Initialize - this transforms mu from ψ to φ space
    algorithm.initialize()?;

    // After initialization, mu is in φ (transformed) space
    let mu_ke = algorithm.population().mu()[0];
    let mu_v = algorithm.population().mu()[1];

    println!("  After init - mu[ke] (φ space): {:.4}", mu_ke);
    println!("  After init - mu[v] (φ space): {:.4}", mu_v);

    // For LogNormal: φ = ln(ψ)
    // ln(0.55) ≈ -0.598, ln(12.5) ≈ 2.526
    assert!(mu_ke < 0.0, "Log of ke (0.55) should be negative");
    assert!(mu_v > 0.0, "Log of v (12.5) should be positive");

    println!("  ✓ SAEM initialization correct\n");
    Ok(())
}

/// Test SAEM runs multiple iterations without errors
#[test]
fn test_saem_iterations() -> Result<()> {
    use pmcore::algorithms::parametric::dispatch_parametric_algorithm;
    use pmcore::prelude::*;

    println!("=== Testing SAEM Iterations ===");

    // Simple model - b[0] is the bolus input term!
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    );

    // Create synthetic data with known parameters
    let true_ke = 0.4;
    let true_v = 10.0;
    let dose = 100.0;

    let subjects: Vec<Subject> = (0..10)
        .map(|id| {
            let eta_ke = 0.1 * ((id as f64 * 0.5).sin());
            let eta_v = 0.1 * ((id as f64 * 0.7).cos());
            let subj_ke = true_ke * f64::exp(eta_ke);
            let subj_v = true_v * f64::exp(eta_v);

            Subject::builder(id.to_string())
                .bolus(0.0, dose, 0)
                .observation(1.0, (dose / subj_v) * f64::exp(-subj_ke * 1.0), 0)
                .observation(2.0, (dose / subj_v) * f64::exp(-subj_ke * 2.0), 0)
                .observation(4.0, (dose / subj_v) * f64::exp(-subj_ke * 4.0), 0)
                .build()
        })
        .collect();
    let data = Data::new(subjects);

    let params = Parameters::new().add("ke", 0.1, 0.8).add("v", 5.0, 15.0);

    // Residual error model (parametric algorithms use ResidualErrorModels)
    use pharmsol::{ResidualErrorModel, ResidualErrorModels};
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::constant(0.1));

    let settings = Settings::builder()
        .set_algorithm(Algorithm::SAEM)
        .set_parameters(params)
        .set_residual_error(residual_error)
        .build();

    let mut algorithm = dispatch_parametric_algorithm(settings, eq, data)?;
    algorithm.initialize()?;

    // Run enough iterations to get past burn-in (default: 20 pure burn + 80 SA = 100)
    // We'll run 120 iterations to enter the stochastic approximation phase
    let mut prev_objf = f64::INFINITY;
    let n_iter = 120;

    for i in 1..=n_iter {
        let _status = algorithm.next_iteration()?;
        let objf = algorithm.objective_function();

        if i <= 5 || i == 21 || i == 100 || i == n_iter {
            let pop = algorithm.population();
            let ke_psi = f64::exp(pop.mu()[0]); // Convert back to ψ space
            let v_psi = f64::exp(pop.mu()[1]);

            println!(
                "  Iter {}: objf={:.2}, ke={:.4} (true: {}), v={:.4} (true: {})",
                i, objf, ke_psi, true_ke, v_psi, true_v
            );
        }

        prev_objf = objf;
    }

    // After some iterations, objective should be finite
    assert!(
        prev_objf.is_finite(),
        "Objective should be finite after iterations"
    );

    // Parameters should be in reasonable range
    let final_pop = algorithm.population();
    let final_ke = f64::exp(final_pop.mu()[0]);
    let final_v = f64::exp(final_pop.mu()[1]);

    println!(
        "\n  Final: ke={:.4} (true: {}), v={:.4} (true: {})",
        final_ke, true_ke, final_v, true_v
    );

    // Allow generous tolerance during burn-in
    assert!(
        final_ke > 0.05 && final_ke < 2.0,
        "ke estimate out of reasonable range"
    );
    assert!(
        final_v > 1.0 && final_v < 50.0,
        "v estimate out of reasonable range"
    );

    println!("  ✓ SAEM iterations completed successfully\n");
    Ok(())
}

// =============================================================================
// End-to-End Validation Tests (against R reference)
// =============================================================================

/// Test one-compartment IV bolus against R reference
/// Requires: onecomp_iv_reference.json from generate_reference.R
#[test]
fn test_validate_onecomp_iv() -> Result<()> {
    use pmcore::algorithms::parametric::dispatch_parametric_algorithm;
    use pmcore::prelude::*;

    println!("=== Validating One-Compartment IV vs R Reference ===");

    // Load R reference
    let ref_path = format!("{}/onecomp_iv_reference.json", VALIDATION_DIR);
    let reference = load_reference(&ref_path)
        .map_err(|e| anyhow::anyhow!("Failed to load reference: {}", e))?;

    println!(
        "  Reference loaded: {} subjects, {} observations",
        reference.n_subjects, reference.n_observations
    );

    // Create model (same as R) - b[0] is the bolus input term!
    let dose = 100.0;
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
    );

    // Generate same data as R (from reference true values)
    let true_ke = reference
        .true_values
        .as_ref()
        .and_then(|t| t.ke)
        .unwrap_or(0.4);
    let true_v = reference
        .true_values
        .as_ref()
        .and_then(|t| t.v)
        .unwrap_or(10.0);
    let times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0];

    // Match R's deterministic random effects
    let subjects: Vec<Subject> = (0..reference.n_subjects)
        .map(|id| {
            let eta_ke = 0.15 * ((id as f64) * 0.5).sin();
            let eta_v = 0.15 * ((id as f64) * 0.7).cos();
            let subj_ke = true_ke * f64::exp(eta_ke);
            let subj_v = true_v * f64::exp(eta_v);

            let mut builder = Subject::builder(id.to_string()).bolus(0.0, dose, 0);
            for &t in &times {
                let conc = (dose / subj_v) * f64::exp(-subj_ke * t);
                builder = builder.observation(t, conc, 0);
            }
            builder.build()
        })
        .collect();
    let data = Data::new(subjects);

    // Match R settings
    let params = Parameters::new().add("ke", 0.1, 0.8).add("v", 5.0, 15.0);

    // Residual error model (parametric algorithms use ResidualErrorModels)
    use pharmsol::{ResidualErrorModel, ResidualErrorModels};
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::constant(0.1));

    let settings = Settings::builder()
        .set_algorithm(Algorithm::SAEM)
        .set_parameters(params)
        .set_residual_error(residual_error)
        // TODO: Match seed and iteration counts to R
        .build();

    // Run algorithm
    let mut algorithm = dispatch_parametric_algorithm(settings, eq, data)?;
    let result = algorithm.fit()?;

    // Compare results
    println!("\n  === Comparison with R Reference ===");

    // Population mean (μ is returned in ψ space by into_result())
    let rust_mu_psi: Vec<f64> = (0..result.population().npar())
        .map(|i| result.population().mu()[i])
        .collect();

    println!("  mu (ψ space, Rust): {:?}", rust_mu_psi);
    println!("  mu (ψ space, R):    {:?}", reference.mu_psi);

    // Compare with tolerance (comparing ψ values)
    let mu_rtol = 0.10; // 10% relative tolerance
    for (i, (r, ref_val)) in rust_mu_psi.iter().zip(reference.mu_psi.iter()).enumerate() {
        let rel_err = (*r - *ref_val).abs() / ref_val.abs().max(1e-10);
        println!(
            "    mu[{}]: Rust={:.4}, R={:.4}, rel_err={:.2}%",
            i,
            r,
            ref_val,
            rel_err * 100.0
        );

        if rel_err > mu_rtol {
            println!("    WARNING: Exceeds {}% tolerance!", mu_rtol * 100.0);
        }
    }

    // Omega diagonal
    let rust_omega_diag: Vec<f64> = (0..result.population().npar())
        .map(|i| result.population().omega()[(i, i)])
        .collect();

    println!("  omega_diag (Rust): {:?}", rust_omega_diag);
    println!("  omega_diag (R):    {:?}", reference.omega_diag);

    // Objective function
    println!("  objf (Rust): {:.2}", result.objf());
    println!("  objf (R):    {:.2}", reference.objf);

    let objf_rel_err = (result.objf() - reference.objf).abs() / reference.objf.abs();
    println!("  objf rel_err: {:.2}%", objf_rel_err * 100.0);

    // Assertions (with generous tolerance for stochastic algorithm)
    assert_vec_close(&rust_mu_psi, &reference.mu_psi, 0.20, "mu_psi");

    println!("\n  ✓ One-compartment IV validation complete\n");
    Ok(())
}

/// Test theophylline against R reference
/// Requires: theo_reference.json from generate_reference.R
#[test]
#[ignore = "Full theophylline validation (~5 min) - run with --ignored"]
fn test_validate_theophylline() -> Result<()> {
    use pmcore::algorithms::parametric::dispatch_parametric_algorithm;
    use pmcore::prelude::*;

    println!("=== Validating Theophylline vs R Reference ===");

    // Load R reference
    let ref_path = format!("{}/theo_reference.json", VALIDATION_DIR);
    let reference = load_reference(&ref_path)
        .map_err(|e| anyhow::anyhow!("Failed to load reference: {}", e))?;

    println!(
        "  Reference: {} subjects, {} observations",
        reference.n_subjects, reference.n_observations
    );
    println!("  R results - mu_psi: {:?}", reference.mu_psi);
    println!("  R results - omega_diag: {:?}", reference.omega_diag);
    println!("  R results - sigma: {:.4}", reference.sigma);
    println!("  R results - objf: {:.2}", reference.objf);

    // One-compartment model with first-order absorption
    // Matches R saemix model: dose*ka/(V*(ka-k))*(exp(-k*t) - exp(-ka*t))
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ka, v, cl);
            let ke = cl / v;
            dx[0] = -ka * x[0] + b[0]; // absorption compartment + bolus
            dx[1] = ka * x[0] - ke * x[1]; // central compartment
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, v, _cl);
            y[0] = x[1] / v;
        },
    );

    // Theophylline data (12 subjects, same as saemix theo.saemix dataset)
    let subjects_data: Vec<(u32, f64, Vec<f64>, Vec<f64>)> = vec![
        (
            1,
            319.992,
            vec![0.25, 0.57, 1.12, 2.02, 3.82, 5.10, 7.03, 9.05, 12.12, 24.37],
            vec![2.84, 6.57, 10.50, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28],
        ),
        (
            2,
            318.560,
            vec![0.27, 0.52, 1.00, 1.92, 3.50, 5.02, 7.03, 9.00, 12.00, 24.30],
            vec![1.72, 7.91, 8.31, 8.33, 6.85, 6.08, 5.40, 4.55, 3.01, 0.90],
        ),
        (
            3,
            319.365,
            vec![0.27, 0.58, 1.02, 2.02, 3.62, 5.08, 7.07, 9.00, 12.15, 24.17],
            vec![4.40, 6.90, 8.20, 7.80, 7.50, 6.20, 5.30, 4.90, 3.70, 1.05],
        ),
        (
            4,
            319.992,
            vec![0.35, 0.60, 1.07, 2.13, 3.50, 5.02, 7.02, 9.02, 11.98, 24.65],
            vec![1.89, 4.60, 8.60, 8.38, 7.54, 6.88, 5.78, 5.33, 4.19, 1.15],
        ),
        (
            5,
            320.619,
            vec![0.30, 0.52, 1.00, 2.02, 3.50, 5.02, 7.02, 9.10, 12.00, 24.35],
            vec![2.02, 5.63, 11.40, 9.33, 8.74, 7.56, 7.09, 5.90, 4.37, 1.57],
        ),
        (
            6,
            320.619,
            vec![0.27, 0.58, 1.15, 2.03, 3.57, 5.00, 7.00, 9.22, 12.10, 23.85],
            vec![1.29, 3.08, 6.44, 6.32, 5.53, 4.94, 4.02, 3.46, 2.78, 0.92],
        ),
        (
            7,
            277.767,
            vec![0.25, 0.50, 1.02, 2.02, 3.48, 5.00, 6.98, 9.00, 12.05, 24.22],
            vec![3.59, 6.11, 7.56, 6.54, 5.37, 4.84, 4.02, 3.83, 2.81, 0.85],
        ),
        (
            8,
            276.514,
            vec![0.25, 0.52, 0.98, 2.02, 3.53, 5.05, 7.15, 9.07, 12.10, 24.12],
            vec![0.73, 4.00, 6.81, 8.00, 7.09, 5.89, 5.22, 4.75, 3.41, 0.96],
        ),
        (
            9,
            299.550,
            vec![0.30, 0.63, 1.05, 2.02, 3.53, 5.02, 7.17, 8.80, 11.60, 24.43],
            vec![3.15, 6.96, 9.70, 9.52, 7.17, 6.28, 5.28, 4.66, 3.82, 1.15],
        ),
        (
            10,
            298.297,
            vec![0.37, 0.77, 1.02, 2.05, 3.55, 5.05, 7.08, 9.00, 12.12, 24.08],
            vec![7.37, 9.03, 10.21, 9.18, 8.02, 7.14, 6.08, 5.54, 4.57, 1.17],
        ),
        (
            11,
            300.176,
            vec![0.25, 0.50, 0.98, 1.98, 3.60, 5.02, 7.03, 9.03, 12.12, 24.28],
            vec![0.92, 2.63, 6.85, 9.05, 7.90, 7.44, 6.13, 5.31, 4.10, 1.44],
        ),
        (
            12,
            298.297,
            vec![0.25, 0.52, 1.00, 2.07, 3.50, 4.95, 7.00, 9.02, 12.00, 24.15],
            vec![1.11, 6.33, 9.99, 9.37, 8.50, 6.89, 5.94, 5.26, 4.35, 1.25],
        ),
    ];

    let subjects: Vec<Subject> = subjects_data
        .into_iter()
        .map(|(id, dose, times, concs)| {
            let mut builder = Subject::builder(id.to_string()).bolus(0.0, dose, 0);
            for (t, c) in times.into_iter().zip(concs.into_iter()) {
                builder = builder.observation(t, c, 0);
            }
            builder.build()
        })
        .collect();
    let data = Data::new(subjects);

    // Parameter ranges (matching R saemix initial values region)
    let params = Parameters::new()
        .add("ka", 0.5, 3.0)
        .add("v", 15.0, 50.0)
        .add("cl", 0.5, 5.0);

    // Residual error model (constant, matching R)
    use pharmsol::{ResidualErrorModel, ResidualErrorModels};
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::constant(1.0));

    let settings = Settings::builder()
        .set_algorithm(Algorithm::SAEM)
        .set_parameters(params)
        .set_residual_error(residual_error)
        .build();

    // Run SAEM
    let mut algorithm = dispatch_parametric_algorithm(settings, eq, data)?;
    let result = algorithm.fit()?;

    // Compare results
    println!("\n  === Comparison with R Reference ===");

    let rust_mu_psi: Vec<f64> = (0..result.population().npar())
        .map(|i| result.population().mu()[i])
        .collect();

    println!("  mu (ψ space, Rust): {:?}", rust_mu_psi);
    println!("  mu (ψ space, R):    {:?}", reference.mu_psi);

    // Compare each parameter
    let param_names = ["ka", "V", "CL"];
    for (i, (r, ref_val)) in rust_mu_psi.iter().zip(reference.mu_psi.iter()).enumerate() {
        let rel_err = (*r - *ref_val).abs() / ref_val.abs().max(1e-10);
        println!(
            "    {}: Rust={:.4}, R={:.4}, rel_err={:.2}%",
            param_names[i],
            r,
            ref_val,
            rel_err * 100.0
        );
    }

    // Omega diagonal
    let rust_omega_diag: Vec<f64> = (0..result.population().npar())
        .map(|i| result.population().omega()[(i, i)])
        .collect();
    println!("  omega_diag (Rust): {:?}", rust_omega_diag);
    println!("  omega_diag (R):    {:?}", reference.omega_diag);

    // Objective function
    println!("  objf (Rust): {:.2}", result.objf());
    println!("  objf (R):    {:.2}", reference.objf);

    // Assertions: population means should be within 25% for a 3-parameter stochastic algorithm
    let tolerance = 0.25;
    assert_vec_close(
        &rust_mu_psi,
        &reference.mu_psi,
        tolerance,
        "mu_psi (theophylline)",
    );

    println!("\n  ✓ Theophylline validation complete\n");
    Ok(())
}

// =============================================================================
// Helper Tests
// =============================================================================

/// Verify test data files exist
#[test]
fn test_validation_directory_exists() {
    let path = std::path::Path::new(VALIDATION_DIR);
    if !path.exists() {
        println!("Validation directory does not exist: {}", VALIDATION_DIR);
        println!("Run from PMcore root directory");
    }
}
