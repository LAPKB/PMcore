use anyhow::Result;
use faer::Mat;
use pmcore::structs::psi::Psi;

/// Test the IPM with a simple 2x2 matrix
#[test]
fn test_burke_ipm_simple() -> Result<()> {
    // Create a simple 2x2 psi matrix
    // Subject 1: [0.8, 0.2]
    // Subject 2: [0.3, 0.7]
    let mat = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => 0.8,
        (0, 1) => 0.2,
        (1, 0) => 0.3,
        (1, 1) => 0.7,
        _ => 0.0,
    });

    let psi = Psi::from(mat);

    // Run Burke's IPM
    let result = pmcore::routines::evaluation::ipm::burke(&psi);

    // Should succeed
    assert!(result.is_ok());

    let (weights, objf) = result.unwrap();

    // Weights should sum to 1
    let sum: f64 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Weights should sum to 1, got {}",
        sum
    );

    // All weights should be non-negative
    for w in weights.iter() {
        assert!(w >= 0.0, "All weights should be non-negative, got {}", w);
    }

    // Objective function should be finite
    assert!(objf.is_finite(), "Objective function should be finite");

    Ok(())
}

/// Test the IPM with a larger matrix
#[test]
fn test_burke_ipm_larger() -> Result<()> {
    // Create a 5x10 psi matrix with random-like values
    let mat = Mat::from_fn(5, 10, |i, j| {
        // Generate deterministic "random-like" values
        let val = ((i * 7 + j * 13) % 100) as f64 / 100.0 + 0.01;
        val
    });

    let psi = Psi::from(mat);

    // Run Burke's IPM
    let result = pmcore::routines::evaluation::ipm::burke(&psi);

    assert!(result.is_ok());

    let (weights, objf) = result.unwrap();

    // Weights should sum to 1
    let sum: f64 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Weights should sum to 1, got {}",
        sum
    );

    // All weights should be non-negative
    for w in weights.iter() {
        assert!(w >= 0.0, "All weights should be non-negative, got {}", w);
    }

    // Objective function should be finite
    assert!(objf.is_finite(), "Objective function should be finite");

    Ok(())
}

/// Test the IPM with uniform likelihoods
#[test]
fn test_burke_ipm_uniform() -> Result<()> {
    // Create a matrix where all likelihoods are equal
    let mat = Mat::from_fn(3, 5, |_i, _j| 1.0);

    let psi = Psi::from(mat);

    // Run Burke's IPM
    let result = pmcore::routines::evaluation::ipm::burke(&psi);

    assert!(result.is_ok());

    let (weights, _objf) = result.unwrap();

    // With uniform likelihoods, weights should be approximately equal
    let expected_weight = 1.0 / 5.0;
    for w in weights.iter() {
        assert!(
            (w - expected_weight).abs() < 0.01,
            "Weights should be approximately {}, got {}",
            expected_weight,
            w
        );
    }

    Ok(())
}

/// Test the IPM with a matrix containing negative values (should be made absolute)
#[test]
fn test_burke_ipm_with_negatives() -> Result<()> {
    // Create a matrix with some negative values
    let mat = Mat::from_fn(2, 3, |i, j| {
        match (i, j) {
            (0, 0) => -0.5, // Negative value
            (0, 1) => 0.3,
            (0, 2) => 0.2,
            (1, 0) => 0.4,
            (1, 1) => -0.1, // Negative value
            (1, 2) => 0.5,
            _ => 0.0,
        }
    });

    let psi = Psi::from(mat);

    // Run Burke's IPM - should handle negatives by taking absolute value
    let result = pmcore::routines::evaluation::ipm::burke(&psi);

    assert!(result.is_ok());

    let (weights, _objf) = result.unwrap();

    // Weights should sum to 1
    let sum: f64 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Weights should sum to 1, got {}",
        sum
    );

    Ok(())
}

/// Test the IPM with a matrix containing infinite values (should fail)
#[test]
fn test_burke_ipm_with_infinites() {
    // Create a matrix with an infinite value
    let mat = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => f64::INFINITY,
        _ => 1.0,
    });

    let psi = Psi::from(mat);

    // Run Burke's IPM - should fail with infinite values
    let result = pmcore::routines::evaluation::ipm::burke(&psi);

    assert!(result.is_err(), "Should fail with infinite values");
}

/// Test the IPM with a matrix containing NaN values (should fail)
#[test]
fn test_burke_ipm_with_nan() {
    // Create a matrix with a NaN value
    let mat = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => f64::NAN,
        _ => 1.0,
    });

    let psi = Psi::from(mat);

    // Run Burke's IPM - should fail with NaN values
    let result = pmcore::routines::evaluation::ipm::burke(&psi);

    assert!(result.is_err(), "Should fail with NaN values");
}

/// Test the IPM with high-dimensional matrix
#[test]
fn test_burke_ipm_high_dimensional() -> Result<()> {
    // Create a larger matrix (20 subjects, 50 support points)
    let mat = Mat::from_fn(20, 50, |i, j| {
        // Generate deterministic values
        let val = ((i * 11 + j * 17) % 1000) as f64 / 1000.0 + 0.001;
        val
    });

    let psi = Psi::from(mat);

    // Run Burke's IPM
    let result = pmcore::routines::evaluation::ipm::burke(&psi);

    assert!(result.is_ok());

    let (weights, objf) = result.unwrap();

    // Weights should sum to 1
    let sum: f64 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Weights should sum to 1, got {}",
        sum
    );

    // All weights should be non-negative
    for w in weights.iter() {
        assert!(w >= 0.0, "All weights should be non-negative, got {}", w);
    }

    // Objective function should be finite
    assert!(objf.is_finite(), "Objective function should be finite");

    Ok(())
}
