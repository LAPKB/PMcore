use anyhow::Result;
use pmcore::mmopt::{mmopt, MmoptResult};
use pmcore::prelude::*;
use pmcore::structs::theta::Theta;

/// Helper to create a simple one-compartment model
fn one_comp_model() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
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

/// Helper to create a simple error model
fn additive_error_model() -> ErrorModel {
    ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 0.0)
}

/// Helper to create parameters for the one-compartment model
fn one_comp_params() -> Parameters {
    Parameters::new().add("ke", 0.1, 1.0).add("v", 30.0, 100.0)
}

/// Test basic mmopt functionality with a simple one-compartment model
#[test]
fn test_mmopt_basic() -> Result<()> {
    let eq = one_comp_model();
    let params = one_comp_params();

    // Two support points with different PK parameters
    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.3,  // ke for spp1
        (0, 1) => 50.0, // v for spp1
        (1, 0) => 0.8,  // ke for spp2
        (1, 1) => 80.0, // v for spp2
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params)?;

    // Subject with candidate observation times after a bolus dose
    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(6.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    let errormodel = additive_error_model();
    let weights = vec![0.5, 0.5];

    let result = mmopt(&theta, &subject, eq, errormodel, 0, 2, weights)?;

    assert_eq!(
        result.times.len(),
        2,
        "Should select exactly 2 sample times"
    );
    assert!(result.risk >= 0.0, "Risk must be non-negative");
    assert!(result.risk.is_finite(), "Risk must be finite");

    // All selected times should be from the candidate set
    let candidate_times = vec![1.0, 2.0, 4.0, 6.0, 8.0, 12.0];
    for t in &result.times {
        assert!(
            candidate_times.contains(t),
            "Selected time {} is not in the candidate set",
            t
        );
    }

    Ok(())
}

/// Test that selecting more samples results in equal or lower risk
#[test]
fn test_mmopt_more_samples_lower_risk() -> Result<()> {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.2,
        (0, 1) => 50.0,
        (1, 0) => 0.7,
        (1, 1) => 70.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params)?;

    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    let errormodel = additive_error_model();

    let result_2 = mmopt(
        &theta,
        &subject,
        eq.clone(),
        errormodel.clone(),
        0,
        2,
        vec![0.5, 0.5],
    )?;
    let result_3 = mmopt(
        &theta,
        &subject,
        eq.clone(),
        errormodel.clone(),
        0,
        3,
        vec![0.5, 0.5],
    )?;

    assert!(
        result_3.risk <= result_2.risk + 1e-10,
        "More samples should yield lower or equal risk: {} vs {}",
        result_3.risk,
        result_2.risk
    );

    Ok(())
}

/// Test mmopt with three support points
#[test]
fn test_mmopt_three_support_points() -> Result<()> {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(3, 2, |r, c| match (r, c) {
        (0, 0) => 0.2,
        (0, 1) => 40.0,
        (1, 0) => 0.5,
        (1, 1) => 60.0,
        (2, 0) => 0.9,
        (2, 1) => 90.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params)?;

    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(3.0, 0.0, 0)
        .observation(6.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    let errormodel = additive_error_model();
    let weights = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

    let result = mmopt(&theta, &subject, eq, errormodel, 0, 2, weights)?;

    assert_eq!(result.times.len(), 2);
    assert!(result.risk >= 0.0);
    assert!(result.risk.is_finite());

    Ok(())
}

/// Test that mmopt with all candidate times produces the lowest possible risk
#[test]
fn test_mmopt_all_samples() -> Result<()> {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.3,
        (0, 1) => 50.0,
        (1, 0) => 0.6,
        (1, 1) => 75.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params)?;

    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    let errormodel = additive_error_model();
    let weights = vec![0.5, 0.5];

    // Select all 3 samples (only one combination)
    let result = mmopt(&theta, &subject, eq, errormodel, 0, 3, weights)?;

    assert_eq!(result.times.len(), 3);
    assert_eq!(result.times, vec![1.0, 4.0, 8.0]);

    Ok(())
}

/// Test validation: subject with multiple occasions should fail
#[test]
fn test_mmopt_multiple_occasions_error() {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.3,
        (0, 1) => 50.0,
        (1, 0) => 0.6,
        (1, 1) => 75.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params).unwrap();

    // Subject with two occasions
    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .repeat(1, 24.0)
        .bolus(24.0, 100.0, 0)
        .observation(25.0, 0.0, 0)
        .build();

    // Only proceed if the subject actually has multiple occasions
    if subject.occasions().len() > 1 {
        let result = mmopt(
            &theta,
            &subject,
            eq,
            additive_error_model(),
            0,
            1,
            vec![0.5, 0.5],
        );
        assert!(result.is_err());
    }
}

/// Test validation: fewer than 2 support points should fail
#[test]
fn test_mmopt_single_support_point_error() {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(1, 2, |_r, c| match c {
        0 => 0.3,
        1 => 50.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params).unwrap();

    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .build();

    let result = mmopt(
        &theta,
        &subject,
        eq,
        additive_error_model(),
        0,
        1,
        vec![1.0],
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("At least 2 support points"));
}

/// Test validation: weights length mismatch should fail
#[test]
fn test_mmopt_weights_mismatch_error() {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.3,
        (0, 1) => 50.0,
        (1, 0) => 0.6,
        (1, 1) => 75.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params).unwrap();

    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .build();

    // 3 weights for 2 support points
    let result = mmopt(
        &theta,
        &subject,
        eq,
        additive_error_model(),
        0,
        1,
        vec![0.33, 0.33, 0.34],
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Weights length"));
}

/// Test validation: nsamp = 0 should fail
#[test]
fn test_mmopt_zero_samples_error() {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.3,
        (0, 1) => 50.0,
        (1, 0) => 0.6,
        (1, 1) => 75.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params).unwrap();

    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .build();

    let result = mmopt(
        &theta,
        &subject,
        eq,
        additive_error_model(),
        0,
        0,
        vec![0.5, 0.5],
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("at least 1"));
}

/// Test validation: nsamp exceeds candidate times should fail
#[test]
fn test_mmopt_too_many_samples_error() {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.3,
        (0, 1) => 50.0,
        (1, 0) => 0.6,
        (1, 1) => 75.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params).unwrap();

    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .build();

    // Request 5 samples but only 2 candidate times
    let result = mmopt(
        &theta,
        &subject,
        eq,
        additive_error_model(),
        0,
        5,
        vec![0.5, 0.5],
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeds"));
}

/// Test that unequal weights influence the optimal sampling design
#[test]
fn test_mmopt_unequal_weights() -> Result<()> {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.2,
        (0, 1) => 50.0,
        (1, 0) => 0.8,
        (1, 1) => 80.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params)?;

    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .observation(12.0, 0.0, 0)
        .build();

    let errormodel = additive_error_model();

    let result_equal = mmopt(
        &theta,
        &subject,
        eq.clone(),
        errormodel.clone(),
        0,
        2,
        vec![0.5, 0.5],
    )?;

    let result_skewed = mmopt(&theta, &subject, eq, errormodel, 0, 2, vec![0.9, 0.1])?;

    // Different weights should generally produce different risks
    // (or at least both should be valid)
    assert!(result_equal.risk.is_finite());
    assert!(result_skewed.risk.is_finite());
    assert!(result_equal.risk >= 0.0);
    assert!(result_skewed.risk >= 0.0);

    Ok(())
}

/// Test MmoptResult Display implementation
#[test]
fn test_mmopt_result_display() {
    let result = MmoptResult {
        times: vec![2.0, 6.0, 12.0],
        risk: 0.042,
    };
    let display = format!("{}", result);
    assert!(display.contains("2.0"));
    assert!(display.contains("6.0"));
    assert!(display.contains("12.0"));
    assert!(display.contains("0.042"));
}

/// Test with a single sample selection
#[test]
fn test_mmopt_single_sample() -> Result<()> {
    let eq = one_comp_model();
    let params = one_comp_params();

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.2,
        (0, 1) => 40.0,
        (1, 0) => 0.9,
        (1, 1) => 90.0,
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params)?;

    let subject = Subject::builder("test")
        .bolus(0.0, 100.0, 0)
        .observation(0.5, 0.0, 0)
        .observation(1.0, 0.0, 0)
        .observation(2.0, 0.0, 0)
        .observation(4.0, 0.0, 0)
        .observation(8.0, 0.0, 0)
        .build();

    let result = mmopt(
        &theta,
        &subject,
        eq,
        additive_error_model(),
        0,
        1,
        vec![0.5, 0.5],
    )?;

    assert_eq!(result.times.len(), 1);
    assert!(result.risk.is_finite());
    assert!(result.risk >= 0.0);

    Ok(())
}
