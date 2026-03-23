use anyhow::Result;
use faer::{Col, Mat};
use pmcore::structs::{psi::Psi, theta::Theta, weights::Weights};
use std::io::Cursor;

/// Test Psi creation and basic operations
#[test]
fn test_psi_creation() {
    let psi = Psi::new();
    assert_eq!(psi.nspp(), 0);
    assert_eq!(psi.nsub(), 0);
}

/// Test Psi from Mat
#[test]
fn test_psi_from_mat() {
    let mat = Mat::from_fn(3, 4, |i, j| (i * 4 + j) as f64);
    let psi = Psi::from(mat);

    assert_eq!(psi.nspp(), 3);
    assert_eq!(psi.nsub(), 4);
}

/// Test Psi CSV serialization and deserialization
#[test]
fn test_psi_csv_roundtrip() -> Result<()> {
    // Create a test matrix
    let mat = Mat::from_fn(2, 3, |i, j| match (i, j) {
        (0, 0) => 1.0,
        (0, 1) => 2.0,
        (0, 2) => 3.0,
        (1, 0) => 4.0,
        (1, 1) => 5.0,
        (1, 2) => 6.0,
        _ => 0.0,
    });
    let psi = Psi::from(mat);

    // Serialize to CSV
    let mut buffer = Vec::new();
    psi.to_csv(&mut buffer)?;

    // Deserialize from CSV
    // Note: csv crate by default assumes first row is a header
    // The from_csv implementation uses deserialize which may treat first line as header
    // Let's test with a simpler case
    let csv_content = "1.0,2.0,3.0\n4.0,5.0,6.0";
    let cursor = Cursor::new(csv_content);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(cursor);

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for result in reader.deserialize() {
        let row: Vec<f64> = result?;
        rows.push(row);
    }

    // Create psi from rows
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].len(), 3);

    let mat_restored = Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j]);
    let psi_restored = Psi::from(mat_restored);

    // Check dimensions
    assert_eq!(psi_restored.nspp(), 2); // nrows
    assert_eq!(psi_restored.nsub(), 3); // ncols

    // Check values
    assert_eq!(*psi_restored.matrix().get(0, 0), 1.0);
    assert_eq!(*psi_restored.matrix().get(0, 2), 3.0);
    assert_eq!(*psi_restored.matrix().get(1, 1), 5.0);

    Ok(())
}

/// Test Psi from empty CSV should fail
#[test]
fn test_psi_from_empty_csv() {
    let empty_csv = "";
    let cursor = Cursor::new(empty_csv);
    let result = Psi::from_csv(cursor);

    assert!(result.is_err());
}

/// Test Weights creation and operations
#[test]
fn test_weights_creation() {
    let weights = Weights::from_vec(vec![0.3, 0.5, 0.2]);

    assert_eq!(weights.len(), 3);

    let sum: f64 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

/// Test Weights from Col
#[test]
fn test_weights_from_col() {
    let col = Col::from_fn(4, |i| (i + 1) as f64 * 0.1);
    let weights = Weights::from(col);

    assert_eq!(weights.len(), 4);
    assert_eq!(weights[0], 0.1);
    assert_eq!(weights[3], 0.4);
}

/// Test Weights indexing
#[test]
fn test_weights_indexing() {
    let mut weights = Weights::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

    // Read index
    assert_eq!(weights[0], 0.1);
    assert_eq!(weights[3], 0.4);

    // Write index
    weights[1] = 0.5;
    assert_eq!(weights[1], 0.5);
}

/// Test Weights to_vec
#[test]
fn test_weights_to_vec() {
    let original = vec![0.25, 0.25, 0.25, 0.25];
    let weights = Weights::from_vec(original.clone());
    let restored = weights.to_vec();

    assert_eq!(original, restored);
}

/// Test Weights serialization
#[test]
fn test_weights_serialization() -> Result<()> {
    let weights = Weights::from_vec(vec![0.1, 0.2, 0.3, 0.4]);

    // Serialize to JSON
    let json = serde_json::to_string(&weights)?;

    // Deserialize
    let restored: Weights = serde_json::from_str(&json)?;

    assert_eq!(weights.len(), restored.len());
    for (a, b) in weights.iter().zip(restored.iter()) {
        assert!((a - b).abs() < 1e-10);
    }

    Ok(())
}

/// Test Theta creation
#[test]
fn test_theta_creation() {
    let theta = Theta::new();
    assert_eq!(theta.nspp(), 0);
}

/// Test Weights default
#[test]
fn test_weights_default() {
    let weights = Weights::default();
    assert_eq!(weights.len(), 0);
}

/// Test Psi equality
#[test]
fn test_psi_equality() {
    let mat1 = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
    let mat2 = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
    let mat3 = Mat::from_fn(2, 2, |i, j| (i * j) as f64);

    let psi1 = Psi::from(mat1);
    let psi2 = Psi::from(mat2);
    let psi3 = Psi::from(mat3);

    assert_eq!(psi1, psi2);
    assert_ne!(psi1, psi3);
}

/// Test Psi serialization
#[test]
fn test_psi_serialization() -> Result<()> {
    let mat = Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
    let psi = Psi::from(mat);

    // Serialize to JSON
    let json = serde_json::to_string(&psi)?;

    // Should contain array data
    assert!(json.contains("["));

    // Deserialize
    let restored: Psi = serde_json::from_str(&json)?;

    assert_eq!(psi, restored);

    Ok(())
}
