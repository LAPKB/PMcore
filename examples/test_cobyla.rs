//! Test COBYLA optimizer to verify it works before integration
//!
//! Run with: cargo run --example test_cobyla

use cobyla::{minimize, RhoBeg};

fn main() {
    println!("Testing COBYLA optimizer...\n");

    // Test 1: Simple quadratic minimization (Rosenbrock-like)
    // f(x,y) = (1-x)² + 100(y-x²)²
    // Minimum at (1, 1)
    test_rosenbrock();

    // Test 2: Bounded optimization
    // Minimize x² + y² subject to 0.5 <= x <= 2, 0.5 <= y <= 2
    // Constrained minimum at (0.5, 0.5)
    test_bounded();

    // Test 3: Higher dimensional (4D like our PK problems)
    test_4d_bounded();

    println!("\n✅ All COBYLA tests passed!");
}

fn test_rosenbrock() {
    println!("Test 1: Rosenbrock function (unbounded)");
    
    fn rosenbrock(x: &[f64], _data: &mut ()) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    let x0 = vec![-1.0, -1.0];
    let bounds: Vec<(f64, f64)> = vec![(-10.0, 10.0), (-10.0, 10.0)];
    let cons: Vec<fn(&[f64], &mut ()) -> f64> = vec![];  // No constraints
    
    let result = minimize(
        rosenbrock,
        &x0,
        &bounds,
        &cons,
        (),           // User data
        5000,         // Max evaluations (Rosenbrock needs more)
        RhoBeg::All(0.5),
        None,         // Default stopping tolerances
    );

    match result {
        Ok((status, x, fval)) => {
            println!("  Status: {:?}", status);
            println!("  Solution: x = [{:.6}, {:.6}]", x[0], x[1]);
            println!("  Expected: x = [1.0, 1.0]");
            println!("  f(x) = {:.6}", fval);
            
            assert!((x[0] - 1.0).abs() < 0.02, "x[0] should be ~1.0, got {}", x[0]);
            assert!((x[1] - 1.0).abs() < 0.05, "x[1] should be ~1.0, got {}", x[1]);
            println!("  ✓ PASSED\n");
        }
        Err((status, x, fval)) => {
            println!("  Failed status: {:?}", status);
            println!("  Last x: [{:.6}, {:.6}], f(x) = {:.6}", x[0], x[1], fval);
            panic!("COBYLA failed");
        }
    }
}

fn test_bounded() {
    println!("Test 2: Bounded quadratic");
    
    fn quadratic(x: &[f64], _data: &mut ()) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }

    let x0 = vec![1.5, 1.5];
    let bounds: Vec<(f64, f64)> = vec![(0.5, 2.0), (0.5, 2.0)];
    let cons: Vec<fn(&[f64], &mut ()) -> f64> = vec![];
    
    let result = minimize(
        quadratic,
        &x0,
        &bounds,
        &cons,
        (),
        500,
        RhoBeg::All(0.3),
        None,
    );

    match result {
        Ok((status, x, fval)) => {
            println!("  Status: {:?}", status);
            println!("  Solution: x = [{:.6}, {:.6}]", x[0], x[1]);
            println!("  Expected: x = [0.5, 0.5] (at lower bounds)");
            println!("  f(x) = {:.6}", fval);
            
            assert!((x[0] - 0.5).abs() < 0.01, "x[0] should be ~0.5");
            assert!((x[1] - 0.5).abs() < 0.01, "x[1] should be ~0.5");
            println!("  ✓ PASSED\n");
        }
        Err((status, x, fval)) => {
            println!("  Failed status: {:?}", status);
            println!("  Last x: [{:.6}, {:.6}], f(x) = {:.6}", x[0], x[1], fval);
            panic!("COBYLA failed");
        }
    }
}

fn test_4d_bounded() {
    println!("Test 3: 4D bounded optimization (like PK parameters)");
    
    // Simulate a multi-modal function with one clear minimum
    // f(x) = sum((x_i - target_i)²) where targets are inside bounds
    fn multivariate(x: &[f64], _data: &mut ()) -> f64 {
        let targets = [0.3, 1.5, 0.8, 2.0];
        x.iter()
            .zip(targets.iter())
            .map(|(xi, ti)| (xi - ti).powi(2))
            .sum()
    }

    let x0 = vec![1.0, 1.0, 1.0, 1.0];  // Start away from optimum
    let bounds: Vec<(f64, f64)> = vec![
        (0.1, 2.0),   // Like Ke
        (0.5, 5.0),   // Like V
        (0.1, 3.0),   // Like Ka
        (0.5, 10.0),  // Like another param
    ];
    let cons: Vec<fn(&[f64], &mut ()) -> f64> = vec![];
    
    let result = minimize(
        multivariate,
        &x0,
        &bounds,
        &cons,
        (),
        1000,
        RhoBeg::All(0.2),
        None,
    );

    match result {
        Ok((status, x, fval)) => {
            println!("  Status: {:?}", status);
            println!("  Solution: x = [{:.4}, {:.4}, {:.4}, {:.4}]", x[0], x[1], x[2], x[3]);
            println!("  Expected: x = [0.3, 1.5, 0.8, 2.0]");
            println!("  f(x) = {:.6}", fval);
            
            let targets = [0.3, 1.5, 0.8, 2.0];
            for (i, (&xi, &ti)) in x.iter().zip(targets.iter()).enumerate() {
                assert!((xi - ti).abs() < 0.05, "x[{}] should be ~{}", i, ti);
            }
            println!("  ✓ PASSED\n");
        }
        Err((status, x, fval)) => {
            println!("  Failed status: {:?}", status);
            println!("  Last x: {:?}, f(x) = {:.6}", x, fval);
            panic!("COBYLA failed");
        }
    }
}
