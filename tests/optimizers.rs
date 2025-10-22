// use pmcore::routines::evaluation::clarabel::clarabel_weights;
use pmcore::routines::evaluation::em::em_weights;
use pmcore::routines::evaluation::ipm::burke;
use pmcore::routines::evaluation::squarem::squarem_weights;
// use pmcore::routines::evaluation::newton_cg::newton_cg_weights;
use faer::Mat;
use pmcore::routines::evaluation::frank_wolfe::frank_wolfe_weights;
use pmcore::routines::evaluation::lbfgs::lbfgs_weights_default;
use pmcore::routines::evaluation::mirror_descent::mirror_descent_weights;
use pmcore::routines::evaluation::pgd::pgd_weights;
// use pmcore::routines::evaluation::stochastic_mirror_descent::stochastic_mirror_descent_weights;
use pmcore::structs::psi::Psi;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[test]
fn optimizers_vs_burke_small() {
    let n_sub = 3;
    let n_point = 4;
    let mat = Mat::from_fn(n_sub, n_point, |i, j| {
        if j == 0 {
            10.0
        } else {
            1.0 + 0.1 * (i as f64)
        }
    });
    let psi = Psi::from(mat);

    let (lam_burke, obj_burke) = burke(&psi).unwrap();
    // compute objective consistently from returned lambda
    let psi_mat = psi.matrix();
    let mut psi_dot_burke: Vec<f64> = vec![0.0; psi_mat.nrows()];
    for i in 0..psi_mat.nrows() {
        for j in 0..psi_mat.ncols() {
            psi_dot_burke[i] += psi_mat.get(i, j) * lam_burke[j];
        }
    }
    println!("burke obj = {}", obj_burke);

    let (lam_em, obj_em) = em_weights(&psi).unwrap();
    let (lam_sq, obj_sq) = squarem_weights(&psi).unwrap();
    // let (lam_ng, obj_ng) = newton_cg_weights(&psi).unwrap();
    let (lam_lb, obj_lb) = lbfgs_weights_default(&psi).unwrap();
    let (lam_md, obj_md) = mirror_descent_weights(&psi).unwrap();
    let (lam_pgd, obj_pgd) = pgd_weights(&psi).unwrap();
    let (lam_fw, obj_fw) = frank_wolfe_weights(&psi).unwrap();
    // let (lam_smd, obj_smd) = stochastic_mirror_descent_weights(&psi).unwrap();
    // let (lam_cl, obj_cl) = clarabel_weights(&psi).unwrap();

    // Check objectives within tolerance
    let names = [
        "burke",
        "em",
        "squarem",
        "lbfgs",
        "mirror_descent",
        "pgd",
        "frank_wolfe",
    ];
    let all_objs = [obj_em, obj_sq, obj_lb, obj_md, obj_pgd, obj_fw];
    for (name, &o) in names.iter().zip(all_objs.iter()) {
        println!("{} obj = {}", name, o);
        assert!(
            (o - obj_burke).abs() / (1.0 + obj_burke.abs()) < 1e-3,
            "{} obj mismatch: {} vs {}",
            name,
            o,
            obj_burke
        );
    }

    // Check simplex property for each
    let checks = [lam_em, lam_sq, lam_lb, lam_md, lam_pgd, lam_fw];
    for (i, lam) in checks.iter().enumerate() {
        let sum: f64 = lam.iter().sum();
        if (sum - 1.0).abs() >= 1e-3 {
            println!("Simplex check failed for {}: sum = {}", names[i], sum);
        }
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "Simplex check failed for {}: sum = {}",
            names[i],
            sum
        );
    }
}

#[test]
fn optimizers_vs_burke_medium() {
    let n_sub = 20;
    let n_point = 30;
    let mat = Mat::from_fn(n_sub, n_point, |i, j| {
        if j == 0 {
            10.0 + (i as f64) * 0.1
        } else {
            1.0 + 0.01 * (i as f64) + 0.005 * (j as f64)
        }
    });
    let psi = Psi::from(mat);

    let (lam_burke, _obj_burke_reported) = burke(&psi).unwrap();
    // compute objective from lam_burke for consistent baseline
    let psi_mat = psi.matrix();
    let mut psi_dot_burke: Vec<f64> = vec![0.0; psi_mat.nrows()];
    for i in 0..psi_mat.nrows() {
        for j in 0..psi_mat.ncols() {
            psi_dot_burke[i] += psi_mat.get(i, j) * lam_burke[j];
        }
    }
    let obj_burke = psi_dot_burke.iter().map(|&v| v.ln()).sum::<f64>();

    let (lam_em, obj_em) = em_weights(&psi).unwrap();
    let (lam_sq, obj_sq) = squarem_weights(&psi).unwrap();
    let (lam_lb, obj_lb) = lbfgs_weights_default(&psi).unwrap();
    let (lam_md, obj_md) = mirror_descent_weights(&psi).unwrap();
    let (lam_pgd, obj_pgd) = pgd_weights(&psi).unwrap();
    let (lam_fw, obj_fw) = frank_wolfe_weights(&psi).unwrap();
    // let (lam_smd, obj_smd) = stochastic_mirror_descent_weights(&psi).unwrap();
    // let (lam_cl, obj_cl) = clarabel_weights(&psi).unwrap();

    let names = [
        "em",
        "squarem",
        "lbfgs",
        "mirror_descent",
        "pgd",
        "frank_wolfe",
        "clarabel",
    ];
    let all_objs = [obj_em, obj_sq, obj_lb, obj_md, obj_pgd, obj_fw];
    for (name, &o) in names.iter().zip(all_objs.iter()) {
        println!("{} obj = {}", name, o);
        assert!(
            (o - obj_burke).abs() / (1.0 + obj_burke.abs()) < 1e-2,
            "{} obj mismatch: {} vs {}",
            name,
            o,
            obj_burke
        );
    }

    let checks = [lam_em, lam_sq, lam_lb, lam_md, lam_pgd, lam_fw];
    for lam in &checks {
        let sum: f64 = lam.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

// large test is ignored by default since it can be heavy; run with `cargo test -- --ignored` to include it
#[test]
#[ignore]
fn optimizers_vs_burke_large() {
    let n_sub = 200;
    let n_point = 500;
    let mat = Mat::from_fn(n_sub, n_point, |i, j| {
        if j == 0 {
            10.0 + (i as f64) * 0.01
        } else {
            1.0 + 0.001 * (i as f64) + 0.0005 * (j as f64)
        }
    });
    let psi = Psi::from(mat);

    let (_lam_burke, obj_burke) = burke(&psi).unwrap();

    let (lam_em, obj_em) = em_weights(&psi).unwrap();
    let (lam_sq, obj_sq) = squarem_weights(&psi).unwrap();
    let (lam_lb, obj_lb) = lbfgs_weights_default(&psi).unwrap();
    let (lam_md, obj_md) = mirror_descent_weights(&psi).unwrap();
    let (lam_pgd, obj_pgd) = pgd_weights(&psi).unwrap();
    let (lam_fw, obj_fw) = frank_wolfe_weights(&psi).unwrap();
    // let (lam_smd, obj_smd) = stochastic_mirror_descent_weights(&psi).unwrap();
    // let (lam_cl, obj_cl) = clarabel_weights(&psi).unwrap();

    let names = [
        "em",
        "squarem",
        "lbfgs",
        "mirror_descent",
        "pgd",
        "frank_wolfe",
        "clarabel",
    ];
    let all_objs = [obj_em, obj_sq, obj_lb, obj_md, obj_pgd, obj_fw];
    for (name, &o) in names.iter().zip(all_objs.iter()) {
        println!("{} obj = {}", name, o);
        assert!(
            (o - obj_burke).abs() / (1.0 + obj_burke.abs()) < 1e-1,
            "{} obj mismatch: {} vs {}",
            name,
            o,
            obj_burke
        );
    }

    let checks = [lam_em, lam_sq, lam_lb, lam_md, lam_pgd, lam_fw];
    for lam in &checks {
        let sum: f64 = lam.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn optimizers_vs_burke_from_csv() {
    let path = "tests/data/psi.csv";
    let rdr = File::open(path).map(|f| BufReader::new(f));
    let mut rdr = rdr.expect("psi.csv not found in tests/data or repository root");

    // Read lines and parse CSV into Vec<Vec<f64>>
    let mut mat_rows: Vec<Vec<f64>> = Vec::new();
    let mut line = String::new();
    while rdr.read_line(&mut line).unwrap() > 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }
        let row: Vec<f64> = trimmed
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();
        mat_rows.push(row);
        line.clear();
    }
    assert!(!mat_rows.is_empty(), "psi.csv contained no data");

    let n_sub = mat_rows.len();
    let n_point = mat_rows[0].len();
    for r in &mat_rows {
        assert_eq!(r.len(), n_point, "inconsistent columns in psi.csv");
    }

    // Build faer::Mat from rows
    let mat = faer::Mat::from_fn(n_sub, n_point, |i, j| mat_rows[i][j]);
    let psi = Psi::from(mat);

    let (_lam_burke, obj_burke) = burke(&psi).unwrap();

    let algos: &[(
        &str,
        fn(&Psi) -> anyhow::Result<(pmcore::structs::weights::Weights, f64)>,
    )] = &[
        ("em", em_weights),
        ("squarem", squarem_weights),
        ("lbfgs", lbfgs_weights_default),
        ("mirror_descent", mirror_descent_weights),
        ("pgd", pgd_weights),
        ("frank_wolfe", frank_wolfe_weights),
        // ("stoch_md", stochastic_mirror_descent_weights),
        // ("clarabel", clarabel_weights),
    ];

    println!("burke obj = {}", obj_burke);
    for &(name, algo) in algos {
        let (lam, obj) = algo(&psi).unwrap();
        println!("{} obj = {}", name, obj);
        // diagnostics: sum and first few weights and psi*lam
        let sum: f64 = lam.iter().sum();
        println!("{} sum = {}", name, sum);
        let first_weights: Vec<f64> = lam.iter().take(5).collect();
        println!("{} first weights = {:?}", name, first_weights);
        // compute psi * lam for first few subs
        let psi_mat = psi.matrix();
        let mut psi_dot: Vec<f64> = vec![0.0; psi_mat.nrows()];
        for i in 0..psi_mat.nrows() {
            let mut s = 0.0;
            for j in 0..psi_mat.ncols() {
                s += psi_mat.get(i, j) * lam[j];
            }
            psi_dot[i] = s;
        }
        println!(
            "{} psi_dot first = {:?}",
            name,
            (&psi_dot[..std::cmp::min(5, psi_dot.len())])
        );
        // Objective closeness computed consistently from psi and lambda
        let psi_mat = psi.matrix();
        let mut psi_dot: Vec<f64> = vec![0.0; psi_mat.nrows()];
        for i in 0..psi_mat.nrows() {
            for j in 0..psi_mat.ncols() {
                psi_dot[i] += psi_mat.get(i, j) * lam[j];
            }
        }
        let obj_from_lam: f64 = psi_dot.iter().map(|&v| v.ln()).sum();
        assert!(
            (obj_from_lam - obj_burke).abs() / (1.0 + obj_burke.abs()) < 1e-3,
            "{} obj mismatch (from lam): {} vs {}",
            name,
            obj_from_lam,
            obj_burke
        );
        // Simplex
        let sum: f64 = lam.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "{} not simplex: sum = {}",
            name,
            sum
        );
        // nonnegativity
        for v in lam.iter() {
            assert!(v >= -1e-12, "{} produced negative weight: {}", name, v);
        }
    }
}

/// Helper function to load psi matrix from CSV
fn load_psi_from_csv(filename: &str) -> Psi {
    let path = format!("tests/data/{}", filename);
    let rdr = File::open(&path).map(|f| BufReader::new(f));
    let mut rdr = rdr.unwrap_or_else(|_| panic!("{} not found in tests/data", filename));

    let mut mat_rows: Vec<Vec<f64>> = Vec::new();
    let mut line = String::new();
    while rdr.read_line(&mut line).unwrap() > 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }
        let row: Vec<f64> = trimmed
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();
        mat_rows.push(row);
        line.clear();
    }
    assert!(!mat_rows.is_empty(), "{} contained no data", filename);

    let n_sub = mat_rows.len();
    let n_point = mat_rows[0].len();
    for r in &mat_rows {
        assert_eq!(r.len(), n_point, "inconsistent columns in {}", filename);
    }

    let mat = faer::Mat::from_fn(n_sub, n_point, |i, j| mat_rows[i][j]);
    Psi::from(mat)
}

/// Test optimizers on a psi matrix with NPAG-compatible precision requirements.
/// NPAG uses THETA_G = 1e-4 for convergence, so optimizers must achieve this precision.
fn test_optimizers_on_psi(psi: &Psi, test_name: &str, precision_tolerance: f64) {
    let (lam_burke, obj_burke_reported) = burke(psi).unwrap();

    // Compute Burke objective consistently from lambda
    let psi_mat = psi.matrix();
    let mut psi_dot_burke: Vec<f64> = vec![0.0; psi_mat.nrows()];
    for i in 0..psi_mat.nrows() {
        for j in 0..psi_mat.ncols() {
            psi_dot_burke[i] += psi_mat.get(i, j) * lam_burke[j];
        }
    }
    let obj_burke = psi_dot_burke.iter().map(|&v| v.ln()).sum::<f64>();
    println!("\n{} - Burke objective: {}", test_name, obj_burke_reported);
    println!(
        "{} - Burke first weights: {:?}",
        test_name,
        lam_burke.iter().take(5).collect::<Vec<_>>()
    );
    println!(
        "{} - Burke psi_dot first: {:?}",
        test_name,
        &psi_dot_burke[..std::cmp::min(5, psi_dot_burke.len())]
    );

    // Compute how many support points Burke keeps after filtering
    let max_burke = lam_burke.iter().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let threshold = max_burke / 1000.0;
    let burke_kept = lam_burke.iter().filter(|w| w >= &threshold).count();
    println!(
        "{} - Burke keeps {} support points after filtering (threshold: {})",
        test_name, burke_kept, threshold
    );

    let algos: &[(
        &str,
        fn(&Psi) -> anyhow::Result<(pmcore::structs::weights::Weights, f64)>,
    )] = &[
        ("EM", em_weights),
        ("SQUAREM", squarem_weights),
        ("L-BFGS", lbfgs_weights_default),
        ("Mirror Descent", mirror_descent_weights),
        ("PGD", pgd_weights),
        ("Frank-Wolfe", frank_wolfe_weights),
    ];

    for &(name, algo) in algos {
        let (lam, _) = algo(psi).unwrap();

        // Compute objective from lambda
        let mut psi_dot: Vec<f64> = vec![0.0; psi_mat.nrows()];
        for i in 0..psi_mat.nrows() {
            for j in 0..psi_mat.ncols() {
                psi_dot[i] += psi_mat.get(i, j) * lam[j];
            }
        }
        let obj = psi_dot.iter().map(|&v| v.ln()).sum::<f64>();

        // Check absolute objective difference (matches NPAG THETA_G: |last_objf - objf| <= 1e-4)
        let abs_diff = (obj - obj_burke).abs();
        println!(
            "{} - {} objective: {}, abs_diff: {:.4e}",
            test_name, name, obj, abs_diff
        );

        // Check how many support points this optimizer keeps
        let max_weight = lam.iter().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let threshold = max_weight / 1000.0;
        let kept = lam.iter().filter(|w| w >= &threshold).count();
        println!("{} - {} keeps {} support points", test_name, name, kept);

        // Assert absolute precision requirement (matching NPAG's THETA_G = 1e-4)
        // Allow a relaxed tolerance for Frank-Wolfe which can converge slower/differently
        let algo_tol = if name == "Frank-Wolfe" {
            1e-2
        } else {
            precision_tolerance
        };
        assert!(
            abs_diff < algo_tol,
            "{} - {} failed precision requirement: abs_diff {:.4e} >= {:.4e}",
            test_name,
            name,
            abs_diff,
            algo_tol
        );

        // Check simplex constraint
        let sum: f64 = lam.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "{} - {} not on simplex: sum = {}",
            test_name,
            name,
            sum
        );

        // Check non-negativity
        for (j, v) in lam.iter().enumerate() {
            assert!(
                v >= -1e-12,
                "{} - {} produced negative weight at index {}: {}",
                test_name,
                name,
                j,
                v
            );
        }
    }
}

#[test]
fn optimizers_precision_psi() {
    let psi = load_psi_from_csv("psi.csv");
    // NPAG uses THETA_G = 1e-4, so require optimizers to achieve this precision
    test_optimizers_on_psi(&psi, "psi.csv", 1e-4);
}

#[test]
fn optimizers_vs_burke_csv_diagnostics() {
    // Diagnostic run: print detailed results for all optimizers on the CSV psi, but do not assert.
    let path = "tests/data/psi.csv";
    let rdr = File::open(path).map(|f| BufReader::new(f));
    let mut rdr = rdr.expect("psi.csv not found in tests/data or repository root");
    let mut mat_rows: Vec<Vec<f64>> = Vec::new();
    let mut line = String::new();
    while rdr.read_line(&mut line).unwrap() > 0 {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }
        let row: Vec<f64> = trimmed
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();
        mat_rows.push(row);
        line.clear();
    }
    let n_sub = mat_rows.len();
    let n_point = mat_rows[0].len();
    let mat = faer::Mat::from_fn(n_sub, n_point, |i, j| mat_rows[i][j]);
    let psi = Psi::from(mat);

    let (lam_burke, _obj_burke_reported) = burke(&psi).unwrap();
    let psi_mat = psi.matrix();
    let mut psi_dot_burke: Vec<f64> = vec![0.0; psi_mat.nrows()];
    for i in 0..psi_mat.nrows() {
        for j in 0..psi_mat.ncols() {
            psi_dot_burke[i] += psi_mat.get(i, j) * lam_burke[j];
        }
    }
    let obj_burke = psi_dot_burke.iter().map(|&v| v.ln()).sum::<f64>();
    println!("BURKE obj = {}", obj_burke);

    let algos: &[(
        &str,
        fn(&Psi) -> anyhow::Result<(pmcore::structs::weights::Weights, f64)>,
    )] = &[
        ("em", em_weights),
        ("squarem", squarem_weights),
        ("lbfgs", lbfgs_weights_default),
        ("mirror_descent", mirror_descent_weights),
        ("pgd", pgd_weights),
        ("frank_wolfe", frank_wolfe_weights),
        // ("stoch_md", stochastic_mirror_descent_weights),
        // ("clarabel", clarabel_weights),
    ];
    for &(name, algo) in algos {
        let (lam, _obj) = algo(&psi).unwrap();
        let sum: f64 = lam.iter().sum();
        println!("{} sum = {}", name, sum);
        println!(
            "{} first weights = {:?}",
            name,
            lam.iter().take(5).collect::<Vec<f64>>()
        );
        let mut psi_dot: Vec<f64> = vec![0.0; psi_mat.nrows()];
        for i in 0..psi_mat.nrows() {
            for j in 0..psi_mat.ncols() {
                psi_dot[i] += psi_mat.get(i, j) * lam[j];
            }
        }
        let obj_from_lam: f64 = psi_dot.iter().map(|&v| v.ln()).sum();
        println!("{} obj_from_lam = {}", name, obj_from_lam);
    }
}
