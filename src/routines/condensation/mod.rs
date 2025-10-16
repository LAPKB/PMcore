use crate::algorithms::npag::{burke, qr};
use crate::structs::psi::Psi;
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;
use anyhow::Result;

/// Apply lambda filtering and QR decomposition to condense support points
///
/// This implements the condensation step used in NPAG algorithms:
/// 1. Filter support points by lambda (probability) threshold
/// 2. Apply QR decomposition to remove linearly dependent points
/// 3. Recalculate weights with Burke's IPM on filtered points
///
/// # Arguments
///
/// * `theta` - Support points matrix
/// * `psi` - Likelihood matrix (subjects × support points)
/// * `lambda` - Initial probability weights for support points
/// * `lambda_threshold` - Minimum lambda value (relative to max) to keep a point
/// * `qr_threshold` - QR decomposition threshold for linear independence (typically 1e-8)
///
/// # Returns
///
/// Returns filtered theta, psi, and recalculated weights, plus the objective function value
pub fn condense_support_points(
    theta: &Theta,
    psi: &Psi,
    lambda: &Weights,
    lambda_threshold: f64,
    qr_threshold: f64,
) -> Result<(Theta, Psi, Weights, f64)> {
    let mut filtered_theta = theta.clone();
    let mut filtered_psi = psi.clone();

    // Step 1: Lambda filtering
    let max_lambda = lambda.iter().fold(f64::NEG_INFINITY, |acc, x| x.max(acc));

    let threshold = max_lambda * lambda_threshold;

    let keep_lambda: Vec<usize> = lambda
        .iter()
        .enumerate()
        .filter(|(_, lam)| *lam > threshold)
        .map(|(i, _)| i)
        .collect();

    let initial_count = theta.matrix().nrows();
    let after_lambda = keep_lambda.len();

    if initial_count != after_lambda {
        tracing::debug!(
            "Lambda filtering ({:.0e} × max): {} -> {} support points",
            lambda_threshold,
            initial_count,
            after_lambda
        );
    }

    filtered_theta.filter_indices(&keep_lambda);
    filtered_psi.filter_column_indices(&keep_lambda);

    // Step 2: QR decomposition filtering
    let (r, perm) = qr::qrd(&filtered_psi)?;

    let mut keep_qr = Vec::<usize>::new();

    // The minimum between the number of subjects and the actual number of support points
    let keep_n = filtered_psi
        .matrix()
        .ncols()
        .min(filtered_psi.matrix().nrows());

    for i in 0..keep_n {
        let test = r.col(i).norm_l2();
        let r_diag_val = r.get(i, i);
        let ratio = r_diag_val / test;
        if ratio.abs() >= qr_threshold {
            keep_qr.push(*perm.get(i).unwrap());
        }
    }

    let after_qr = keep_qr.len();

    if after_lambda != after_qr {
        tracing::debug!(
            "QR decomposition (threshold {:.0e}): {} -> {} support points",
            qr_threshold,
            after_lambda,
            after_qr
        );
    }

    filtered_theta.filter_indices(&keep_qr);
    filtered_psi.filter_column_indices(&keep_qr);

    // Step 3: Recalculate weights with Burke's IPM
    let (final_weights, objf) = burke(&filtered_psi)?;

    tracing::debug!(
        "Condensation complete: {} -> {} support points (objective: {:.4})",
        initial_count,
        filtered_theta.matrix().nrows(),
        objf
    );

    Ok((filtered_theta, filtered_psi, final_weights, objf))
}
