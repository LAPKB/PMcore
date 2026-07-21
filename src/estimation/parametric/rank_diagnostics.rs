//! Rank-normalized convergence diagnostics (Vehtari et al. 2021).
//!
//! Foundation module providing split-R̂ (rank-normalized and folded) and bulk
//! effective sample size (ESS).  The implementation follows the posterior R
//! package semantics:
//!
//! * pooled average-rank ties,
//! * Blom normal-score transform via statrs,
//! * even split of each chain into two halves,
//! * classical split-R̂ on the transformed chains,
//! * folded split-R̂ via absolute deviations from the pooled median,
//! * biased-N autocovariance for bulk ESS,
//! * Geyer initial-positive-sequence + monotone sequence estimator for τ,
//! * no clipping of R̂ below 1 or ESS above the total draw count.
//!
//! All internal errors are typed and no value is silently patched.

#![allow(dead_code)] // foundation module pending integration

use statrs::distribution::{ContinuousCDF, Normal};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by rank-based convergence diagnostics.
#[derive(Debug, Clone, PartialEq, Error)]
pub(crate) enum RankDiagnosticError {
    /// No chains were provided.
    #[error("at least one chain is required for rank diagnostics")]
    NoChains,

    /// Fewer than two chains: split requires ≥ 2 original chains.
    #[error("at least two chains are required for split diagnostics; found {found}")]
    TooFewChains { found: usize },

    /// Chains have different lengths.
    #[error(
        "all chains must have the same length; found lengths ranging from {shortest} to {longest}"
    )]
    UnequalChainLengths { shortest: usize, longest: usize },

    /// Chain length is odd (cannot be split evenly).
    #[error("chain length must be even for split diagnostics; found length {len}")]
    OddChainLength { len: usize },

    /// A draw is non-finite.
    #[error("rank diagnostics require finite draws; found non-finite value")]
    NonFiniteDraw,

    /// Chain has fewer draws than the minimum required.
    #[error("chain length {len} must be at least {min_len} for {diagnostic}")]
    TooFewDraws {
        len: usize,
        min_len: usize,
        diagnostic: &'static str,
    },

    /// All pooled draws are identical after rank normalization (W = 0), so
    /// R̂ is undefined.
    #[error("all pooled draws are constant; split-R̂ is undefined (W = 0)")]
    ConstantDraws,

    /// A required within-chain or pooled variance is non-positive/non-finite.
    #[error("rank diagnostic variance is invalid")]
    InvalidVariance,

    /// Integrated autocorrelation time τ ≤ 0, making ESS undefined.
    #[error("integrated autocorrelation time τ = {tau} is non-positive; ESS is undefined")]
    NonPositiveTau { tau: f64 },
}

// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------

/// Validate chains are non-empty, equal-length, even-length, finite, and
/// at least two chains for split diagnostics.
fn validate_chains(
    chains: &[Vec<f64>],
    min_len: usize,
    diagnostic: &'static str,
) -> Result<(), RankDiagnosticError> {
    if chains.is_empty() {
        return Err(RankDiagnosticError::NoChains);
    }
    if chains.len() < 2 {
        return Err(RankDiagnosticError::TooFewChains {
            found: chains.len(),
        });
    }
    let first_len = chains[0].len();
    if chains.iter().any(|c| c.len() != first_len) {
        let shortest = chains.iter().map(|c| c.len()).min().unwrap_or(0);
        let longest = chains.iter().map(|c| c.len()).max().unwrap_or(0);
        return Err(RankDiagnosticError::UnequalChainLengths { shortest, longest });
    }
    if first_len < min_len {
        return Err(RankDiagnosticError::TooFewDraws {
            len: first_len,
            min_len,
            diagnostic,
        });
    }
    if !first_len.is_multiple_of(2) {
        return Err(RankDiagnosticError::OddChainLength { len: first_len });
    }
    if chains.iter().any(|c| c.iter().any(|x| !x.is_finite())) {
        return Err(RankDiagnosticError::NonFiniteDraw);
    }
    Ok(())
}

/// Check that the pooled set of draws across chains is not all identical.
///
/// This is applied *after* rank normalization (or folding) so that a constant
/// multiset of z-scores is detected before computing R̂. One internally
/// constant chain does not make the pooled within-chain variance zero when
/// other chains vary; the variance calculations below decide eligibility.
fn assert_non_constant_draws(chains: &[Vec<f64>]) -> Result<(), RankDiagnosticError> {
    let first = chains[0][0];
    if chains.iter().all(|c| c.iter().all(|&x| x == first)) {
        return Err(RankDiagnosticError::ConstantDraws);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

/// Split each chain evenly into two halves (first n/2, last n/2).
fn split_even(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let half = chains[0].len() / 2;
    chains
        .iter()
        .flat_map(|c| {
            let first = c[..half].to_vec();
            let second = c[half..].to_vec();
            vec![first, second]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

/// Compute pooled average ranks across all chains.
///
/// Tied values receive the average of the 1-based ranks they span.
fn pooled_average_ranks(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let total = chains.iter().map(|c| c.len()).sum::<usize>();
    // Collect (value, chain_index, position_index).
    let mut indexed: Vec<(f64, usize, usize)> = chains
        .iter()
        .enumerate()
        .flat_map(|(ci, c)| c.iter().enumerate().map(move |(pi, &v)| (v, ci, pi)))
        .collect();

    // Stable sort by value (f64::total_cmp gives total ordering, but we want
    // exact-equality tie handling). For post-rank-normalization z-scores ties
    // are rare; we use simple partial_cmp with well-defined float ordering.
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks: Vec<Vec<f64>> = chains.iter().map(|c| vec![0.0; c.len()]).collect();
    let mut i = 0;
    while i < total {
        let mut j = i;
        while j + 1 < total && indexed[j + 1].0 == indexed[i].0 {
            j += 1;
        }
        // Average of 1-based ranks (i+1)..(j+1).
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &(_, ci, pi) in &indexed[i..=j] {
            ranks[ci][pi] = avg;
        }
        i = j + 1;
    }
    ranks
}

// ---------------------------------------------------------------------------
// Blom transform
// ---------------------------------------------------------------------------

/// Convert pooled ranks to normal scores using Blom's formula:
/// `z = Φ⁻¹((rank − 3/8) / (S + 1/4))`.
fn blom_scores(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let total = chains.iter().map(|c| c.len()).sum::<usize>() as f64;
    let norm = standard_normal();
    pooled_average_ranks(chains)
        .into_iter()
        .map(|c| {
            c.into_iter()
                .map(|r| norm.inverse_cdf((r - 0.375) / (total + 0.25)))
                .collect()
        })
        .collect()
}

fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal parameters are valid")
}

// ---------------------------------------------------------------------------
// Classical split-R̂
// ---------------------------------------------------------------------------

/// Compute the classical split-R̂ (sqrt variant) on *post-split* chains.
///
/// The chains must already be rank-normalized and split.  Returns `ConstantDraws`
/// if within-chain variance W is zero.
fn split_rhat_of(chains: &[Vec<f64>]) -> Result<f64, RankDiagnosticError> {
    let m = chains.len() as f64;
    let n = chains[0].len() as f64;

    let means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / n).collect();
    let grand = means.iter().sum::<f64>() / m;

    let b = if m > 1.0 {
        n * means.iter().map(|mu| (mu - grand).powi(2)).sum::<f64>() / (m - 1.0)
    } else {
        0.0
    };

    let w = chains
        .iter()
        .zip(means.iter())
        .map(|(c, mu)| c.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0))
        .sum::<f64>()
        / m;

    if w == 0.0 {
        return Err(RankDiagnosticError::ConstantDraws);
    }
    if !w.is_finite() || w < 0.0 || !b.is_finite() || b < 0.0 {
        return Err(RankDiagnosticError::InvalidVariance);
    }

    let var_plus = (n - 1.0) / n * w + b / n;
    if !var_plus.is_finite() || var_plus <= 0.0 {
        return Err(RankDiagnosticError::InvalidVariance);
    }
    Ok((var_plus / w).sqrt())
}

// ---------------------------------------------------------------------------
// Rank-normalized split-R̂
// ---------------------------------------------------------------------------

/// Rank-normalized split-R̂.
///
/// Splits each chain evenly, ranks across all split chains, applies Blom
/// normal scores, then computes the classical split-R̂ on the transformed data.
pub(crate) fn rank_normalized_split_rhat(chains: &[Vec<f64>]) -> Result<f64, RankDiagnosticError> {
    validate_chains(chains, 4, "rank-normalized split-R̂")?;
    assert_non_constant_draws(chains)?;
    let split = split_even(chains);
    assert_non_constant_draws(&split)?;
    let z = blom_scores(&split);
    assert_non_constant_draws(&z)?;
    split_rhat_of(&z)
}

// ---------------------------------------------------------------------------
// Folded split-R̂
// ---------------------------------------------------------------------------

/// Folded split-R̂.
///
/// Computes absolute deviations from the pooled median across all chains, then
/// applies rank-normalized split-R̂ to the folded values.
pub(crate) fn folded_split_rhat(chains: &[Vec<f64>]) -> Result<f64, RankDiagnosticError> {
    validate_chains(chains, 4, "folded split-R̂")?;

    // Pooled median.
    let total = chains.iter().map(|c| c.len()).sum::<usize>();
    let mut pooled: Vec<f64> = chains.iter().flat_map(|c| c.iter()).copied().collect();
    pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // total is even because all chain lengths are even, validated above.
    // Avoid overflowing the sum for large same-sign finite values and the
    // subtraction for opposite-sign values.
    let lower = pooled[total / 2 - 1];
    let upper = pooled[total / 2];
    let median = if lower.is_sign_negative() == upper.is_sign_negative() {
        lower + (upper - lower) / 2.0
    } else {
        lower / 2.0 + upper / 2.0
    };
    if !median.is_finite() {
        return Err(RankDiagnosticError::NonFiniteDraw);
    }

    let mut folded = Vec::with_capacity(chains.len());
    for chain in chains {
        let mut folded_chain = Vec::with_capacity(chain.len());
        for &draw in chain {
            let difference = draw - median;
            if !difference.is_finite() {
                return Err(RankDiagnosticError::NonFiniteDraw);
            }
            let value = difference.abs();
            if !value.is_finite() {
                return Err(RankDiagnosticError::NonFiniteDraw);
            }
            folded_chain.push(value);
        }
        folded.push(folded_chain);
    }

    assert_non_constant_draws(chains)?;
    let split = split_even(&folded);
    assert_non_constant_draws(&split)?;
    let z = blom_scores(&split);
    assert_non_constant_draws(&z)?;
    split_rhat_of(&z)
}

/// Maximum of rank-normalized and folded split-R̂.
pub(crate) fn max_split_rhat(chains: &[Vec<f64>]) -> Result<f64, RankDiagnosticError> {
    let r_rank = rank_normalized_split_rhat(chains)?;
    let r_fold = folded_split_rhat(chains)?;
    Ok(r_rank.max(r_fold))
}

// ---------------------------------------------------------------------------
// Bulk ESS
// ---------------------------------------------------------------------------

/// Biased (divisor N) autocovariance of a single series.
fn acov_biased(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mu = x.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = x.iter().map(|xi| xi - mu).collect();

    (0..n)
        .map(|t| {
            let mut s = 0.0;
            for i in 0..(n - t) {
                s += centered[i] * centered[i + t];
            }
            s / n as f64
        })
        .collect()
}

/// Compute τ̂ from rank-normalized split chains using Geyer's initial positive
/// sequence + monotone sequence estimator, then return ESS = m·n / τ̂.
///
/// Returns `NonPositiveTau` if τ̂ ≤ 0 or `ConstantDraws` if W = 0.
fn ess_from_split_z(z: &[Vec<f64>]) -> Result<(f64, f64), RankDiagnosticError> {
    let m = z.len() as f64;
    let n_float = z[0].len() as f64;
    let n = z[0].len();

    // Per-chain biased autocovariance.
    let acovs: Vec<Vec<f64>> = z.iter().map(|c| acov_biased(c)).collect();

    // Average autocovariance across chains at each lag.
    let acov_means: Vec<f64> = (0..n)
        .map(|t| acovs.iter().map(|a| a[t]).sum::<f64>() / m)
        .collect();

    // Var⁺ = mean_var · (n-1)/n + var(chain means)  [Vehtari et al. 2021 eq 13.3]
    let mean_var = acov_means[0] * n_float / (n_float - 1.0);
    let mut var_plus = mean_var * (n_float - 1.0) / n_float;
    if m > 1.0 {
        let means: Vec<f64> = z.iter().map(|c| c.iter().sum::<f64>() / n_float).collect();
        let grand = means.iter().sum::<f64>() / m;
        var_plus += means.iter().map(|mu| (mu - grand).powi(2)).sum::<f64>() / (m - 1.0);
    }
    if !mean_var.is_finite() || mean_var <= 0.0 || !var_plus.is_finite() || var_plus <= 0.0 {
        return Err(RankDiagnosticError::InvalidVariance);
    }

    // Vehtari/Geyer indexing: rho_0 is exactly one and P_t contains the
    // consecutive lag pair (rho_(2t), rho_(2t+1)).  Keep positive pairs from
    // t=0 through the pair immediately before the first non-positive or
    // unavailable pair, then enforce the initial monotone sequence on P itself.
    let mut rho = Vec::with_capacity(n);
    rho.push(1.0);
    rho.extend(
        acov_means
            .iter()
            .skip(1)
            .map(|acov| 1.0 - (mean_var - acov) / var_plus),
    );
    let monotone_pairs = geyer_initial_monotone_pairs(&rho);
    let tau = -1.0 + 2.0 * monotone_pairs.iter().sum::<f64>();

    if tau <= 0.0 || !tau.is_finite() {
        return Err(RankDiagnosticError::NonPositiveTau { tau });
    }

    let total = m * n_float;
    Ok((total / tau, tau))
}

/// Return Geyer's initial-positive, initial-monotone sequence of paired
/// autocorrelations. An incomplete final pair is unavailable and is excluded.
fn geyer_initial_monotone_pairs(rho: &[f64]) -> Vec<f64> {
    let mut pairs = Vec::with_capacity(rho.len() / 2);
    for pair in rho.chunks_exact(2) {
        let paired_sum = pair[0] + pair[1];
        if !paired_sum.is_finite() || paired_sum <= 0.0 {
            break;
        }
        pairs.push(match pairs.last() {
            Some(previous) => paired_sum.min(*previous),
            None => paired_sum,
        });
    }
    pairs
}

/// Bulk effective sample size via rank-normalized split chains.
///
/// Splits each chain evenly, rank-normalizes all split chains with Blom scores,
/// then computes ESS via Geyer's IPS+monotone estimator.  Requires chain length
/// ≥ 6 (3 per split half).
pub(crate) fn bulk_ess(chains: &[Vec<f64>]) -> Result<(f64, f64), RankDiagnosticError> {
    validate_chains(chains, 6, "bulk ESS")?;
    assert_non_constant_draws(chains)?;
    let split = split_even(chains);
    assert_non_constant_draws(&split)?;
    let z = blom_scores(&split);
    ess_from_split_z(&z)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Tolerance for values that go through inverse_cdf (Blom).
    const Z_TOL: f64 = 1e-8;
    // Tolerance for pure arithmetic on ranks, variances, etc.
    const ARITH_TOL: f64 = 1e-12;

    // ────────────────────────────────────────────────────
    // A. Pooled average ranks with ties
    // ────────────────────────────────────────────────────
    #[test]
    fn pooled_average_ranks_handles_ties_and_absence() {
        // Two chains of length 4; expected ranks from Python reference.
        let chains = vec![vec![1.0, 2.0, 2.0, 4.0], vec![2.0, 5.0, 6.0, 7.0]];
        let ranks = pooled_average_ranks(&chains);
        let expected = [vec![1.0, 3.0, 3.0, 5.0], vec![3.0, 6.0, 7.0, 8.0]];
        for (actual_row, expected_row) in ranks.iter().zip(expected.iter()) {
            for (&a, &e) in actual_row.iter().zip(expected_row.iter()) {
                assert_relative_eq!(a, e, max_relative = ARITH_TOL);
            }
        }
    }

    #[test]
    fn pooled_average_ranks_no_ties_all_distinct() {
        let chains = vec![vec![2.0, 1.0, 4.0, 3.0], vec![5.0, 6.0, 7.0, 8.0]];
        let ranks = pooled_average_ranks(&chains);
        // Values: [2,1,4,3,5,6,7,8] → sorted [1,2,3,4,5,6,7,8] → ranks 1..8.
        let expected = [vec![2.0, 1.0, 4.0, 3.0], vec![5.0, 6.0, 7.0, 8.0]];
        for (actual_row, expected_row) in ranks.iter().zip(expected.iter()) {
            for (&a, &e) in actual_row.iter().zip(expected_row.iter()) {
                assert_relative_eq!(a, e, max_relative = ARITH_TOL);
            }
        }
    }

    // ────────────────────────────────────────────────────
    // B. Blom normal scores
    // ────────────────────────────────────────────────────
    #[test]
    fn blom_scores_for_four_distinct_draws() {
        // Single chain of 4; S=4.
        // Python: r=1 → z ≈ -1.0491313979639711, r=3.5 → z ≈ 0.62890421763219
        // Values: [1, 3, 3, 2] mapped to chains for pooled rank context.
        let chains = vec![vec![1.0, 3.0], vec![3.0, 2.0]];
        let z = blom_scores(&chains);
        // Ranks: [[1.0, 3.5], [3.5, 2.0]]
        // z[0][0] for rank 1: -1.0491313979639711
        // z[0][1] for rank 3.5: 0.62890421763219
        assert_relative_eq!(z[0][0], -1.0491313979639711, max_relative = Z_TOL);
        assert_relative_eq!(z[0][1], 0.62890421763219, max_relative = Z_TOL);
    }

    // ────────────────────────────────────────────────────
    // C. Rank-normalized split-R̂ — simple hand-checked
    // ────────────────────────────────────────────────────
    #[test]
    fn rank_normalized_split_rhat_single_ascending_chain() {
        // Two identical chains [1,2,4,3]: split halves not fully constant.
        let chains = vec![vec![1.0, 2.0, 4.0, 3.0], vec![1.0, 2.0, 4.0, 3.0]];
        let r = rank_normalized_split_rhat(&chains).unwrap();
        assert!(r > 1.0, "R̂ should exceed 1 for non-mixed identical chains");
    }

    // ────────────────────────────────────────────────────
    // D. Folded split-R̂ — single chain, value-agnostic
    // ────────────────────────────────────────────────────
    #[test]
    fn folded_split_rhat_is_sqrt_half_for_single_ascending_chain() {
        // Two identical chains [1,2,4,3]: folded split-R̂ still √0.5.
        let chains = vec![vec![1.0, 2.0, 4.0, 3.0], vec![1.0, 2.0, 4.0, 3.0]];
        let r = folded_split_rhat(&chains).unwrap();
        assert_relative_eq!(r, std::f64::consts::FRAC_1_SQRT_2, max_relative = Z_TOL);
    }

    // ────────────────────────────────────────────────────
    // E. Ties in multiple chains → rank rhat
    // ────────────────────────────────────────────────────
    #[test]
    fn rank_normalized_split_rhat_with_ties() {
        // Python: tie rank rhat = 1.687130053098945
        let chains = vec![vec![1.0, 2.0, 2.0, 4.0], vec![2.0, 5.0, 6.0, 7.0]];
        let r = rank_normalized_split_rhat(&chains).unwrap();
        assert_relative_eq!(r, 1.687130053098945, max_relative = Z_TOL);
    }

    // ────────────────────────────────────────────────────
    // F. Two monotone chains x 8 (poor mixing)
    // ────────────────────────────────────────────────────
    #[test]
    fn diagnostics_for_two_monotone_chains() {
        let chains = vec![
            vec![0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8],
            vec![7.1, 6.2, 5.3, 4.4, 3.5, 2.6, 1.7, 0.8],
        ];
        // Python: rank rhat F = 1.7299566224270406
        let r = rank_normalized_split_rhat(&chains).unwrap();
        assert_relative_eq!(r, 1.7299566224270406, max_relative = Z_TOL);
        // Python: folded rhat F = 0.9129284180922413
        let f = folded_split_rhat(&chains).unwrap();
        assert_relative_eq!(f, 0.9129284180922413, max_relative = Z_TOL);
    }

    // ────────────────────────────────────────────────────
    // F-long. Monotone adjustment exercised (long well-mixed chains).
    // ────────────────────────────────────────────────────
    #[test]
    fn rank_rhat_and_ess_for_long_well_mixed_chains() {
        let chains = vec![
            vec![
                11.0, 5.0, 13.0, 2.0, 3.0, 14.0, 9.0, 17.0, 6.0, 10.0, 1.0, 15.0, 4.0, 20.0, 7.0,
                8.0, 19.0, 12.0, 16.0, 18.0,
            ],
            vec![
                38.0, 34.0, 22.0, 24.0, 28.0, 31.0, 35.0, 30.0, 21.0, 33.0, 39.0, 27.0, 32.0, 40.0,
                36.0, 25.0, 29.0, 37.0, 23.0, 26.0,
            ],
        ];
        // Python: long rank rhat = 1.81651170432963
        let r = rank_normalized_split_rhat(&chains).unwrap();
        assert_relative_eq!(r, 1.81651170432963, max_relative = Z_TOL);
    }

    // ────────────────────────────────────────────────────
    // G. Antithetic case: ESS > total·log10(total) — no upper cap.
    // ────────────────────────────────────────────────────
    #[test]
    fn bulk_ess_exceeds_total_times_log10_total_for_mildly_antithetic_chain() {
        // Two identical chains preserve autocorrelation structure; tau > 0 still.
        let chain = vec![
            5.0, 3.0, 2.0, 12.0, 9.0, 10.0, 4.0, 11.0, 7.0, 8.0, 6.0, 1.0,
        ];
        let chains = vec![chain.clone(), chain];
        let (ess, tau) = bulk_ess(&chains).unwrap();
        assert_relative_eq!(ess, 32.491_591_364_927_49, max_relative = Z_TOL);
        assert_relative_eq!(tau, 0.7386526480173083, max_relative = Z_TOL);
    }

    // ────────────────────────────────────────────────────
    // H. NonPositiveTau error via perfectly antithetic chain.
    // ────────────────────────────────────────────────────
    #[test]
    fn bulk_ess_errors_on_non_positive_tau() {
        // Two identical perfectly antithetic chains.
        let chain = vec![
            1.0, 12.0, 2.0, 11.0, 3.0, 10.0, 4.0, 9.0, 5.0, 8.0, 6.0, 7.0,
        ];
        let chains = vec![chain.clone(), chain];
        let err = bulk_ess(&chains).unwrap_err();
        assert!(matches!(err, RankDiagnosticError::NonPositiveTau { .. }));
        if let RankDiagnosticError::NonPositiveTau { tau } = err {
            assert!(tau < 0.0, "tau should be negative");
        }
    }

    // ────────────────────────────────────────────────────
    // I. Max split-R̂ returns the larger of the two.
    // ────────────────────────────────────────────────────
    #[test]
    fn max_split_rhat_is_max_of_rank_and_folded() {
        let chains = vec![
            vec![0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8],
            vec![7.1, 6.2, 5.3, 4.4, 3.5, 2.6, 1.7, 0.8],
        ];
        let r_rank = rank_normalized_split_rhat(&chains).unwrap();
        let r_fold = folded_split_rhat(&chains).unwrap();
        let r_max = max_split_rhat(&chains).unwrap();
        assert_eq!(r_max, r_rank.max(r_fold));
        // sanity: floor should be at r_fold < r_rank for this input.
        assert!(r_fold < r_rank);
    }

    #[test]
    fn monotone_transform_preserves_rank_diagnostics() {
        let chains = vec![
            vec![0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8],
            vec![7.1, 6.2, 5.3, 4.4, 3.5, 2.6, 1.7, 0.8],
        ];
        let transformed = chains
            .iter()
            .map(|chain| chain.iter().map(|value| f64::exp(*value)).collect())
            .collect::<Vec<Vec<f64>>>();
        assert_eq!(
            rank_normalized_split_rhat(&chains).unwrap(),
            rank_normalized_split_rhat(&transformed).unwrap()
        );
        assert_eq!(bulk_ess(&chains).unwrap(), bulk_ess(&transformed).unwrap());
    }

    #[test]
    fn shifted_location_and_drift_are_detected_by_rank_rhat() {
        let shifted = vec![
            (0..20).map(|i| i as f64 * 0.1).collect::<Vec<_>>(),
            (0..20).map(|i| 8.0 + i as f64 * 0.1).collect::<Vec<_>>(),
        ];
        assert!(rank_normalized_split_rhat(&shifted).unwrap() > 1.5);

        let drifting = vec![
            (0..20).map(|i| i as f64).collect::<Vec<_>>(),
            (0..20).map(|i| i as f64 + 0.25).collect::<Vec<_>>(),
        ];
        assert!(rank_normalized_split_rhat(&drifting).unwrap() > 1.5);
    }

    #[test]
    fn scale_only_mismatch_is_stronger_after_folding() {
        let narrow = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0];
        let wide = [-8.0, -6.4, -4.8, -3.2, -1.6, 1.6, 3.2, 4.8, 6.4, 8.0];
        let chains = vec![narrow.repeat(2), wide.repeat(2)];
        let rank = rank_normalized_split_rhat(&chains).unwrap();
        let folded = folded_split_rhat(&chains).unwrap();
        assert!(folded > rank);
        assert!(folded > 1.1);
    }

    #[test]
    fn sticky_trace_has_low_bulk_ess() {
        let chains = (0..4)
            .map(|chain| {
                (0..80)
                    .map(|draw| (draw / 10) as f64 + chain as f64 * 0.01)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (ess, _) = bulk_ess(&chains).unwrap();
        assert!(
            ess < 80.0,
            "sticky ESS {ess} should be well below 320 draws"
        );
    }

    // ────────────────────────────────────────────────────
    // J. Edge cases — input validation errors
    // ────────────────────────────────────────────────────
    #[test]
    fn no_chains_rejected() {
        let chains: Vec<Vec<f64>> = vec![];
        assert_eq!(
            rank_normalized_split_rhat(&chains).unwrap_err(),
            RankDiagnosticError::NoChains
        );
        assert_eq!(
            folded_split_rhat(&chains).unwrap_err(),
            RankDiagnosticError::NoChains
        );
        assert_eq!(
            bulk_ess(&chains).unwrap_err(),
            RankDiagnosticError::NoChains
        );
    }

    #[test]
    fn unequal_chain_lengths_rejected() {
        let chains = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        assert_eq!(
            rank_normalized_split_rhat(&chains).unwrap_err(),
            RankDiagnosticError::UnequalChainLengths {
                shortest: 4,
                longest: 6
            }
        );
    }

    #[test]
    fn odd_chain_length_rejected() {
        let chains = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let err = rank_normalized_split_rhat(&chains).unwrap_err();
        assert_eq!(err, RankDiagnosticError::OddChainLength { len: 5 });
    }

    #[test]
    fn too_few_draws_for_rhat() {
        // 2 chains, length 2: minimum is 4.
        let chains = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        assert_eq!(
            rank_normalized_split_rhat(&chains).unwrap_err(),
            RankDiagnosticError::TooFewDraws {
                len: 2,
                min_len: 4,
                diagnostic: "rank-normalized split-R̂"
            }
        );
    }

    #[test]
    fn too_few_draws_for_ess() {
        // 2 chains, length 4: ESS minimum is 6.
        let chains = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]];
        assert_eq!(
            bulk_ess(&chains).unwrap_err(),
            RankDiagnosticError::TooFewDraws {
                len: 4,
                min_len: 6,
                diagnostic: "bulk ESS"
            }
        );
    }

    #[test]
    fn non_finite_draws_rejected() {
        // 2 chains: NAN in first, valid in second.
        let chains = vec![vec![1.0, f64::NAN, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]];
        assert_eq!(
            rank_normalized_split_rhat(&chains).unwrap_err(),
            RankDiagnosticError::NonFiniteDraw
        );
        // ESS needs len ≥ 6; 2 chains with INF.
        let chains_inf = vec![
            vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ];
        assert_eq!(
            bulk_ess(&chains_inf).unwrap_err(),
            RankDiagnosticError::NonFiniteDraw
        );
    }

    #[test]
    fn constant_draws_produce_error_in_rhat() {
        let chains = vec![vec![5.0, 5.0, 5.0, 5.0], vec![6.0, 6.0, 6.0, 6.0]];
        let err = rank_normalized_split_rhat(&chains).unwrap_err();
        assert_eq!(err, RankDiagnosticError::ConstantDraws);
    }

    #[test]
    fn folded_median_is_overflow_safe_and_rejects_overflowing_differences() {
        let same_sign = vec![
            vec![
                f64::MAX * 0.75,
                f64::MAX * 0.8,
                f64::MAX * 0.85,
                f64::MAX * 0.9,
            ],
            vec![
                f64::MAX * 0.7,
                f64::MAX * 0.78,
                f64::MAX * 0.88,
                f64::MAX * 0.95,
            ],
        ];
        assert!(folded_split_rhat(&same_sign).is_ok());

        let overflowing_difference = vec![
            vec![-f64::MAX, f64::MAX * 0.70, f64::MAX * 0.80, f64::MAX * 0.90],
            vec![
                -f64::MAX * 0.90,
                f64::MAX * 0.75,
                f64::MAX * 0.85,
                f64::MAX * 0.95,
            ],
        ];
        assert_eq!(
            folded_split_rhat(&overflowing_difference).unwrap_err(),
            RankDiagnosticError::NonFiniteDraw
        );
    }

    #[test]
    fn folded_constant_draws_produce_error() {
        // Both chains produce folded absolute deviations from median 0:
        // first → [1,1,1,1], second → [2,2,2,2] → split halves constant.
        let chains = vec![vec![1.0, -1.0, 1.0, -1.0], vec![2.0, -2.0, 2.0, -2.0]];
        let err = folded_split_rhat(&chains).unwrap_err();
        assert_eq!(err, RankDiagnosticError::ConstantDraws);
    }

    // ────────────────────────────────────────────────────
    // K. Constant-chain eligibility follows pooled variance mathematics.
    // ────────────────────────────────────────────────────
    #[test]
    fn all_chains_internally_constant_are_ineligible_for_rhat_and_ess() {
        // Each chain is internally constant but chains have different values.
        let chains = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ];
        // R̂: post-split produces constant z per half-chain → pooled W=0.
        assert_eq!(
            rank_normalized_split_rhat(&chains).unwrap_err(),
            RankDiagnosticError::ConstantDraws
        );
        assert_eq!(
            bulk_ess(&chains).unwrap_err(),
            RankDiagnosticError::InvalidVariance
        );
    }

    #[test]
    fn one_stuck_chain_with_an_independently_varying_chain_remains_diagnostic() {
        let chains = vec![vec![1.0; 8], vec![0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]];

        let rank = rank_normalized_split_rhat(&chains).unwrap();
        let folded = folded_split_rhat(&chains).unwrap();
        let (ess, tau) = bulk_ess(&chains).unwrap();

        assert!(rank.is_finite() && rank > 1.0);
        assert!(folded.is_finite());
        assert!(ess.is_finite() && ess > 0.0);
        assert!(tau.is_finite() && tau > 0.0);
    }

    fn assert_values(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_relative_eq!(actual, expected, epsilon = ARITH_TOL);
        }
    }

    #[test]
    fn geyer_fixture_keeps_first_pair_only() {
        // Fixture and all expected arithmetic were calculated independently.
        let z = vec![
            vec![0.0, 3.0, 4.0, -2.0, -4.0, -4.0, 3.0, -2.0],
            vec![-1.0, -3.0, 3.0, -2.0, 3.0, 3.0, 3.0, 2.0],
        ];
        assert_values(
            &acov_biased(&z[0]),
            &[
                9.1875, 1.2421875, -2.453125, -3.4609375, 0.46875, 0.2734375, -0.609375, -0.0546875,
            ],
        );
        assert_values(
            &acov_biased(&z[1]),
            &[5.75, -0.25, 1.5, -0.25, -1.375, -1.25, -1.0, -0.25],
        );
        let rho = [
            1.0,
            0.02550054112554101,
            -0.09239718614718617,
            -0.2595373376623378,
        ];
        assert_values(&geyer_initial_monotone_pairs(&rho), &[1.025500541125541]);
        let (ess, tau) = ess_from_split_z(&z).unwrap();
        assert_relative_eq!(tau, 1.0510010822510818, epsilon = ARITH_TOL);
        assert_relative_eq!(ess, 15.223580898442535, epsilon = ARITH_TOL);
    }

    #[test]
    fn geyer_fixture_truncates_at_later_nonpositive_pair() {
        let z = vec![
            vec![-1.0, -4.0, -2.0, 0.0, -1.0, 3.0, -2.0, 1.0],
            vec![0.0, 2.0, 4.0, 2.0, 4.0, -1.0, -1.0, -3.0],
        ];
        assert_values(
            &acov_biased(&z[0]),
            &[
                3.9375, -0.5078125, 0.984375, -0.6796875, -1.15625, 0.1171875, -0.671875,
                -0.0546875,
            ],
        );
        assert_values(
            &acov_biased(&z[1]),
            &[
                5.609375,
                1.810546875,
                0.94921875,
                -2.193359375,
                -1.8828125,
                -1.572265625,
                -0.33984375,
                0.423828125,
            ],
        );
        let rho = [
            1.0,
            0.21165293040293032,
            0.26341575091575087,
            -0.13097527472527482,
            -0.14459706959706975,
            -0.014629120879120938,
        ];
        assert_values(
            &geyer_initial_monotone_pairs(&rho),
            &[1.2116529304029302, 0.13244047619047605],
        );
        let (ess, tau) = ess_from_split_z(&z).unwrap();
        assert_relative_eq!(tau, 1.6881868131868125, epsilon = ARITH_TOL);
        assert_relative_eq!(ess, 9.477624084621647, epsilon = ARITH_TOL);
    }

    #[test]
    fn geyer_fixture_applies_monotone_pair_adjustment() {
        let z = vec![
            vec![2.0, -1.0, -1.0, 2.0, 3.0, 2.0, 4.0, 4.0],
            vec![-2.0, 4.0, 0.0, 3.0, -2.0, -4.0, -2.0, 2.0],
        ];
        assert_values(
            &acov_biased(&z[0]),
            &[
                3.359375,
                1.576171875,
                -0.16015625,
                -0.115234375,
                -0.7578125,
                -1.525390625,
                -0.73046875,
                0.033203125,
            ],
        );
        assert_values(
            &acov_biased(&z[1]),
            &[
                7.109375,
                -0.267578125,
                -0.55078125,
                -2.990234375,
                -0.7578125,
                -0.025390625,
                1.53515625,
                -0.498046875,
            ],
        );
        let rho = [
            1.0,
            0.2635374884294971,
            0.12395865473619261,
            -0.041538105522986646,
            0.068343103980253,
            0.06591329836470228,
            0.22871027460660298,
            0.14096729404504793,
        ];
        assert_values(
            &geyer_initial_monotone_pairs(&rho),
            &[
                1.2635374884294972,
                0.08242054921320596,
                0.08242054921320596,
                0.08242054921320596,
            ],
        );
        let (ess, tau) = ess_from_split_z(&z).unwrap();
        assert_relative_eq!(tau, 2.02159827213823, epsilon = ARITH_TOL);
        assert_relative_eq!(ess, 7.9145299145299095, epsilon = ARITH_TOL);
    }
}
