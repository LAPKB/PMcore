#[path = "../src/estimation/parametric/rank_diagnostics.rs"]
mod rank_diagnostics;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rank_diagnostics::{bulk_ess, folded_split_rhat, rank_normalized_split_rhat};

const RHO: f64 = 0.995;
const COMPONENT_SCALE: f64 = 0.2;
const BLOCK_SCALE: f64 = 1.0;
const WARMUP: usize = 250;
const RETAINED: usize = 1_500;

#[derive(Default)]
struct Counts {
    block_proposals: usize,
    block_accepts: usize,
    component_proposals: usize,
    component_accepts: usize,
}

fn log_target(eta: [f64; 2]) -> f64 {
    -0.5 / (1.0 - RHO * RHO) * (eta[0] * eta[0] - 2.0 * RHO * eta[0] * eta[1] + eta[1] * eta[1])
}

fn accept(current: [f64; 2], proposal: [f64; 2], uniform: f64) -> bool {
    let log_ratio = log_target(proposal) - log_target(current);
    log_ratio >= 0.0 || uniform.ln() < log_ratio
}

fn block_proposal(current: [f64; 2], z: [f64; 2], scale: f64) -> [f64; 2] {
    let conditional_sd = (1.0 - RHO * RHO).sqrt();
    [
        current[0] + scale * z[0],
        current[1] + scale * (RHO * z[0] + conditional_sd * z[1]),
    ]
}

fn component_sweep(state: &mut [f64; 2], rng: &mut StdRng, counts: &mut Counts) {
    for coordinate in 0..2 {
        let z: f64 = StandardNormal.sample(rng);
        let mut proposal = *state;
        proposal[coordinate] += COMPONENT_SCALE * z;
        counts.component_proposals += 1;
        if accept(*state, proposal, rng.random()) {
            *state = proposal;
            counts.component_accepts += 1;
        }
    }
}

fn transition(state: &mut [f64; 2], rng: &mut StdRng, block: bool, counts: &mut Counts) {
    if block {
        let proposal = block_proposal(
            *state,
            [StandardNormal.sample(rng), StandardNormal.sample(rng)],
            BLOCK_SCALE,
        );
        counts.block_proposals += 1;
        if accept(*state, proposal, rng.random()) {
            *state = proposal;
            counts.block_accepts += 1;
        }
    }
    component_sweep(state, rng, counts);
}

fn run_chains(block: bool) -> (Vec<Vec<[f64; 2]>>, Counts) {
    let starts = [[-5.0, -5.0], [-2.0, -2.0], [2.0, 2.0], [5.0, 5.0]];
    let mut retained = Vec::with_capacity(4);
    let mut total = Counts::default();
    for (chain, start) in starts.into_iter().enumerate() {
        let mut state = start;
        let mut rng = StdRng::seed_from_u64(0x4e37_2026 + chain as u64);
        let mut counts = Counts::default();
        for _ in 0..WARMUP {
            transition(&mut state, &mut rng, block, &mut counts);
        }
        let mut draws = Vec::with_capacity(RETAINED);
        for _ in 0..RETAINED {
            transition(&mut state, &mut rng, block, &mut counts);
            draws.push(state);
        }
        total.block_proposals += counts.block_proposals;
        total.block_accepts += counts.block_accepts;
        total.component_proposals += counts.component_proposals;
        total.component_accepts += counts.component_accepts;
        retained.push(draws);
    }
    (retained, total)
}

fn diagnostics(chains: &[Vec<[f64; 2]>]) -> (f64, f64) {
    let features = [
        chains
            .iter()
            .map(|chain| chain.iter().map(|eta| eta[0]).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        chains
            .iter()
            .map(|chain| chain.iter().map(|eta| eta[1]).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .map(|eta| (eta[0] + eta[1]) / (2.0 * (1.0 + RHO)).sqrt())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    ];
    let mut max_rhat = 0.0_f64;
    let mut min_ess = f64::INFINITY;
    for feature in features {
        max_rhat = max_rhat
            .max(rank_normalized_split_rhat(&feature).unwrap())
            .max(folded_split_rhat(&feature).unwrap());
        min_ess = min_ess.min(bulk_ess(&feature).unwrap().0);
    }
    (max_rhat, min_ess)
}

#[test]
fn fixed_reference_trace_has_symmetric_posterior_ratio_and_declared_adaptation() {
    let lower = [[1.0, 0.0], [0.8, 0.6]];
    let normals = [[0.5, -1.0], [-0.25, 0.75], [1.2, 0.1], [-0.8, -0.4]];
    let uniforms = [0.2_f64, 0.9, 0.4, 0.7];
    let expected_trace = [
        [0.65, -0.3],
        [0.525, -0.175],
        [0.525, -0.175],
        [0.525, -0.175],
    ];
    let expected_ratios = [
        -0.9451955782312924,
        0.6944515306122447,
        -2.211747363945578,
        -0.4124850340136057,
    ];
    let expected_scales = [0.55, 0.495];
    let log_posterior = |eta: [f64; 2]| {
        let likelihood =
            -0.5 * ((eta[0] - 0.3) / 0.5).powi(2) - 0.5 * ((eta[1] + 0.1) / 0.7).powi(2);
        let prior = -0.5 / (1.0 - 0.8_f64.powi(2))
            * (eta[0].powi(2) - 1.6 * eta[0] * eta[1] + eta[1].powi(2));
        likelihood + prior
    };

    let mut eta = [0.4, -0.2];
    let mut scale = 0.5;
    let mut accepted = 0;
    for (step, (z, uniform)) in normals.iter().zip(uniforms).enumerate() {
        let proposal = [
            eta[0] + scale * lower[0][0] * z[0],
            eta[1] + scale * (lower[1][0] * z[0] + lower[1][1] * z[1]),
        ];
        let ratio = log_posterior(proposal) - log_posterior(eta);
        assert!((ratio - expected_ratios[step]).abs() < 1e-12);
        let accepted_step = ratio >= 0.0 || uniform.ln() < ratio;
        if accepted_step {
            eta = proposal;
            accepted += 1;
        }
        assert!((eta[0] - expected_trace[step][0]).abs() < 1e-12);
        assert!((eta[1] - expected_trace[step][1]).abs() < 1e-12);
        if (step + 1) % 2 == 0 {
            scale = if accepted as f64 / 2.0 > 0.40 {
                (scale * 1.1).min(5.0)
            } else {
                (scale * 0.9).max(1e-6)
            };
            assert!((scale - expected_scales[step / 2]).abs() < 1e-12);
            accepted = 0;
        }
    }
}

#[test]
fn correlated_gaussian_mixing_comparison_is_characterization_not_tuning() {
    let (component_chains, component_counts) = run_chains(false);
    let (compound_chains, compound_counts) = run_chains(true);
    let (component_rhat, component_ess) = diagnostics(&component_chains);
    let (compound_rhat, compound_ess) = diagnostics(&compound_chains);
    let component_acceptance =
        component_counts.component_accepts as f64 / component_counts.component_proposals as f64;
    let compound_component_acceptance =
        compound_counts.component_accepts as f64 / compound_counts.component_proposals as f64;
    let compound_block_acceptance =
        compound_counts.block_accepts as f64 / compound_counts.block_proposals as f64;

    println!(
        "N7 rho={RHO:.3} retained={RETAINED} component[accept={component_acceptance:.3},max_rhat={component_rhat:.4},min_bulk_ess={component_ess:.1}] compound[block_accept={compound_block_acceptance:.3},component_accept={compound_component_acceptance:.3},max_rhat={compound_rhat:.4},min_bulk_ess={compound_ess:.1}]"
    );

    assert!(component_rhat.is_finite() && component_ess.is_finite());
    assert!(compound_rhat.is_finite() && compound_ess.is_finite());
    assert!(compound_rhat < 1.05);
    assert!(compound_ess > component_ess);
    assert!((0.0..=1.0).contains(&component_acceptance));
    assert!((0.0..=1.0).contains(&compound_component_acceptance));
    assert!((0.0..=1.0).contains(&compound_block_acceptance));
}
