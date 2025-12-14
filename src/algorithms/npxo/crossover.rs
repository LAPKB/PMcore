//! Crossover operators for NPXO

use super::constants::*;
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;

use anyhow::Result;
use rand::prelude::*;

/// Generate offspring via crossover of high-weight parents
pub fn generate_offspring<R: Rng>(
    theta: &Theta,
    weights: &Weights,
    ranges: &[(f64, f64)],
    count: usize,
    rng: &mut R,
) -> Result<Vec<Vec<f64>>> {
    let n_spp = theta.nspp();
    if n_spp < 2 {
        return Ok(Vec::new());
    }

    // Get parents sorted by weight
    let mut indexed: Vec<(usize, f64)> = weights.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let n_parents = (n_spp / 2).max(2).min(10);
    let parents: Vec<usize> = indexed.iter().take(n_parents).map(|(i, _)| *i).collect();

    let mut offspring = Vec::with_capacity(count);
    let matrix = theta.matrix();

    for _ in 0..count {
        // Select two parents (weighted selection favors high-weight)
        let p1_idx = parents[rng.random_range(0..parents.len())];
        let p2_idx = loop {
            let idx = parents[rng.random_range(0..parents.len())];
            if idx != p1_idx {
                break idx;
            }
        };

        let parent1: Vec<f64> = matrix.row(p1_idx).iter().copied().collect();
        let parent2: Vec<f64> = matrix.row(p2_idx).iter().copied().collect();

        // Choose crossover operator randomly
        let child = match rng.random_range(0..3) {
            0 => arithmetic_crossover(&parent1, &parent2, rng),
            1 => blx_alpha_crossover(&parent1, &parent2, ranges, rng),
            _ => sbx_crossover(&parent1, &parent2, ranges, rng),
        };

        // Optional mutation
        let child = if rng.random::<f64>() < MUTATION_PROB {
            mutate(&child, ranges, rng)
        } else {
            child
        };

        // Clamp to bounds
        let clamped: Vec<f64> = child
            .iter()
            .zip(ranges.iter())
            .map(|(&v, (lo, hi))| {
                let margin = (hi - lo) * BOUNDARY_MARGIN;
                v.clamp(lo + margin, hi - margin)
            })
            .collect();

        offspring.push(clamped);
    }

    Ok(offspring)
}

/// Arithmetic crossover: child = α·p1 + (1-α)·p2
fn arithmetic_crossover<R: Rng>(p1: &[f64], p2: &[f64], rng: &mut R) -> Vec<f64> {
    let alpha: f64 = rng.random();
    p1.iter()
        .zip(p2.iter())
        .map(|(&a, &b)| alpha * a + (1.0 - alpha) * b)
        .collect()
}

/// BLX-α crossover: sample from extended box between parents
fn blx_alpha_crossover<R: Rng>(
    p1: &[f64],
    p2: &[f64],
    ranges: &[(f64, f64)],
    rng: &mut R,
) -> Vec<f64> {
    p1.iter()
        .zip(p2.iter())
        .zip(ranges.iter())
        .map(|((&a, &b), (lo, hi))| {
            let (min_val, max_val) = if a < b { (a, b) } else { (b, a) };
            let range = max_val - min_val;
            let extension = range * BLX_ALPHA;
            
            let lower = (min_val - extension).max(*lo);
            let upper = (max_val + extension).min(*hi);
            
            rng.random_range(lower..=upper)
        })
        .collect()
}

/// Simulated Binary Crossover (SBX)
fn sbx_crossover<R: Rng>(
    p1: &[f64],
    p2: &[f64],
    ranges: &[(f64, f64)],
    rng: &mut R,
) -> Vec<f64> {
    let eta = SBX_ETA;
    
    p1.iter()
        .zip(p2.iter())
        .zip(ranges.iter())
        .map(|((&y1, &y2), (lo, hi))| {
            if (y2 - y1).abs() < 1e-14 {
                return y1;
            }

            let (y1, y2) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
            
            let u: f64 = rng.random();
            let beta = if u <= 0.5 {
                (2.0 * u).powf(1.0 / (eta + 1.0))
            } else {
                (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
            };

            let c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1));
            let c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1));

            // Return one child randomly
            let child = if rng.random() { c1 } else { c2 };
            child.clamp(*lo, *hi)
        })
        .collect()
}

/// Small random mutation
fn mutate<R: Rng>(point: &[f64], ranges: &[(f64, f64)], rng: &mut R) -> Vec<f64> {
    point
        .iter()
        .zip(ranges.iter())
        .map(|(&v, (lo, hi))| {
            let scale = (hi - lo) * MUTATION_SCALE;
            let delta: f64 = rng.random_range(-scale..scale);
            (v + delta).clamp(*lo, *hi)
        })
        .collect()
}
