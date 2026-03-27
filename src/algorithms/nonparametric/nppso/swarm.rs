//! Particle Swarm implementation for D-criterion guided optimization

use super::constants::*;
use rand::prelude::*;

/// A particle in the swarm
#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub pbest_position: Vec<f64>,
    pub pbest_fitness: f64,
}

impl Particle {
    pub fn new<R: Rng>(ranges: &[(f64, f64)], rng: &mut R) -> Self {
        let position: Vec<f64> = ranges
            .iter()
            .map(|(lo, hi)| {
                let margin = (hi - lo) * BOUNDARY_MARGIN;
                rng.random_range((lo + margin)..(hi - margin))
            })
            .collect();

        let velocity: Vec<f64> = ranges
            .iter()
            .map(|(lo, hi)| {
                let max_v = (hi - lo) * MAX_VELOCITY_FRACTION;
                rng.random_range(-max_v..max_v) * 0.1 // Start with small velocities
            })
            .collect();

        Self {
            position: position.clone(),
            velocity,
            pbest_position: position,
            pbest_fitness: f64::NEG_INFINITY,
        }
    }

    pub fn update_velocity<R: Rng>(
        &mut self,
        gbest: &[f64],
        inertia: f64,
        cognitive: f64,
        social: f64,
        ranges: &[(f64, f64)],
        rng: &mut R,
    ) {
        for i in 0..self.position.len() {
            let r1: f64 = rng.random();
            let r2: f64 = rng.random();

            self.velocity[i] = inertia * self.velocity[i]
                + cognitive * r1 * (self.pbest_position[i] - self.position[i])
                + social * r2 * (gbest[i] - self.position[i]);

            // Clamp velocity
            let (lo, hi) = ranges[i];
            let max_v = (hi - lo) * MAX_VELOCITY_FRACTION;
            self.velocity[i] = self.velocity[i].clamp(-max_v, max_v);
        }
    }

    pub fn update_position(&mut self, ranges: &[(f64, f64)]) {
        for i in 0..self.position.len() {
            self.position[i] += self.velocity[i];

            // Reflect off boundaries
            let (lo, hi) = ranges[i];
            let margin = (hi - lo) * BOUNDARY_MARGIN;
            let lo_bound = lo + margin;
            let hi_bound = hi - margin;

            if self.position[i] < lo_bound {
                self.position[i] = lo_bound + (lo_bound - self.position[i]).min(hi_bound - lo_bound);
                self.velocity[i] *= -0.5; // Bounce with damping
            } else if self.position[i] > hi_bound {
                self.position[i] = hi_bound - (self.position[i] - hi_bound).min(hi_bound - lo_bound);
                self.velocity[i] *= -0.5;
            }
        }
    }

    /// Update personal best if current fitness is better
    pub fn update_pbest(&mut self, fitness: f64) {
        if fitness > self.pbest_fitness {
            self.pbest_fitness = fitness;
            self.pbest_position = self.position.clone();
        }
    }
}

/// The particle swarm
#[derive(Debug)]
pub struct Swarm {
    particles: Vec<Particle>,
    gbest_position: Vec<f64>,
    gbest_fitness: f64,
}

impl Swarm {
    pub fn new(n_dims: usize, ranges: &[(f64, f64)], seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Initialize particles
        let particles: Vec<Particle> = (0..SWARM_SIZE)
            .map(|_| Particle::new(ranges, &mut rng))
            .collect();

        // Initialize gbest to center
        let gbest_position: Vec<f64> = ranges.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();

        // Use n_dims to validate
        assert_eq!(n_dims, ranges.len(), "n_dims must match ranges length");

        Self {
            particles,
            gbest_position,
            gbest_fitness: f64::NEG_INFINITY,
        }
    }

    /// Get reference to particles
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Get global best fitness
    pub fn gbest_fitness(&self) -> f64 {
        self.gbest_fitness
    }

    /// Get all particle positions
    pub fn get_positions(&self) -> Vec<Vec<f64>> {
        self.particles.iter().map(|p| p.position.clone()).collect()
    }

    /// Update global best
    pub fn update_global_best(&mut self, position: &[f64], fitness: f64) {
        if fitness > self.gbest_fitness {
            self.gbest_fitness = fitness;
            self.gbest_position = position.to_vec();
        }
    }

    /// Update personal bests for all particles given their fitness values
    pub fn update_personal_bests(&mut self, fitness_values: &[f64]) {
        for (particle, &fitness) in self.particles.iter_mut().zip(fitness_values.iter()) {
            particle.update_pbest(fitness);
        }

        // Also update global best
        if let Some((best_idx, &best_fitness)) = fitness_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            if best_fitness > self.gbest_fitness {
                self.gbest_fitness = best_fitness;
                self.gbest_position = self.particles[best_idx].position.clone();
            }
        }
    }

    /// Update all particles
    pub fn update_all<R: Rng>(
        &mut self,
        inertia: f64,
        cognitive: f64,
        social: f64,
        ranges: &[(f64, f64)],
        rng: &mut R,
    ) {
        let gbest = self.gbest_position.clone();

        for particle in &mut self.particles {
            particle.update_velocity(&gbest, inertia, cognitive, social, ranges, rng);
            particle.update_position(ranges);
        }
    }

    /// Reinject random particles to maintain diversity
    pub fn reinject_random<R: Rng>(&mut self, ranges: &[(f64, f64)], rng: &mut R, n_reinject: usize) {
        // Sort by fitness, reset worst performers
        let mut indices: Vec<usize> = (0..self.particles.len()).collect();
        indices.sort_by(|&a, &b| {
            self.particles[a]
                .pbest_fitness
                .partial_cmp(&self.particles[b].pbest_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &i in indices.iter().take(n_reinject) {
            self.particles[i] = Particle::new(ranges, rng);
        }
    }

    /// Convenience wrapper that reinjects a fraction of the swarm.
    #[allow(dead_code)]
    pub fn reinject_diversity<R: Rng>(&mut self, ranges: &[(f64, f64)], rng: &mut R, fraction: f64) {
        let n_reset = (self.particles.len() as f64 * fraction) as usize;
        self.reinject_random(ranges, rng, n_reset);
    }
}
