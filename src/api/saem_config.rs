use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct SaemConfig {
    pub k1_iterations: usize,
    pub k2_iterations: usize,
    pub burn_in: usize,
    pub sa_iterations: usize,
    pub sa_cooling_factor: f64,
    pub mcmc_step_size: f64,
    pub rw_init: f64,
    pub n_chains: usize,
    pub mcmc_iterations: usize,
    pub omega_min_variance: f64,
    pub use_gibbs: bool,
    pub n_kernels: usize,
    pub transform_par: Vec<u8>,
    pub compute_map: bool,
    pub compute_fim: bool,
    pub compute_ll_is: bool,
    pub compute_ll_gq: bool,
    pub n_mc_is: usize,
    pub nu_is: usize,
    pub n_nodes_gq: usize,
    pub n_sd_gq: f64,
    pub display_progress: usize,
    pub seed: u64,
    pub fix_seed: bool,
}

impl Default for SaemConfig {
    fn default() -> Self {
        Self {
            k1_iterations: 300,
            k2_iterations: 100,
            burn_in: 5,
            sa_iterations: 0,
            sa_cooling_factor: 0.97,
            mcmc_step_size: 0.4,
            rw_init: 0.5,
            n_chains: 1,
            mcmc_iterations: 1,
            omega_min_variance: 1e-6,
            use_gibbs: false,
            n_kernels: 4,
            transform_par: vec![],
            compute_map: true,
            compute_fim: true,
            compute_ll_is: true,
            compute_ll_gq: false,
            n_mc_is: 5000,
            nu_is: 4,
            n_nodes_gq: 12,
            n_sd_gq: 4.0,
            display_progress: 10,
            seed: 123456,
            fix_seed: true,
        }
    }
}

impl SaemConfig {
    pub fn total_iterations(&self) -> usize {
        self.k1_iterations + self.k2_iterations
    }

    pub fn is_exploration_phase(&self, iteration: usize) -> bool {
        iteration <= self.k1_iterations
    }

    pub fn is_smoothing_phase(&self, iteration: usize) -> bool {
        iteration > self.k1_iterations
    }

    pub fn is_sa_active(&self, iteration: usize) -> bool {
        self.sa_iterations > 0 && iteration <= self.sa_iterations
    }

    pub fn sa_temperature(&self, iteration: usize) -> f64 {
        if self.is_sa_active(iteration) {
            self.sa_cooling_factor.powi(iteration as i32)
        } else {
            1.0
        }
    }

    pub fn step_size(&self, iteration: usize) -> f64 {
        if iteration <= self.k1_iterations {
            1.0
        } else {
            let k_smooth = iteration - self.k1_iterations;
            1.0 / (k_smooth as f64 + 1.0)
        }
    }

    pub fn get_transform(&self, param_idx: usize) -> u8 {
        self.transform_par.get(param_idx).copied().unwrap_or(1)
    }

    pub fn get_transforms(&self, n_params: usize) -> Vec<u8> {
        let mut transforms = self.transform_par.clone();
        while transforms.len() < n_params {
            transforms.push(1);
        }
        transforms.truncate(n_params);
        transforms
    }

    pub fn infer_transforms_from_ranges(&mut self, ranges: &[(f64, f64)]) {
        self.transform_par = ranges
            .iter()
            .map(|(lower, upper)| {
                if *lower >= 0.0 && *upper > 0.0 && lower.is_finite() && upper.is_finite() {
                    1
                } else if (*lower - 0.0).abs() < 1e-10 && (*upper - 1.0).abs() < 1e-10 {
                    3
                } else {
                    0
                }
            })
            .collect();
    }
}