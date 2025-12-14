//! Convergence checking for NPOPT

use super::constants::*;
use super::{Phase, NPOPT};

use anyhow::Result;
use pharmsol::prelude::simulator::Equation;

impl<E: Equation + Send + 'static> NPOPT<E> {
    /// Multi-criterion convergence check
    pub(crate) fn check_convergence(&mut self) -> Result<bool> {
        // Need minimum history
        if self.objf_history.len() < CONVERGENCE_WINDOW {
            return Ok(false);
        }

        // Criterion 1: Objective function stability
        let recent: Vec<f64> = self
            .objf_history
            .iter()
            .rev()
            .take(CONVERGENCE_WINDOW)
            .cloned()
            .collect();

        let objf_stable = recent.windows(2).all(|w| (w[0] - w[1]).abs() < THETA_G);

        if !objf_stable {
            return Ok(false);
        }

        // Criterion 2: Weight stability
        if !self.weights_stable() {
            return Ok(false);
        }

        // Criterion 3: Global optimality (in Polishing phase)
        if self.phase == Phase::Polishing {
            // Run global check if we haven't recently
            if self.global_check_passes < CONVERGENCE_PASSES {
                self.sobol_global_check()?;
            }

            if self.global_check_passes >= CONVERGENCE_PASSES {
                // All criteria met
                tracing::info!(
                    "Convergence: objf stable, weights stable, {} global checks passed",
                    self.global_check_passes
                );
                return Ok(true);
            }
        }

        // In Refinement phase, require global checks + phase transition
        if self.phase == Phase::Refinement && self.global_check_passes >= CONVERGENCE_PASSES {
            // Transition to polishing
            self.phase = Phase::Polishing;
            tracing::info!("NPOPT: Refinement → Polishing (global check passed)");
        }

        Ok(false)
    }

    /// Check if weight distribution is stable
    pub(crate) fn weights_stable(&self) -> bool {
        if self.w.len() != self.w_prev.len() || self.w.len() == 0 {
            return false;
        }

        let max_change = self
            .w
            .iter()
            .zip(self.w_prev.iter())
            .map(|(w_new, w_old)| {
                if w_new > 1e-10 {
                    ((w_new - w_old) / w_new).abs()
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, |a, b| a.max(b));

        max_change < THETA_W
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_constants() {
        assert!(CONVERGENCE_WINDOW > 1);
        assert!(CONVERGENCE_PASSES > 0);
        assert!(THETA_G > 0.0);
        assert!(THETA_W > 0.0 && THETA_W < 1.0);
    }
}
