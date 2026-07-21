mod cycles;

mod expansion;
pub(crate) mod ipm;
mod parameter_optimizer;
mod posterior;
mod predictions;

mod psi;
pub(crate) mod qr;
mod result;
pub mod sampling;
mod statistics;
mod summaries;
mod theta;
mod weights;

pub use cycles::{CycleLog, NPCycle};
pub(crate) use expansion::adaptative_grid;
pub use ipm::burke;
pub(crate) use parameter_optimizer::ParameterOptimizer;
pub use posterior::{posterior, Posterior};
pub use predictions::{NPPredictionRow, NPPredictions};
pub(crate) use psi::calculate_psi;
pub use psi::Psi;
pub use result::NonParametricResult;
pub use statistics::{median, population_mean_median, posterior_mean_median, weighted_median};
pub use summaries::{fit_summary, individual_summaries, population_summary};
pub use theta::Theta;
pub use weights::Weights;

use std::path::Path;

/// Create the parent directory of `path` if it has a non-empty one.
///
/// For a bare relative file name such as `theta.csv`, [`Path::parent`] returns
/// `Some("")` (the empty path); calling `create_dir_all("")` would error. This
/// helper skips that case so writing to the current directory works.
pub(crate) fn create_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}
