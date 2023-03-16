#![feature(slice_group_by)]
pub mod algorithms;
pub mod base;
pub mod tui;
// extern crate openblas_src;

pub mod prelude {
    pub use crate::algorithms::npag::npag;
    pub use crate::base::array_permutation::*;
    pub use crate::base::datafile::Scenario;
    pub use crate::base::lds::*;
    pub use crate::base::prob::*;
    pub use crate::base::settings::Data;
    pub use crate::base::simulator::Engine;
    pub use crate::base::simulator::Simulate;
    pub use crate::base::*;
    pub use crate::tui::ui::*;
}

//Tests
mod tests;
