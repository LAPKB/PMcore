#![feature(slice_group_by)]
pub mod base;
pub mod algorithms;

pub mod prelude {
    pub use crate::base::lds::*;
    pub use crate::base::*;
    pub use crate::base::settings::Data;
    pub use crate::base::datafile::Scenario;
    pub use crate::base::simulator::Simulate;
    pub use crate::base::simulator::Engine;
    pub use crate::base::prob::*;
    pub use crate::algorithms::npag::npag;
}

//Tests
mod tests;

