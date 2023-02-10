#![feature(slice_group_by)]
pub mod base;
pub mod npag;

pub mod prelude {
    pub use crate::base::lds::*;
    pub use crate::base::*;
    pub use crate::base::datafile::Scenario;
    pub use crate::base::simulator::Simulate;
    pub use crate::base::simulator::Engine;
}

//Tests
mod tests;

