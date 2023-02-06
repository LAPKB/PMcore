#![feature(slice_group_by)]
pub mod base;

pub mod prelude {
    pub use crate::base::lds::*;
    pub use crate::base::*;
}

//Tests
mod tests;

