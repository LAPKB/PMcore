//! MCMC sampling infrastructure for parametric algorithms.

mod kernels;

pub(crate) use kernels::{sample_eta_from_population, ChainState, KernelConfig};