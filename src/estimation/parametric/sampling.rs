//! MCMC sampling infrastructure for parametric algorithms.

mod kernels;

pub(crate) use kernels::{
    advance_saem_chains, sample_eta_from_population, ChainState, KernelConfig, SaemMcmcState,
};
