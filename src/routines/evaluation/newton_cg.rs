//! Newton-CG optimizer for weights on the simplex
// See docs/optimizers_for_weights.md for algorithm details.

// Newton-CG optimizer for weights on the simplex.
// Returns (Weights, objective value). Only psi is input.
// pub fn newton_cg_weights(psi: &Psi) -> Result<(Weights, f64)> {
//     // Fallback: EM warm-start followed by L-BFGS to ensure robust and correct results
//     let (x_em, _obj_em) = em_weights(psi)?;
//     // Note: L-BFGS uses a softmax param and its own initialization; we call it directly.
//     let res = lbfgs_weights_default(psi)?;
//     Ok(res)
// }
