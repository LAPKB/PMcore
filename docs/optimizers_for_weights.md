# Optimizers for estimating weights x on the simplex

## Problem statement

We want to solve:

maximize f(x) = \sum*{i=1}^n log(\sum*{j=1}^m \psi[i,j] x[j])

subject to:

- x[j] >= 0 for j=1..m
- \sum\_{j=1}^m x[j] = 1

Context: non-parametric population modeling (pharmacometrics). Typically n_sub (subjects) is a few hundreds and m (support points) is <= n_sub. The matrices are mostly dense but can be sparse.

## Evaluation criteria

- Speed (time to acceptable tolerance)
- Accuracy (objective value, gradient norm)
- Robustness (convergence guarantees)
- Memory usage
- Ease of implementation/maintenance

## Recommended algorithms (concrete)

1. EM / Multiplicative update (fast baseline)

   - Update rule per iteration:
     - s = psi \* x
     - g = psi.T \* (1.0 / s)
     - x = x \* g
     - x = x / sum(x)
   - Pros: simple, monotone increase in objective, respects simplex
   - Cons: can be slow to high precision
   - Complexity: O(n\*m) per iteration
   - Tests/benchmarks: implement and measure time-to-1e-6 objective on standard sizes; compare against `burke`.

2. L-BFGS on softmax parameterization (recommended first implement)

   - Parameterize x via y: x = softmax(y)
   - Optimize F(y) = f(softmax(y)) with L-BFGS (argmin crate available)
   - Gradient: compute s = psi _ x, g_x = psi^T _ (1.0 / s), then g_y = x \* (g_x - (x.g_x))
   - Pros: few iterations to high precision; per-iteration cost O(n\*m)
   - Cons: more code; need line search and stable numerics
   - Tests: compare speed and objective vs `burke` and EM on small/medium/large sizes

3. Newton-CG (Hessian-vector product)

   - Hessian: H = -psi^T _ diag(1/s^2) _ psi
   - Hessian-vector product via two mat-vecs: psi*v, scale by 1/s^2, psi^T*(scaled)
   - Use CG to compute Newton step and line-search
   - Pros: quadratic convergence near optimum
   - Cons: more expensive per iteration; needs preconditioning

4. Accelerated Gradient / Mirror descent (entropic mirror)

   - Use mirror descent with KL geometry; performs well on simplex
   - Complexity O(n\*m) / iter; better theoretical rates than plain gradient

5. Anderson / SQUAREM acceleration for EM

   - Moderate implementation cost; sometimes dramatically faster than EM

6. Conic solvers / Clarabel / SCS (if you need general cones)
   - Use only if you need the cone structure; heavy overhead compared to tailored methods

## Implementation checklist

- Each implementation must include unit tests and benchmarks comparing against `burke` for the same inputs.
- Benchmarks: small, medium, large (e.g., 10x10, 100x100, 500x500, 1000x1000) and domain-representative sizes (nsub up to a few hundreds, m <= nsub)
- Report: timing (ms), objective value, gradient norm at termination

## Practical tips

- Use sparse mat-vec when psi is sparse.
- Use robust numerical guards: clamp s = psi\*x to a small eps > 0 before division.
- Warm-start: run a few EM iterations to get close, then switch to L-BFGS/Newton for refinement.
- Use an appropriate stopping criterion: relative objective tolerance and/or gradient norm.

## Next steps (prioritized)

1. Implement L-BFGS (best speed/implementation tradeoff). Use argmin; add tests/benchmarks.
2. Implement Newton-CG for high-precision needs.
3. Implement EM (if not present) and SQUAREM/Anderson acceleration as a fallback.

## Contributors and authors

This document was generated to guide optimizer development for `PMcore`.
