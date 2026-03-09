# Comprehensive Analysis of Non-Parametric Population Pharmacokinetic Algorithms

## Executive Summary

This document provides a detailed analysis of non-parametric algorithms implemented in PMcore for population pharmacokinetic modeling. The analysis focuses on the theoretical foundations, implementation details, and comparative characteristics of each algorithm.

---

## 1. Foundational Theory: Non-Parametric Maximum Likelihood (NPML)

### 1.1 The Mixing Distribution Problem

The core problem in population pharmacokinetics is estimating the distribution of parameters across a population. Given:

- **Observations**: $Y_1, \ldots, Y_N$ - independent random vectors from N subjects
- **Parameters**: $\theta_1, \ldots, \theta_N$ - unknown parameter values belonging to compact set $\Theta$
- **Distribution**: $F$ - unknown probability distribution on $\Theta$

The likelihood function is:
$$L(F) = \prod_{i=1}^{N} \int p(Y_i|\theta_i) dF(\theta_i)$$

The goal is to maximize $L(F)$ over all probability distributions on $\Theta$.

### 1.2 Lindsay-Mallet Theorem (Key Result)

**Theorem**: The global maximizer $F_{ML}$ of $L(F)$ is a **discrete distribution** with at most N support points (N = number of subjects).

This transforms the infinite-dimensional optimization into a finite-dimensional problem:
$$\max_{\theta_k, \lambda_k} \sum_{i=1}^{N} \log\left(\sum_{k=1}^{K} \lambda_k p(Y_i|\theta_k)\right)$$

subject to $\lambda_k \geq 0$, $\sum_k \lambda_k = 1$, and $K \leq N$.

### 1.3 Two-Problem Structure

**Problem 1 (Convex)**: Given support points $\{\theta_k\}$, find optimal weights $\{\lambda_k\}$

- Solved by Burke's Primal-Dual Interior Point Method (PDIP)

**Problem 2 (Non-convex, Global)**: Find optimal support point locations

- This is where algorithms differ fundamentally

### 1.4 Burke's Interior Point Method (IPM)

The weight optimization problem is solved using Burke's IPM, which maximizes:

$$f(\mathbf{x}) = \sum_{i=1}^{N} \log\left(\sum_{j=1}^{K} \Psi_{ij} x_j\right)$$

subject to $x_j \geq 0$ and $\sum_j x_j = 1$, where $\Psi_{ij} = p(Y_i|\theta_j)$.

**Algorithm** (Burke's IPM):

```
Input: Ψ matrix (N subjects × K support points)
Initialize: λ = [1, ..., 1], w = 1/P(Y|λ)
While gap > ε and norm_r > ε:
    1. Compute inner = λ / y
    2. Compute H = Ψ · diag(inner) · Ψᵀ + diag(P(Y|λ)/w)
    3. Cholesky: H = UᵀU
    4. Solve for Δw using forward/backward substitution
    5. Compute Δy = -Ψᵀ · Δw
    6. Compute Δλ = σμ/y - λ - inner ⊙ Δy
    7. Line search for step lengths αpri, αdual
    8. Update: λ += αpri·Δλ, w += αdual·Δw, y += αdual·Δy
    9. Adapt σ based on feasibility vs duality gap
Output: Normalized λ (weights summing to 1), objective value
```

**Convergence Criteria**:

- Duality gap < ε (default: 1e-8)
- Residual norm < ε
- Typically converges in 10-50 iterations

### 1.5 Rank-Revealing QR Decomposition

After weight optimization, redundant support points are removed using QR decomposition:

```
Input: Ψ matrix (N×K)
Output: Indices of linearly independent columns

1. Compute QR with column pivoting: ΨP = QR
   where P is permutation matrix, R is upper triangular

2. For i = 1 to min(N, K):
   ratio = |R_ii| / ||R[:,i]||₂
   if ratio ≥ 1e-8:
       keep.append(perm[i])

3. Return keep (indices of independent support points)
```

This removes support points that are linear combinations of others (in terms of their likelihood contributions), preventing numerical issues in subsequent IPM iterations.

---

## 2. The D-Optimality Criterion (D-Function)

### 2.1 Definition

The directional derivative of the log-likelihood in direction of Dirac distribution at $\xi$:
$$D(\xi, F) = \sum_{i=1}^{N} \frac{p(Y_i|\xi)}{p(Y_i|F)} - N$$

where $p(Y_i|F) = \sum_k \lambda_k p(Y_i|\theta_k)$

### 2.2 Optimality Conditions

**Lindsay's Theorem**: $F^* = F_{ML}$ if and only if $\max_{\xi \in \Theta} D(\xi, F^*) = 0$

**Corollary**: If $\max_{\xi} D(\xi, F^*) \neq 0$, then:
$$L(F_{ML}) - L(F^*) \leq \max_{\xi} D(\xi, F^*)$$

This provides both:

1. A stopping criterion for convergence
2. A bound on optimality gap

### 2.3 Physical Interpretation

D is large when:

- $p(Y_i|\xi)$ is high: parameter $\xi$ explains subject i well
- $p(Y_i|F)$ is low: current mixture explains subject i poorly

**Insight**: Maximizing D finds parameters for **poorly-fit subjects** - targeting modes the mixture is missing.

### 2.4 Computational Implementation

```
Function D(ξ, F):
    Input: candidate point ξ, current mixture F = (θ, w)

    // Compute P(Y_i | ξ) for all subjects
    psi_xi = [P(Y_i | ξ) for i in 1..N]

    // Compute P(Y_i | F) = Σ_k w_k × P(Y_i | θ_k)
    // This is pre-computed as P(Y|G) = Ψ · w
    pyl = [P(Y_i | F) for i in 1..N]

    // D-criterion
    D = -N
    For i in 1..N:
        D += psi_xi[i] / pyl[i]

    Return D
```

**Interpretation of D values**:

- $D > 0$: Adding point ξ would improve the mixture (should add)
- $D = 0$: Point ξ is already optimally covered (at convergence)
- $D < 0$: Point ξ would worsen the mixture (don't add)

---

## 3. Algorithm Implementations in PMcore

### 3.1 NPAG (Non-Parametric Adaptive Grid)

**Principle**: "Throw and catch" - systematic grid exploration

**Key Constants** (from source code):

```rust
const THETA_E: f64 = 1e-4;  // Grid spacing convergence threshold
const THETA_G: f64 = 1e-4;  // Objective function convergence threshold
const THETA_F: f64 = 1e-2;  // P(Y|L) convergence criterion
const THETA_D: f64 = 1e-4;  // Minimum distance between support points
```

**Detailed Algorithm**:

```
Input: data Y, error models, parameter ranges, initial eps = 0.2
Initialize: θ = Sobol_quasi_random(n_initial, ranges)
            objf = -∞, last_objf = -∞, f0 = -∞

While not converged:
    cycle++

    // ======== 1. ESTIMATION ========
    // Compute likelihood matrix Ψ_ij = P(Y_i | θ_j)
    For each subject i, support point j (in parallel):
        Ψ_ij = likelihood(Y_i, model(θ_j), error_model)

    [λ, _] = Burke_IPM(Ψ)  // Initial weights

    // ======== 2. CONDENSATION ========
    // Step 2a: Lambda filter (remove negligible weights)
    max_λ = max(λ)
    keep = {j : λ_j > max_λ / 1000}
    θ = θ[keep], Ψ = Ψ[:, keep]

    // Step 2b: QR rank-revealing factorization
    [R, perm] = QR_with_pivoting(Ψ)
    keep = {i : |R_ii / ||R_i||₂| ≥ 1e-8}
    θ = θ[perm[keep]], Ψ = Ψ[:, perm[keep]]

    // Step 2c: Final weight computation
    [w, objf] = Burke_IPM(Ψ)

    // ======== 3. ERROR MODEL OPTIMIZATION ========
    For each output equation with optimizable error:
        γ_up = γ × (1 + δ)
        γ_down = γ / (1 + δ)

        Ψ_up = recalculate_psi(θ, γ_up)
        Ψ_down = recalculate_psi(θ, γ_down)

        [_, objf_up] = Burke_IPM(Ψ_up)
        [_, objf_down] = Burke_IPM(Ψ_down)

        if objf_up > objf:
            Accept γ_up, δ *= 4
        if objf_down > objf:
            Accept γ_down, δ *= 4

        δ *= 0.5
        if δ < 0.01: δ = 0.1

    // ======== 4. ADAPTIVE GRID EXPANSION ========
    // For each support point, add daughter points at ±eps×range
    candidates = []
    For each θ_k in θ:
        For each dimension d:
            step = eps × (range_d_max - range_d_min)

            θ_plus = θ_k.copy()
            θ_plus[d] += step
            if θ_plus[d] < range_d_max:
                candidates.append(θ_plus)

            θ_minus = θ_k.copy()
            θ_minus[d] -= step
            if θ_minus[d] > range_d_min:
                candidates.append(θ_minus)

    // Add candidates that are far enough from existing points
    For candidate in candidates:
        if min_distance(candidate, θ) > THETA_D:
            θ = θ ∪ {candidate}

    // ======== 5. CONVERGENCE CHECK ========
    // Primary: objective function stability with eps halving
    if |last_objf - objf| ≤ THETA_G and eps > THETA_E:
        eps = eps / 2

        if eps ≤ THETA_E:
            // Secondary: P(Y|L) criterion
            P(Y|L) = Ψ · w
            f1 = Σᵢ log(P(Yᵢ|L))

            if |f1 - f0| ≤ THETA_F:
                STOP (converged)
            else:
                f0 = f1
                eps = 0.2  // Reset grid spacing

    if cycle ≥ max_cycles:
        STOP (max cycles)

    last_objf = objf

Output: θ (support points), w (weights), -2×objf (-2LL)
```

**Adaptive Grid Expansion Details**:
The grid expands by adding 2×d new candidate points for each existing support point (one in each direction along each dimension). The step size is `eps × range_width`, where eps starts at 0.2 and halves when the objective function stabilizes.

Example with 2D parameter space (Ke, V):

```
Original point: (Ke=0.5, V=10)
Ranges: Ke ∈ [0.1, 1.0], V ∈ [1, 20]
eps = 0.2

Step sizes: Ke: 0.2×0.9=0.18, V: 0.2×19=3.8

New candidates:
  - (0.68, 10)   // Ke + step
  - (0.32, 10)   // Ke - step
  - (0.5, 13.8)  // V + step
  - (0.5, 6.2)   // V - step
```

**Convergence Behavior**:

1. **Outer loop**: eps halves from 0.2 → 0.1 → 0.05 → ... → 0.0001
2. **Inner criterion**: At each eps level, iterate until objective stabilizes
3. **Final criterion**: P(Y|L) must also stabilize

**Strengths**:

- Robust exploration of entire parameter space
- Guaranteed to find all modes (given enough iterations)
- Well-understood convergence behavior
- No tuning parameters beyond grid spacing

**Weaknesses**:

- Computationally expensive: O(K×2d) new points per cycle
- Many evaluations in empty regions (no signal)
- Slow convergence in high dimensions (curse of dimensionality)
- Cannot adapt to problem structure

### 3.2 NPOD (Non-Parametric Optimal Design)

**Principle**: D-function guided directional search

**Key Difference from NPAG**: Instead of grid expansion, uses Nelder-Mead optimization of D-function to suggest new support points.

**Detailed Algorithm**:

```
Input: Initial θ (support points), data Y, error models
Initialize: eps = 0.2, objf = -∞

While not converged:
    1. ESTIMATION
       Compute Ψ_ij = P(Y_i | θ_j) for all subjects i, points j
       [λ, _] = Burke_IPM(Ψ)

    2. CONDENSATION
       keep = {j : λ_j > max(λ)/1000}
       θ = θ[keep], Ψ = Ψ[:,keep]

       [R, perm] = QR_RankRevealing(Ψ)
       keep = {i : |R_ii / ||R_i||₂| ≥ 1e-8}
       θ = θ[perm[keep]], Ψ = Ψ[:,perm[keep]]

       [w, objf] = Burke_IPM(Ψ)

    3. ERROR MODEL OPTIMIZATION
       For each output equation:
           γ_up = γ × (1 + δ), γ_down = γ / (1 + δ)
           Evaluate objf at γ_up, γ_down
           Accept if improvement, adapt δ

    4. D-OPTIMAL EXPANSION (Key difference from NPAG)
       P(Y|G) = Ψ · w  // Subject-wise mixture probability

       For each support point θ_k (in parallel):
           θ_k^new = argmax_ξ D(ξ, F)
                   = argmax_ξ [Σᵢ P(Yᵢ|ξ)/P(Yᵢ|G) - N]
           using Nelder-Mead starting from θ_k

       For each candidate θ^new:
           if dist(θ^new, θ) > THETA_D:
               θ = θ ∪ {θ^new}

    5. CONVERGENCE CHECK
       if |objf^(n) - objf^(n-1)| < THETA_F:
           STOP (converged)

Output: θ (support points), w (weights), -2×objf
```

**Key Constants**:

- `THETA_F = 1e-2`: Objective function convergence threshold
- `THETA_D = 1e-4`: Minimum distance between support points

**Computational Details**:

- Nelder-Mead optimizes negative D (to maximize D)
- Parallel optimization of all support points
- Simplex initialized with 5% perturbation of each dimension

**Advantages**:

- Faster convergence (10-20x fewer cycles than NPAG)
- Information-directed search
- Efficient use of D-criterion gradient

**Limitations**:

- Local search (Nelder-Mead) - can miss global modes
- No exploration mechanism beyond current support
- May converge to local optima in multimodal spaces

### 3.3 NPSAH (Simulated Annealing Hybrid)

**Principle**: Combine NPAG exploration with NPOD refinement and SA for mode discovery

**Three Components**:

1. **NPAG-style grid expansion** (warm-up phase)
2. **NPOD D-optimal refinement** (high-importance points)
3. **Simulated Annealing injection** (escape local optima)

**Key Constants** (from source code):

```rust
const THETA_E: f64 = 1e-4;           // Grid spacing convergence
const THETA_G: f64 = 1e-4;           // Objective function convergence
const THETA_F: f64 = 1e-2;           // P(Y|L) convergence
const THETA_D: f64 = 1e-4;           // Min distance between points

const WARMUP_CYCLES: usize = 5;      // NPAG-style warmup
const INITIAL_TEMPERATURE: f64 = 1.0;
const COOLING_RATE: f64 = 0.95;
const SA_INJECT_COUNT: usize = 10;   // SA points per cycle
const HIGH_IMPORTANCE_THRESHOLD: f64 = 0.1;  // Weight threshold
const HIGH_IMPORTANCE_MAX_ITERS: u64 = 100;  // Nelder-Mead iters
const LOW_IMPORTANCE_MAX_ITERS: u64 = 10;
const CONVERGENCE_WINDOW: usize = 3;
const GLOBAL_OPTIMALITY_SAMPLES: usize = 500;
const GLOBAL_OPTIMALITY_THRESHOLD: f64 = 0.01;
const MIN_TEMPERATURE: f64 = 0.01;
```

**Detailed Algorithm**:

```
Input: Initial θ, data Y, error models
Initialize: T = 1.0, eps = 0.2, in_warmup = true

While not converged:
    1. ESTIMATION & CONDENSATION (same as NPAG/NPOD)

    2. EXPANSION (phase-dependent)
       if cycle ≤ WARMUP_CYCLES:
           // Phase 1: NPAG-style grid expansion
           adaptive_grid(θ, eps, ranges, THETA_D)
       else:
           // Phase 2: Hybrid expansion

           // 2a. D-optimal refinement with adaptive iterations
           P(Y|G) = Ψ · w
           For each support point θ_k (in parallel):
               importance = w_k / max(w)
               if importance > HIGH_IMPORTANCE_THRESHOLD:
                   max_iters = 100
               else:
                   max_iters = 10
               θ_k^new = Nelder_Mead(D, θ_k, max_iters)
               if dist(θ_k^new, θ) > THETA_D:
                   θ = θ ∪ {θ_k^new}

           // 2b. Sparse grid expansion
           adaptive_grid(θ, eps/2, ranges, THETA_D×2)

           // 2c. Simulated Annealing injection
           n_inject = ceil(SA_INJECT_COUNT × T)
           accepted = 0
           For _ in 0..(n_inject × 10):
               ξ = random_point_in_ranges()
               D_val = D(ξ, F)

               // Metropolis acceptance
               if D_val > 0:
                   accept = true
               else:
                   p_accept = exp(D_val / T)
                   accept = (random() < p_accept)

               if accept and dist(ξ, θ) > THETA_D:
                   θ = θ ∪ {ξ}
                   accepted++

               if accepted ≥ n_inject: break

           // Cool temperature
           T = max(T × COOLING_RATE, MIN_TEMPERATURE)

    3. MULTI-CRITERION CONVERGENCE CHECK
       // Criterion 1: Objective stability
       if objf_history stable over CONVERGENCE_WINDOW cycles:
           // Criterion 2: Global optimality (Monte Carlo)
           max_D = 0
           For _ in 0..GLOBAL_OPTIMALITY_SAMPLES:
               ξ = random_point()
               max_D = max(max_D, D(ξ, F))

           if max_D < GLOBAL_OPTIMALITY_THRESHOLD:
               STOP (converged)

Output: θ, w, -2×objf
```

**Why SA Helps**:

- NPOD's Nelder-Mead gets trapped in local basins
- SA explores parameter space stochastically
- Metropolis criterion allows "uphill" moves (accepting negative D)
- Temperature schedule balances exploration (high T) vs exploitation (low T)

### 3.4 NPSAH2 (Simulated Annealing Hybrid v2)

**Principle**: Improved NPSAH with adaptive temperature, elite preservation, and four-phase architecture

**Key Improvements over NPSAH v1**:

1. **Adaptive Temperature Schedule**: Temperature adapts based on acceptance ratio (not fixed cooling)
2. **Elite Preservation**: Best points preserved across cycles (prevents regression)
3. **Four-Phase Architecture**: Warmup → Hybrid → Exploitation → Convergence
4. **Latin Hypercube Sampling**: Better initial coverage than random sampling
5. **Restart Mechanism**: Can restart from cold when stuck
6. **Hierarchical D-optimal Refinement**: Iteration count based on point importance

**Key Constants** (from source code):

```rust
// Phase Control
const WARMUP_CYCLES: usize = 3;
const EXPLOITATION_CYCLES: usize = 3;

// Temperature Schedule (Adaptive)
const INITIAL_TEMPERATURE: f64 = 1.5;
const BASE_COOLING_RATE: f64 = 0.88;
const MIN_TEMPERATURE: f64 = 0.01;
const TARGET_ACCEPTANCE_RATIO: f64 = 0.25;
const REHEAT_FACTOR: f64 = 1.3;

// Exploration Parameters
const SA_INJECT_BASE: usize = 10;
const ELITE_COUNT: usize = 3;
const LHS_SAMPLES: usize = 30;

// D-Optimal Refinement (Hierarchical)
const HIGH_IMPORTANCE_THRESHOLD: f64 = 0.05;
const HIGH_IMPORTANCE_MAX_ITERS: u64 = 80;
const MEDIUM_IMPORTANCE_MAX_ITERS: u64 = 30;
const LOW_IMPORTANCE_MAX_ITERS: u64 = 10;

// Safety
const BOUNDARY_MARGIN_RATIO: f64 = 0.01;

// Restart
const STAGNATION_CYCLES: usize = 15;
const MAX_RESTARTS: usize = 2;
```

**Four-Phase Architecture**:

```
Phase 1: WARMUP (cycles 1-3)
    - Latin Hypercube Sampling for space-filling coverage
    - NPAG-style adaptive grid expansion
    - No SA injection yet

Phase 2: HYBRID (cycles 4-6)
    - D-optimal refinement (high-weight points only)
    - Local SA moves around high-weight points
    - Sparse grid expansion
    - Global SA injection (temperature-scaled count)
    - Elite point re-injection

Phase 3: EXPLOITATION (cycles 7+ while T > MIN_TEMPERATURE×2)
    - D-optimal refinement (only high-weight points)
    - Light grid expansion (eps×0.5, THETA_D×2)
    - No SA injection (temperature too low)

Phase 4: CONVERGENCE (when T approaches minimum)
    - Minimal expansion (eps×0.25)
    - Focus on convergence verification
```

**Adaptive Temperature Control**:

```
adapt_temperature():
    if sa_proposed > 0:
        acceptance_ratio = sa_accepted / sa_proposed

        if acceptance_ratio < TARGET_ACCEPTANCE_RATIO × 0.5:
            // Too cold - slow down cooling
            cooling_rate = min(cooling_rate + 0.02, 0.98)

            // Maybe reheat if very cold and low acceptance
            if acceptance_ratio < 0.1 and T < 0.5:
                T *= REHEAT_FACTOR

        elif acceptance_ratio > TARGET_ACCEPTANCE_RATIO × 1.5:
            // Too hot - speed up cooling
            cooling_rate = max(cooling_rate - 0.02, 0.85)

    // Apply cooling
    T = max(T × cooling_rate, MIN_TEMPERATURE)
```

**Why NPSAH2 Outperforms NPSAH**:

1. **Adaptive cooling prevents premature freezing**: Fixed cooling can be too aggressive
2. **Elite preservation prevents regression**: Good points are never lost
3. **LHS provides better initial coverage**: More uniform than random sampling
4. **Phase structure adapts strategy**: Explore early, exploit late
5. **Restart escapes deep local optima**: Can escape when truly stuck

**Benchmark Performance**:

- NPSAH: -422.46 (15 cycles, 43.08s)
- NPSAH2: -439.68 (35 cycles, 121.26s) — **Best overall -2LL**

The ~17 point improvement in -2LL demonstrates that adaptive temperature control and elite preservation are crucial for finding the global optimum in multimodal problems.

### 3.5 NPCAT (Covariance-Adaptive Trajectory)

**Principle**: Fisher Information-guided exploration with Sobol quasi-random global checks

**Key Innovations**:

1. **Fisher Information-guided sampling**: Generates candidates along directions of high parameter uncertainty
2. **Sobol quasi-random sequences**: Provably better coverage than Monte Carlo for global optimality checks
3. **Three-phase convergence state machine**: Exploring → Refining → Polishing
4. **L-BFGS-B local refinement**: Gradient-based optimization for high-weight points

**Key Constants** (from source code):

```rust
// Convergence thresholds
const THETA_W: f64 = 1e-3;           // Weight stability threshold
const THETA_G: f64 = 1e-4;           // Objective function threshold
const THETA_D_GLOBAL: f64 = 0.01;    // Global optimality D-criterion threshold
const THETA_F: f64 = 1e-2;           // P(Y|L) convergence criterion
const MIN_DISTANCE: f64 = 1e-4;      // Minimum support point distance

// Expansion parameters
const INITIAL_K: usize = 40;         // Initial candidates per cycle
const K_DECAY_RATE: f64 = 0.95;      // Decay rate (exponential)
const MIN_K: usize = 4;              // Minimum candidates

// Refinement parameters
const BASE_OPTIM_ITERS: u64 = 20;    // Base L-BFGS-B iterations
const OPTIM_ITER_GROWTH: u64 = 10;   // Additional iterations per log(cycle)
const OPTIM_TOLERANCE: f64 = 1e-4;   // Optimization tolerance

// Global check parameters
const SOBOL_SAMPLES: usize = 256;    // Samples for global optimality check
const GLOBAL_CHECK_INTERVAL: usize = 5; // Cycles between global checks

// Candidate generation ratios
const FISHER_RATIO: f64 = 0.60;      // 60% from Fisher Information
const DOPT_RATIO: f64 = 0.30;        // 30% from D-optimal perturbations
const BOUNDARY_RATIO: f64 = 0.10;    // 10% from boundary exploration
```

**Three-Phase Convergence State Machine**:

```
Phase 1: EXPLORING (first cycles)
    - High expansion rate (INITIAL_K candidates)
    - Fisher Information-guided candidate generation
    - Transitions when: objective stabilizes AND coverage sufficient

Phase 2: REFINING (middle cycles)
    - Balanced expansion/refinement
    - Periodic Sobol global optimality checks (every 5 cycles)
    - L-BFGS-B refinement of high-weight points
    - Transitions when: global check passes AND objective stable

Phase 3: POLISHING (final cycles)
    - No expansion (expansion disabled)
    - Full refinement of all surviving points
    - Converges when: P(Y|L) criterion met
```

**Fisher Information-Guided Exploration**:

```
For each high-weight support point θ:
    1. Compute Fisher Information Matrix F(θ) = -E[∂²logL/∂θ²]
    2. Decompose: F = V Λ V^T (eigendecomposition)
    3. Identify directions of high uncertainty: eigenvectors with small eigenvalues
    4. Generate candidates: θ_new = θ ± step × v_i (for uncertain directions)
```

**Why NPCAT Works Well**:

1. **Intelligent exploration**: Fisher Information targets regions where we're most uncertain
2. **Quasi-random global checks**: Sobol sequences guarantee better coverage than random
3. **Phase adaptation**: Different strategies for different convergence stages
4. **L-BFGS-B refinement**: Efficient gradient-based local optimization
5. **Balanced candidate generation**: 60% information, 30% D-optimal, 10% boundary

**Benchmark Performance**:

- NPCAT: -437.80 (29 cycles, 35.12s) — **Excellent quality/speed balance**

NPCAT achieves near-best -2LL in ~1/3 the time of NPSAH2, making it the best speed-quality tradeoff.

### 3.6 NPPSO (Particle Swarm Optimization)

**Principle**: Swarm intelligence for D-criterion optimization

**Key Innovation**: Particles search for regions maximizing D-optimality + Subject targeting for poorly-fit subjects

**Key Constants** (from source code):

```rust
// PSO Parameters
const SWARM_SIZE: usize = 40;
const INERTIA_MAX: f64 = 0.9;
const INERTIA_MIN: f64 = 0.4;
const COGNITIVE_WEIGHT: f64 = 2.0;   // c₁: personal best attraction
const SOCIAL_WEIGHT: f64 = 2.0;      // c₂: global best attraction
const MAX_VELOCITY_FRACTION: f64 = 0.15;
const BOUNDARY_MARGIN: f64 = 0.001;

// Phases
const WARMUP_CYCLES: usize = 3;
const D_THRESHOLD_FRACTION: f64 = 0.5;
const CONVERGENCE_THRESHOLD: f64 = 0.8;
const REINJECT_FRACTION: f64 = 0.25;

// Simulated Annealing (key for escaping local optima)
const SA_INITIAL_TEMP: f64 = 3.0;
const SA_COOLING_RATE: f64 = 0.95;
const SA_MIN_TEMP: f64 = 0.05;
const SA_INJECT_COUNT: usize = 15;

// Subject MAP & D-Optimal
const RESIDUAL_SUBJECTS: usize = 2;
const SUBJECT_MAP_EVALS: usize = 100;
const DOPT_REFINE_EVALS: usize = 50;
const DOPT_REFINE_INTERVAL: usize = 10;

// Elite Preservation
const ELITE_COUNT: usize = 10;
const ELITE_MAX_AGE: usize = 15;
```

**Detailed Algorithm**:

```
Input: Initial θ, data Y, error models
Initialize: swarm[40 particles], T_sa = 3.0

For each particle p in swarm:
    p.position = random_in_ranges()
    p.velocity = random × MAX_VELOCITY_FRACTION × range
    p.pbest_position = p.position
    p.pbest_fitness = -∞

While not converged:
    1. ESTIMATION & CONDENSATION (standard NP)
       Update P(Y|G) = Ψ · w

    2. UPDATE SWARM FITNESS
       For each particle p (in parallel):
           p.fitness = D(p.position, F)
           if p.fitness > p.pbest_fitness:
               p.pbest_position = p.position
               p.pbest_fitness = p.fitness

       gbest = particle with max fitness
       global_best_position = gbest.position
       global_best_fitness = gbest.fitness

    3. PSO VELOCITY/POSITION UPDATE
       inertia = adaptive_inertia()  // Based on improvement rate

       For each particle p:
           r₁, r₂ = random(0,1)

           // Velocity update equation
           v_new = inertia × p.velocity
                 + c₁ × r₁ × (p.pbest_position - p.position)
                 + c₂ × r₂ × (global_best_position - p.position)

           // Velocity clamping
           v_new = clamp(v_new, -v_max, v_max)

           // Position update
           p.position = p.position + v_new
           p.position = clamp(p.position, ranges)
           p.velocity = v_new

    4. EXPANSION (after warm-up)
       if cycle > WARMUP_CYCLES:
           // 4a. Add high-fitness particles as candidates
           max_D = max(all particle fitness)
           threshold = max_D × D_THRESHOLD_FRACTION
           For each particle with fitness > max(threshold, 0):
               if dist(particle.position, θ) > THETA_D:
                   θ = θ ∪ {particle.position}

           // 4b. SA injection (KEY for escaping local optima)
           For _ in 0..SA_INJECT_COUNT×3:
               ξ = random_point_in_ranges()
               D_val = D(ξ, F)

               accept = (D_val > 0) OR (random() < exp(D_val/T_sa))
               if accept and dist(ξ, θ) > THETA_D:
                   θ = θ ∪ {ξ}

           T_sa = max(T_sa × SA_COOLING_RATE, SA_MIN_TEMP)

           // 4c. Subject MAP injection for poorly-fit subjects
           worst_subjects = bottom RESIDUAL_SUBJECTS by P(Y|G)
           For subject s in worst_subjects:
               θ_map = COBYLA(maximize P(Y_s|θ), start=centroid)
               if D(θ_map, F) > 0 and dist(θ_map, θ) > THETA_D:
                   θ = θ ∪ {θ_map}

           // 4d. D-optimal refinement (every DOPT_REFINE_INTERVAL cycles)
           if cycle % DOPT_REFINE_INTERVAL == 0:
               For high-weight support points:
                   θ_refined = COBYLA(maximize D, start=θ_k)
                   if improvement:
                       θ = θ ∪ {θ_refined}

           // 4e. Elite preservation
           age_elite_points()
           add_top_weighted_points_to_elite()
           reinject_elite_points_to_θ()

           // 4f. Diversity maintenance
           if swarm_convergence_ratio() > CONVERGENCE_THRESHOLD:
               reinject_random_particles(25%)
       else:
           // Warm-up: NPAG-style grid expansion
           adaptive_grid(θ, eps, ranges, THETA_D)

    5. GLOBAL OPTIMALITY CHECK
       max_D = max over 500 random points of D(ξ, F)
       if max_D < GLOBAL_D_THRESHOLD:
           STOP (converged)

Output: θ, w, -2×objf
```

**Why PSO + SA Works**:

1. **Momentum**: Particles overshoot, exploring beyond local basins
2. **Collective Learning**: Swarm shares information about good regions
3. **SA Injection**: Provides exploration that pure PSO might miss
4. **Subject Targeting**: MAP for poorly-fit subjects directly targets missing modes
5. **Elite Preservation**: Prevents loss of good solutions during exploration

**Adaptive Inertia**:

```
if improvement > 1.0: return INERTIA_MAX (0.9)  // Explore
if improvement > 0.1: return (MAX+MIN)/2 (0.65) // Balance
else: return INERTIA_MIN (0.4)                  // Exploit
```

### 3.7 NPCMA (CMA-ES Approach)

**Principle**: Covariance Matrix Adaptation Evolution Strategy

**Key Innovation**: Adapts a multivariate normal distribution to sample promising solutions, learning covariance structure

**Key Constants** (from source code):

```rust
const WARMUP_CYCLES: usize = 3;
const THETA_E: f64 = 1e-4;
const THETA_G: f64 = 1e-4;
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;

// CMA-ES specific
const CMA_LAMBDA: usize = 20;        // Population size
const CMA_MU: usize = 10;            // Parent size (top half)
const CMA_SIGMA_INIT: f64 = 0.3;     // Initial step size
```

**CMA-ES State**:

```
State:
    mean: Vec<f64>          // Distribution mean (center)
    sigma: f64              // Step size (overall scale)
    C: Mat<f64>             // Covariance matrix
    p_c: Vec<f64>           // Evolution path for C
    p_sigma: Vec<f64>       // Evolution path for σ
```

**Detailed Algorithm**:

```
Input: Initial θ, data Y, error models
Initialize: CMA state (mean = center of ranges, σ = 0.3, C = I)

While not converged:
    1. ESTIMATION & CONDENSATION (standard NP)
       Update P(Y|G) = Ψ · w

    2. CMA-ES EXPANSION (after warm-up)
       if cycle > WARMUP_CYCLES:
           // Step 1: Sample λ candidates from N(mean, σ²C)
           candidates = []
           For k = 1 to CMA_LAMBDA:
               z_k ~ N(0, I)
               x_k = mean + σ × B × D × z_k  // BD = sqrt(C)
               x_k = clamp(x_k, ranges)
               candidates.append(x_k)

           // Step 2: Evaluate D-criterion (in parallel)
           fitness = [D(x_k, F) for x_k in candidates]

           // Step 3: Selection (best μ individuals)
           sorted_idx = argsort(fitness, descending=true)
           selected = [candidates[i] for i in sorted_idx[:CMA_MU]]

           // Step 4: Update mean (weighted recombination)
           weights = [w_i for i in 1..CMA_MU]  // Sum to 1
           mean_new = Σᵢ wᵢ × selected[i]

           // Step 5: Update evolution paths
           p_c = (1-c_c) × p_c + sqrt(c_c×(2-c_c)×μ_eff) × (mean_new-mean)/σ

           // Step 6: Update covariance matrix
           // Rank-μ update + Rank-one update
           y = [(selected[i] - mean) / σ for i in 1..μ]
           C = (1-c_1-c_μ) × C
             + c_1 × p_c × p_cᵀ                    // Rank-one
             + c_μ × Σᵢ wᵢ × yᵢ × yᵢᵀ             // Rank-μ

           // Step 7: Update step size (CSA)
           norm_expected = E[||N(0,I)||]
           p_sigma = (1-c_σ) × p_sigma + sqrt(c_σ×(2-c_σ)×μ_eff) × B⁻¹ × (mean_new-mean)/σ
           σ = σ × exp((c_σ/d_σ) × (||p_sigma||/norm_expected - 1))

           mean = mean_new

           // Step 8: Add high-D samples to support points
           For x_k with fitness > 0:
               if dist(x_k, θ) > THETA_D:
                   θ = θ ∪ {x_k}

           // Step 9: Restart if converged prematurely
           if σ < σ_stop or all eigenvalues of C < threshold:
               Reinitialize CMA state
       else:
           // Warm-up: NPAG-style grid
           adaptive_grid(θ, eps, ranges, THETA_D)

    3. CONVERGENCE CHECK (standard NPAG-style)

Output: θ, w, -2×objf
```

**Why CMA-ES Works for NPML**:

1. **Covariance Learning**: Automatically discovers parameter correlations
2. **Step Size Adaptation**: Prevents premature convergence
3. **Invariant to Linear Transformations**: Robust to parameter scaling
4. **D-Criterion Fitness**: Directs search toward information-maximizing regions

**Limitations**:

- Population-based: requires many evaluations per generation
- May struggle with highly multimodal problems
- No explicit global search beyond distribution tails

### 3.8 NPXO (Crossover Optimization)

**Principle**: Genetic crossover operators between good support points

**Crossover Operators**:

1. **Arithmetic**: $\text{child} = \alpha \cdot \text{parent}_1 + (1-\alpha) \cdot \text{parent}_2$
2. **BLX-α**: child sampled from extended box around parents
3. **SBX**: Simulated Binary Crossover with polynomial distribution

**Key Constants** (typical values):

```rust
const CROSSOVER_PROBABILITY: f64 = 0.9;
const ARITHMETIC_ALPHA: f64 = 0.5;
const BLX_ALPHA: f64 = 0.5;
const SBX_ETA: f64 = 2.0;
```

**Detailed Algorithm**:

```
While not converged:
    1. ESTIMATION & CONDENSATION (standard)

    2. CROSSOVER EXPANSION
       // Select parents based on weight (roulette wheel)
       parents = weighted_sample(θ, w, n_pairs)

       For each (parent1, parent2) pair:
           // Choose crossover operator randomly
           op = random_choice([Arithmetic, BLX, SBX])

           if op == Arithmetic:
               α = random(0.3, 0.7)
               child = α × parent1 + (1-α) × parent2

           elif op == BLX-α:
               // Sample from extended bounding box
               For each dimension d:
                   lo = min(parent1[d], parent2[d])
                   hi = max(parent1[d], parent2[d])
                   I = hi - lo
                   child[d] = random(lo - α×I, hi + α×I)
                   child[d] = clamp(child[d], ranges[d])

           elif op == SBX:
               // Simulated Binary Crossover
               For each dimension d:
                   u = random(0, 1)
                   if u < 0.5:
                       β = (2×u)^(1/(η+1))
                   else:
                       β = (1/(2×(1-u)))^(1/(η+1))
                   child[d] = 0.5 × ((1+β)×parent1[d] + (1-β)×parent2[d])

           // Evaluate and add if good
           D_val = D(child, F)
           if D_val > 0 and dist(child, θ) > THETA_D:
               θ = θ ∪ {child}

    3. CONVERGENCE (standard)
```

**Why Crossover Works**:

- Exploits correlations between good points (interpolation/extrapolation)
- Preserves "genetic material" from successful regions
- Fast convergence when modes are already partially discovered
- Low computational cost per offspring

**Limitations**:

- Limited exploration (depends on existing diversity)
- Cannot discover new modes far from current support
- Performance degrades on highly multimodal problems

### 3.9 NPBO (Bayesian Optimization)

**Principle**: Gaussian Process surrogate with Expected Improvement acquisition

**Key Idea**: Build a surrogate model (GP) of the D-criterion landscape, then use acquisition function to balance exploration and exploitation.

**Key Constants**:

```rust
const WARMUP_CYCLES: usize = 5;
const SOBOL_SAMPLES: usize = 50;     // Initial space-filling samples
const BO_SAMPLES_PER_CYCLE: usize = 20;
const GP_NOISE: f64 = 1e-4;          // Observation noise
const EI_SAMPLES: usize = 1000;      // Candidates for EI optimization
```

**Detailed Algorithm**:

```
Input: Initial θ, data Y, error models
Initialize: D_observations = [], GP = None

While not converged:
    1. ESTIMATION & CONDENSATION (standard)
       Update P(Y|G) = Ψ · w

    2. COLLECT D-CRITERION OBSERVATIONS
       // Evaluate D at current support points
       For each θ_k:
           D_k = D(θ_k, F)
           D_observations.append((θ_k, D_k))

    3. GP-BASED EXPANSION (after warm-up)
       if cycle > WARMUP_CYCLES:
           // Step 1: Train GP on D-criterion observations
           X = [obs[0] for obs in D_observations]  // Locations
           y = [obs[1] for obs in D_observations]  // D values
           GP.fit(X, y)

           // Step 2: Generate candidate points
           candidates = []
           For _ in 0..EI_SAMPLES:
               candidates.append(random_in_ranges())
           // Also add points near current support
           For θ_k in θ:
               candidates.append(perturb(θ_k, small_noise))

           // Step 3: Compute Expected Improvement for each candidate
           μ, σ = GP.predict(candidates)  // Mean and std
           f_best = max(y)                // Best observed D

           EI = []
           For μ_i, σ_i in zip(μ, σ):
               if σ_i > 0:
                   z = (μ_i - f_best) / σ_i
                   ei = σ_i × (z × Φ(z) + φ(z))  // Φ=CDF, φ=PDF
               else:
                   ei = max(0, μ_i - f_best)
               EI.append(ei)

           // Step 4: Select top EI candidates
           top_k = argsort(EI, descending=true)[:BO_SAMPLES_PER_CYCLE]

           // Step 5: Evaluate and add promising points
           For idx in top_k:
               candidate = candidates[idx]
               D_actual = D(candidate, F)  // True evaluation
               D_observations.append((candidate, D_actual))

               if D_actual > 0 and dist(candidate, θ) > THETA_D:
                   θ = θ ∪ {candidate}
       else:
           // Warm-up: Sobol sampling for space-filling initial design
           sobol_points = sobol_sequence(SOBOL_SAMPLES, n_dims)
           For point in sobol_points:
               D_val = D(point, F)
               D_observations.append((point, D_val))
               if D_val > 0:
                   θ = θ ∪ {point}

    4. CONVERGENCE (standard)

Output: θ, w, -2×objf
```

**Expected Improvement (EI)**:
$$\text{EI}(\mathbf{x}) = \sigma(\mathbf{x}) \left[ z \Phi(z) + \phi(z) \right]$$
where $z = \frac{\mu(\mathbf{x}) - f_{\text{best}}}{\sigma(\mathbf{x})}$

EI balances:

- **Exploitation**: High μ(x) → likely good point
- **Exploration**: High σ(x) → uncertain region worth exploring

**Advantages**:

- Principled exploration/exploitation trade-off
- Efficient use of expensive D-criterion evaluations
- Works well in low-to-moderate dimensions

**Limitations**:

- GP training cost scales cubically with observations: O(n³)
- Degrades in high dimensions (> 10-15 parameters)
- Requires hyperparameter tuning (kernel, noise)

### 3.10 NEXUS (Unified Subject-driven Search)

**Principle**: Cross-Entropy Method with GMM + Subject-guided exploration

**Key Innovations**:

1. **Cross-Entropy Method**: GMM learns distribution of good solutions
2. **Subject-guided exploration**: Target poorly-fit subjects
3. **Adaptive SA**: Temperature feedback
4. **D-optimal refinement**: Hierarchical iteration allocation
5. **Multi-scale global verification**

**Key Constants** (from source code):

```rust
// Convergence
const THETA_G: f64 = 1e-4;
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;
const THETA_W: f64 = 1e-3;           // Weight stability

// Cross-Entropy Method
const CE_SAMPLE_SIZE: usize = 50;
const CE_ELITE_FRACTION: f64 = 0.10;  // Top 10%
const CE_GMM_COMPONENTS: usize = 3;
const CE_MIN_VARIANCE: f64 = 1e-6;
const CE_SMOOTHING: f64 = 0.3;

// Subject-guided
const RESIDUAL_SUBJECT_FRACTION: f64 = 0.3;
const MIN_RESIDUAL_SUBJECTS: usize = 3;
const SUBJECT_MAP_MAX_ITERS: u64 = 30;

// D-optimal refinement (hierarchical)
const DOPT_HIGH_WEIGHT_ITERS: u64 = 100;
const DOPT_MED_WEIGHT_ITERS: u64 = 40;
const DOPT_LOW_WEIGHT_ITERS: u64 = 15;
const HIGH_WEIGHT_THRESHOLD: f64 = 0.10;
const MED_WEIGHT_THRESHOLD: f64 = 0.01;

// Adaptive SA
const INITIAL_TEMPERATURE: f64 = 5.0;
const TARGET_ACCEPTANCE_RATIO: f64 = 0.25;
const REHEAT_FACTOR: f64 = 1.2;

// Multi-scale global check
const GLOBAL_CHECK_SCALES: [usize; 3] = [64, 256, 1024];
const GLOBAL_D_THRESHOLD: f64 = 0.005;
```

**Gaussian Mixture Model (GMM)**:

```
GMM with K=3 components:
    components: [(mean₁, Σ₁, π₁), (mean₂, Σ₂, π₂), (mean₃, Σ₃, π₃)]

Sample from GMM:
    1. Select component k with probability πₖ
    2. Sample x ~ N(meanₖ, Σₖ)
    3. Clamp to parameter bounds

Update GMM from elite points:
    1. E-step: Compute responsibilities r_ik = P(component k | point i)
    2. M-step: Update parameters with smoothing:
       mean_k = (1-α)×mean_k + α × Σᵢ r_ik×D_i×point_i / Σᵢ r_ik×D_i
       Σ_k = (1-α)×Σ_k + α × weighted_covariance(elite, responsibilities)
       π_k = (1-α)×π_k + α × Σᵢ r_ik×D_i / Σᵢ,ₖ r_ik×D_i
```

**Detailed Algorithm**:

```
Input: Initial θ, data Y, error models
Initialize: GMM = None, T = 5.0, phase = Warmup

While not converged:
    1. ESTIMATION & CONDENSATION (standard)

    2. PHASE TRANSITION
       if cycle > WARMUP_CYCLES and phase == Warmup:
           phase = Expansion
           GMM = GMM.from_theta(θ, w)  // Initialize from support points

    3. EXPANSION
       if phase == Warmup:
           // Stratified Sobol + adaptive grid
           adaptive_grid(θ, eps, ranges, THETA_D)

       else:  // Expansion or Convergence phase
           // === Cross-Entropy Sampling ===
           ce_samples = GMM.sample(CE_SAMPLE_SIZE)
           D_values = [D(s, F) for s in ce_samples]  // Parallel

           // Select elite (top 10%)
           elite_idx = argsort(D_values)[-int(CE_ELITE_FRACTION×len)]:]
           elite = [(ce_samples[i], D_values[i]) for i in elite_idx]

           // Update GMM toward elite distribution
           GMM.update_from_elite(elite)

           // Add elite points with positive D to theta
           For (point, D_val) in elite:
               if D_val > 0 and dist(point, θ) > THETA_D:
                   θ = θ ∪ {point}

           // === Subject-Guided Exploration ===
           P(Y|G) = Ψ · w
           worst_subjects = bottom 30% by P(Y|G)

           For subject s in worst_subjects[:MIN_RESIDUAL_SUBJECTS]:
               // Find MAP estimate for this subject
               start = weighted_centroid(θ, w)
               θ_map = Nelder_Mead(maximize P(Y_s|θ), start, max_iter=30)

               D_val = D(θ_map, F)
               if D_val > 0 and dist(θ_map, θ) > THETA_D:
                   θ = θ ∪ {θ_map}

           // === Adaptive Simulated Annealing ===
           accepted, proposed = 0, 0
           For _ in 0..SA_INJECT_COUNT:
               ξ = random_in_ranges()
               D_val = D(ξ, F)
               proposed += 1

               accept = (D_val > 0) OR (random() < exp(D_val/T))
               if accept:
                   accepted += 1
                   if dist(ξ, θ) > THETA_D:
                       θ = θ ∪ {ξ}

           // Adapt temperature based on acceptance ratio
           acceptance_ratio = accepted / proposed
           if acceptance_ratio < TARGET_ACCEPTANCE_RATIO:
               T *= REHEAT_FACTOR  // Too cold, reheat
           else:
               T *= COOLING_RATE   // Normal cooling

           // === D-Optimal Refinement (Hierarchical) ===
           max_w = max(w)
           For each θ_k in θ:
               importance = w_k / max_w
               if importance > HIGH_WEIGHT_THRESHOLD:
                   max_iters = 100
               elif importance > MED_WEIGHT_THRESHOLD:
                   max_iters = 40
               else:
                   max_iters = 15

               θ_k^refined = Nelder_Mead(maximize D, start=θ_k, max_iter)
               if improvement and dist(θ_k^refined, θ) > THETA_D:
                   θ = θ ∪ {θ_k^refined}

           // === Elite Preservation ===
           age_elite_points()
           add_top_weighted_to_elite()
           reinject_elite_to_θ()

    4. MULTI-SCALE GLOBAL CONVERGENCE CHECK
       For scale in [64, 256, 1024]:
           max_D = 0
           For _ in 0..scale:
               ξ = sobol_sample(sobol_index++)
               max_D = max(max_D, D(ξ, F))

           if max_D > GLOBAL_D_THRESHOLD:
               break  // Failed at this scale

       if all scales passed:
           if weights_stable and objf_stable:
               phase = Convergence → then STOP

Output: θ, w, -2×objf
```

**Why Cross-Entropy + Subject-Guided Works**:

1. **CE learns problem structure**: Unlike SA which samples blindly, CE maintains a model of where good solutions are
2. **GMM captures multimodality**: Multiple components can represent distinct modes
3. **Subject targeting is principled**: The D-function insight shows poorly-fit subjects indicate missing modes
4. **Hierarchical refinement is efficient**: Spend more effort on important points
5. **Multi-scale verification provides convergence certificate**

### 3.11 NPOPT (Optimal Trajectory)

**Principle**: Three-phase architecture combining best elements from all algorithms

**Design Principles**:

1. D-optimal refinement + Global optimality checks
2. Adaptive SA with reheat (prevents premature cooling)
3. Fisher-guided exploration (high-uncertainty directions)
4. Subject residual injection
5. Elite preservation

**Key Constants** (from source code):

```rust
// Convergence
const THETA_G: f64 = 1e-4;
const THETA_F: f64 = 1e-2;
const THETA_D: f64 = 1e-4;
const THETA_W: f64 = 1e-3;
const GLOBAL_D_THRESHOLD: f64 = 0.008;

// Grid
const INITIAL_EPS: f64 = 0.2;
const MIN_EPS: f64 = 1e-4;

// Phases
const EXPLORATION_CYCLES: usize = 3;
const SOBOL_INIT_SAMPLES: usize = 50;
const GLOBAL_CHECK_INTERVAL: usize = 3;
const SOBOL_GLOBAL_SAMPLES: usize = 256;
const CONVERGENCE_PASSES: usize = 2;

// Adaptive SA
const INITIAL_TEMPERATURE: f64 = 2.0;
const BASE_COOLING_RATE: f64 = 0.90;
const MIN_TEMPERATURE: f64 = 0.01;
const TARGET_ACCEPTANCE: f64 = 0.23;
const REHEAT_TRIGGER: f64 = 0.08;
const REHEAT_FACTOR: f64 = 1.5;
const SA_INJECT_COUNT: usize = 30;
const SA_HISTORY_WINDOW: usize = 5;

// Fisher-guided
const FISHER_RATIO: f64 = 0.70;      // 70% from Fisher directions
const DOPT_RATIO: f64 = 0.30;        // 30% from D-gradient
const FISHER_CANDIDATES: usize = 20;

// D-optimal refinement (hierarchical)
const HIGH_WEIGHT_THRESHOLD: f64 = 0.10;
const MED_WEIGHT_THRESHOLD: f64 = 0.01;
const LOW_WEIGHT_THRESHOLD: f64 = 0.001;
const DOPT_HIGH_ITERS: u64 = 80;
const DOPT_MED_ITERS: u64 = 30;
const DOPT_LOW_ITERS: u64 = 10;

// Subject residual
const RESIDUAL_SUBJECTS: usize = 3;
const SUBJECT_MAP_ITERS: u64 = 30;

// Elite
const ELITE_COUNT: usize = 5;
const ELITE_MAX_AGE: usize = 20;
```

**Three Phases**:

**Phase 1: Exploration (cycles 1-3)**

```
// Stratified Sobol initialization for space-filling coverage
sobol_points = sobol_sequence(SOBOL_INIT_SAMPLES, n_dims)
θ = θ ∪ sobol_points

// Sparse adaptive grid
adaptive_grid(θ, eps, ranges, THETA_D)

// Initialize Fisher Information estimates
fisher_diagonal = estimate_fisher()
```

**Phase 2: Refinement (cycles 4+)**

```
// === D-Optimal Refinement (Parallel, Hierarchical) ===
max_w = max(w)
For each θ_k (in parallel):
    importance = w_k / max_w
    if importance > HIGH_WEIGHT_THRESHOLD: iters = 80
    elif importance > MED_WEIGHT_THRESHOLD: iters = 30
    elif importance > LOW_WEIGHT_THRESHOLD: iters = 10
    else: skip

    θ_k^refined = Nelder_Mead(D, θ_k, iters)
    if D(θ_k^refined) > D(θ_k):
        θ = θ ∪ {θ_k^refined}

// === Adaptive SA with Reheat ===
For _ in 0..SA_INJECT_COUNT:
    ξ = random_in_ranges()
    D_val = D(ξ, F)

    accept = (D_val > 0) OR (random() < exp(D_val/T))
    if accept and dist(ξ, θ) > THETA_D:
        θ = θ ∪ {ξ}
        sa_accepted++

// Adapt temperature
acceptance_ratio = sa_accepted / SA_INJECT_COUNT
sa_history.append(acceptance_ratio)
rolling_avg = mean(sa_history[-SA_HISTORY_WINDOW:])

if rolling_avg < REHEAT_TRIGGER:
    T *= REHEAT_FACTOR  // Reheat when too cold
else:
    T *= BASE_COOLING_RATE

// === Fisher-Guided Exploration ===
// High Fisher Information = high uncertainty = explore there
centroid = weighted_centroid(θ, w)
For _ in 0..FISHER_CANDIDATES:
    // Sample direction biased toward high-Fisher dimensions
    direction = sample_fisher_biased(fisher_diagonal)
    step_size = random(0.1, 0.5) × range
    candidate = centroid + step_size × direction

    D_val = D(candidate, F)
    if D_val > 0 and dist(candidate, θ) > THETA_D:
        θ = θ ∪ {candidate}

// === Subject Residual Injection ===
worst_subjects = bottom RESIDUAL_SUBJECTS by P(Y|G)
For subject s in worst_subjects:
    θ_map = Nelder_Mead(maximize P(Y_s|θ), centroid, SUBJECT_MAP_ITERS)
    D_val = D(θ_map, F)
    if D_val > 0:
        θ = θ ∪ {θ_map}

// === Elite Preservation ===
update_elite_points()
reinject_elite()

// === Periodic Global Check ===
if cycle % GLOBAL_CHECK_INTERVAL == 0:
    max_D = 0
    For _ in 0..SOBOL_GLOBAL_SAMPLES:
        ξ = sobol_sample()
        max_D = max(max_D, D(ξ, F))

    if max_D < GLOBAL_D_THRESHOLD:
        global_check_passes++
        if global_check_passes >= CONVERGENCE_PASSES:
            phase = Polishing
```

**Phase 3: Polishing (when global checks pass)**

```
// Full D-optimal refinement of ALL points (high iterations)
For each θ_k:
    θ_k^refined = Nelder_Mead(D, θ_k, DOPT_HIGH_ITERS)

// No expansion - only refinement

// Convergence when:
// 1. Weights stable (||w - w_prev|| < THETA_W)
// 2. P(Y|L) criterion met (|f1 - f0| < THETA_F)
// 3. Objf stable
```

**Why NPOPT's Three-Phase Architecture Works**:

1. **Exploration**: Ensures broad coverage before intensive refinement
2. **Refinement**: Balances global (SA, Fisher) and local (D-opt) search
3. **Polishing**: Final cleanup with convergence guarantees
4. **Adaptive SA with reheat**: Prevents premature "freezing"
5. **Fisher-guided**: Principled exploration in uncertain directions
6. **Sobol global checks**: Rigorous verification of optimality

---

## 4. Comparative Analysis

### 4.1 Exploration vs Exploitation Trade-off

| Algorithm | Exploration | Exploitation | Primary Mechanism                         |
| --------- | ----------- | ------------ | ----------------------------------------- |
| NPAG      | High        | Low          | Systematic grid coverage                  |
| NPOD      | Low         | High         | D-gradient descent (Nelder-Mead)          |
| NPSAH     | Balanced    | Balanced     | SA injection + Grid + D-opt               |
| NPSAH2    | Adaptive    | Adaptive     | 4-phase SA + Elite + LHS + Restart        |
| NPPSO     | High        | Moderate     | Swarm momentum + Subject MAP              |
| NPCMA     | Adaptive    | Adaptive     | Covariance adaptation                     |
| NPXO      | Moderate    | High         | Genetic crossover (interpolation)         |
| NPBO      | Balanced    | Balanced     | GP uncertainty (EI acquisition)           |
| NEXUS     | High        | High         | CE distribution learning + Subject-guided |
| NPOPT     | Phased      | Phased       | 3-phase: explore→refine→polish            |

### 4.2 Key Algorithmic Components

| Algorithm | Global Search       | Local Refinement | Subject Targeting | Elite Preservation |
| --------- | ------------------- | ---------------- | ----------------- | ------------------ |
| NPAG      | Grid expansion      | None             | No                | No                 |
| NPOD      | None                | Nelder-Mead on D | Implicit (via D)  | No                 |
| NPSAH     | SA injection        | Adaptive NM      | Implicit          | No                 |
| NPSAH2    | SA+LHS+Restart      | Hierarchical NM  | Implicit          | Yes (3 elite)      |
| NPPSO     | SA + Swarm          | COBYLA on D      | Yes (MAP)         | Yes                |
| NPCMA     | Covariance sampling | Evolution paths  | No                | No                 |
| NPXO      | Crossover diversity | None             | No                | No                 |
| NPBO      | GP uncertainty      | None             | No                | No                 |
| NEXUS     | CE + SA             | Hierarchical NM  | Yes (MAP)         | Yes                |
| NPOPT     | SA + Fisher         | Hierarchical NM  | Yes (MAP)         | Yes                |

### 4.3 Computational Complexity per Cycle

| Algorithm | Ψ Computation | Weight Optimization | Expansion           | Total               |
| --------- | ------------- | ------------------- | ------------------- | ------------------- |
| NPAG      | O(N·K)        | O(K³)               | O(K·d) grid         | O(N·K + K³)         |
| NPOD      | O(N·K)        | O(K³)               | O(K·d·I) NM         | O(N·K·I)            |
| NPSAH     | O(N·K)        | O(K³)               | O(K·d·I + SA·N)     | O(N·K·I + SA·N)     |
| NPSAH2    | O(N·K)        | O(K³)               | O(K·d·I + LHS + SA) | O(N·K·I + LHS + SA) |
| NPPSO     | O(N·K)        | O(K³)               | O(S·N + K·I)        | O(S·N + K·I)        |
| NPCMA     | O(N·K)        | O(K³)               | O(λ·N + d³)         | O(λ·N + d³)         |
| NPXO      | O(N·K)        | O(K³)               | O(pairs·N)          | O(pairs·N)          |
| NPBO      | O(N·K)        | O(K³)               | O(n³ + m·N)         | O(n³ + m·N)         |
| NEXUS     | O(N·K)        | O(K³)               | O(CE·N + MAP·I)     | O(CE·N + MAP·I)     |
| NPOPT     | O(N·K)        | O(K³)               | O(SA·N + K·I)       | O(SA·N + K·I)       |

Where: N=subjects, K=support points, d=dimensions, I=NM iterations, S=swarm size, λ=CMA population, n=GP observations, m=EI samples, CE=CE samples

### 4.4 Convergence Properties

| Algorithm | Local Convergence     | Global Guarantee             | Convergence Verification |
| --------- | --------------------- | ---------------------------- | ------------------------ |
| NPAG      | Yes (grid refinement) | Probabilistic (grid density) | ε convergence            |
| NPOD      | Yes (D-gradient)      | No                           | Δobjf threshold          |
| NPSAH     | Yes (D-gradient)      | Probabilistic (SA temp)      | Monte Carlo D-check      |
| NPSAH2    | Yes (D-gradient)      | Probabilistic (SA+Restart)   | Adaptive D-check         |
| NPPSO     | Yes (D-gradient)      | Probabilistic (swarm)        | Sobol D-check            |
| NPCMA     | Yes (adaptation)      | Probabilistic (restart)      | σ convergence            |
| NPXO      | Yes (crossover)       | No                           | Δobjf threshold          |
| NPBO      | Yes (GP mean)         | Probabilistic (EI)           | GP uncertainty           |
| NEXUS     | Yes (D-gradient)      | Yes (multi-scale Sobol)      | 3-scale verification     |
| NPOPT     | Yes (D-gradient)      | Yes (repeated Sobol)         | 2-pass verification      |

### 4.5 Memory and State Requirements

| Algorithm | Additional State                   | Memory Overhead         |
| --------- | ---------------------------------- | ----------------------- | ---- |
| NPAG      | ε, grid history                    | Minimal                 |
| NPOD      | P(Y                                | G) cache                | O(N) |
| NPSAH     | T, objf history, elite             | O(cycles + elite)       |
| NPSAH2    | T, cooling_rate, elite, stagnation | O(cycles + elite + LHS) |
| NPPSO     | Swarm (S particles), elite         | O(S·d + elite)          |
| NPCMA     | C matrix, evolution paths          | O(d² + d)               |
| NPXO      | Parent selection buffer            | O(K)                    |
| NPBO      | GP (X, y, kernel)                  | O(n² + n·d)             |
| NEXUS     | GMM (K components), elite          | O(K·d² + elite)         |
| NPOPT     | Fisher diagonal, elite, SA history | O(d + elite + window)   |

---

## 5. Algorithm Selection Guidelines

### 5.1 Decision Tree for Algorithm Selection

```
Start
  │
  ├─ Is speed critical? ──Yes──► NPOD (fast, but may miss modes)
  │
  ├─ Is the problem likely unimodal? ──Yes──► NPOD or NPAG
  │
  ├─ Are there expected correlations between parameters?
  │     │
  │     └─ Yes ──► NPCMA (learns correlations automatically)
  │
  ├─ Is the problem highly multimodal (multiple populations)?
  │     │
  │     └─ Yes ──► NPSAH2, NPPSO, or NEXUS (global exploration)
  │
  ├─ Do you need best-quality solution (time not critical)?
  │     │
  │     └─ Yes ──► NPSAH2 (adaptive temperature + elite preservation)
  │
  ├─ Do you need convergence guarantees for publication?
  │     │
  │     └─ Yes ──► NEXUS or NPOPT (multi-scale verification)
  │
  ├─ Is the dimensionality high (>8 parameters)?
  │     │
  │     └─ Yes ──► NPPSO or NEXUS (scale better than NPBO)
  │
  └─ Default ──► NPSAH2 (best quality) or NPSAH (faster, still good)
```

### 5.2 Recommended Use Cases

**For Publication/Clinical Use**:

- **NPAG**: Gold standard, well-documented, conservative (always safe)
- **NPOD**: When speed critical and simple models expected
- **NPSAH2**: Best solution quality when time permits
- **NEXUS/NPOPT**: Complex models requiring convergence guarantees

**For Research/Development**:

- **NPPSO**: Exploratory analysis, unknown parameter spaces
- **NPCMA**: When parameter correlations are important
- **NPSAH**: Balanced approach, good speed-quality tradeoff
- **NPSAH2**: When best quality is needed regardless of time

**For High-Dimensional Problems (>8 params)**:

- **NPPSO**: Subject-guided exploration scales with subjects
- **NEXUS**: CE-based, doesn't suffer from curse of dimensionality as much
- **Avoid**: NPBO (GP scales poorly), NPCMA (covariance matrix grows)

### 5.3 Expected Performance Characteristics

Based on benchmark results (bimodal Ke problem):

| Algorithm | Typical -2LL | Typical Cycles | Typical Time | Best For          |
| --------- | ------------ | -------------- | ------------ | ----------------- |
| NPSAH2    | -440         | 30-50          | 100-150s     | Best quality      |
| NPCAT     | -438         | 25-35          | 30-40s       | Quality + speed   |
| NPPSO     | -437         | 80-120         | 25-35s       | Multimodal        |
| NPSAH     | -422         | 10-20          | 40-50s       | Balanced          |
| NPOPT     | -376         | 10-15          | 35-45s       | Phased approach   |
| NPOD      | -375         | 10-15          | 2-5s         | Speed             |
| NPAG      | -348         | 200-400        | 8-15s        | Baseline          |
| NPCMA     | -347         | 100-150        | 4-8s         | Correlated params |
| NPBO      | -346         | 100-150        | 6-10s        | Low-dim only      |

---

## 6. Paper Focus: NPAG → NPOD → Advanced Optimizers

### 6.1 Narrative Arc

The progression we propose to highlight:

1. **NPAG (Baseline)**:
   - Established, robust, but slow
   - "Throw and catch" - systematic but wasteful
   - Many unnecessary evaluations in empty regions

2. **NPOD (First Improvement)**:
   - D-function guided, faster convergence
   - "Informed search" - follows gradient of optimality
   - But: local search can miss modes

3. **Advanced Hybrids (NPSAH, NPPSO, NEXUS)**:
   - Global exploration + local refinement
   - "Intelligent exploration" - learns where to look
   - Multiple mechanisms to escape local optima:
     - SA injection (stochastic escape)
     - Subject targeting (mode discovery)
     - Elite preservation (prevent regression)

### 6.2 Key Innovation Claims

1. **D-criterion is not just for stopping**: Using D as objective for global search (not just convergence check)

2. **Subject-guided exploration**: Poorly-fit subjects indicate missing modes - target them directly

3. **Adaptive temperature control**: Feedback-based SA prevents premature cooling

4. **Hierarchical refinement**: Allocate computational resources proportional to importance

5. **Multi-scale global verification**: Rigorous convergence certificates

### 6.3 Experimental Questions

1. Does global exploration (SA, swarm, CE) significantly improve -2LL?
2. Is subject targeting necessary, or is generic SA sufficient?
3. How do algorithms compare on truly multimodal problems?
4. What is the cost/benefit of convergence verification?

---

## 7. Summary of Algorithm Mechanisms

### 7.1 Quick Reference Table

| Algorithm | Expansion                  | Global Exploration  | Local Refinement       | Stopping                |
| --------- | -------------------------- | ------------------- | ---------------------- | ----------------------- |
| **NPAG**  | Adaptive grid (±eps×range) | Grid coverage       | None                   | eps → 0, P(Y\|L) stable |
| **NPOD**  | D-gradient NM              | None                | Nelder-Mead            | Δobjf < θ_F             |
| **NPSAH** | Grid + D-opt + SA          | SA with Metropolis  | NM with adaptive iters | Monte Carlo D-check     |
| **NPPSO** | Swarm + SA + MAP           | PSO velocity + SA   | COBYLA on D            | Sobol D-check           |
| **NPCMA** | Covariance sampling        | Distribution tails  | Evolution paths        | σ convergence           |
| **NPXO**  | Crossover operators        | Crossover diversity | None                   | Δobjf < θ_F             |
| **NPBO**  | EI acquisition             | GP uncertainty      | None                   | GP variance             |
| **NEXUS** | CE + Subject-guided + SA   | GMM learning + SA   | Hierarchical NM        | Multi-scale Sobol       |
| **NPOPT** | Fisher-guided + SA + MAP   | SA with reheat      | Hierarchical NM        | Repeated Sobol          |

### 7.2 Key Takeaways

1. **NPAG remains the baseline**: Well-understood, robust, but slow. Use when convergence guarantees matter more than speed.

2. **NPOD is the fast option**: 10-20x faster than NPAG, but may miss modes in multimodal problems.

3. **Global exploration is essential for multimodal problems**: Algorithms with SA (NPSAH, NPPSO, NEXUS, NPOPT) consistently outperform those without (NPAG, NPOD, NPCMA, NPXO, NPBO) on the bimodal benchmark.

4. **Subject targeting adds value**: NPPSO and NEXUS's subject-guided injection helps discover modes that random exploration might miss.

5. **Temperature management matters**: Adaptive SA with reheat (NPOPT) or feedback (NEXUS) prevents premature cooling.

6. **Convergence verification provides confidence**: Multi-scale Sobol checks (NEXUS, NPOPT) give rigorous optimality certificates.

### 7.3 Recommended Reading Order

For understanding the algorithmic progression:

1. Read NPAG first (Section 3.1) - the foundation
2. Read NPOD (Section 3.2) - the D-function innovation
3. Read NPSAH (Section 3.3) - the first hybrid
4. Read NPPSO (Section 3.4) - swarm intelligence approach
5. Read NEXUS (Section 3.8) - the most complete hybrid

---

_Document generated from PMcore source code analysis. All algorithm constants and pseudocode are extracted directly from the Rust implementations._
