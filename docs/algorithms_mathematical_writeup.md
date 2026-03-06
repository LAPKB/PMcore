# Mathematical Formulations of Novel Non-Parametric Population PK/PD Algorithms

**Authors:** Laboratory of Applied Pharmacokinetics and Bioinformatics (LAPKB)  
**Date:** January 2026  
**Document Version:** 1.0

---

## Abstract

This document presents rigorous mathematical formulations for four novel non-parametric (NP) algorithms for population pharmacokinetic/pharmacodynamic (PK/PD) modeling: NPSAH (Non-Parametric Simulated Annealing Hybrid), NPCAT (Non-Parametric Covariance-Adaptive Trajectory), NPOPT (Non-Parametric OPTimal Trajectory), and NPPSO (Non-Parametric Particle Swarm Optimization). These algorithms extend the foundational work of NPAG (Non-Parametric Adaptive Grid) and NPOD (Non-Parametric Optimal Design) by incorporating advanced optimization techniques to improve convergence speed, solution quality, and robustness against local optima. We provide detailed mathematical derivations, convergence criteria, and theoretical justifications for each algorithmic component.

---

## Table of Contents

1. [Introduction and Background](#1-introduction-and-background)
2. [Mathematical Preliminaries](#2-mathematical-preliminaries)
3. [NPSAH: Non-Parametric Simulated Annealing Hybrid](#3-npsah-non-parametric-simulated-annealing-hybrid)
4. [NPCAT: Non-Parametric Covariance-Adaptive Trajectory](#4-npcat-non-parametric-covariance-adaptive-trajectory)
5. [NPOPT: Non-Parametric OPTimal Trajectory](#5-npopt-non-parametric-optimal-trajectory)
6. [NPPSO: Non-Parametric Particle Swarm Optimization](#6-nppso-non-parametric-particle-swarm-optimization)
7. [Comparative Analysis](#7-comparative-analysis)

---

## 1. Introduction and Background

### 1.1 The Non-Parametric Maximum Likelihood Problem

In population PK/PD modeling, we seek to characterize the distribution of pharmacokinetic parameters across a population of $N$ subjects. Let $\boldsymbol{y}_i = (y_{i1}, \ldots, y_{in_i})^T$ denote the vector of $n_i$ observations for subject $i$, and let $\boldsymbol{\theta} \in \Omega \subset \mathbb{R}^d$ represent the $d$-dimensional parameter vector.

The non-parametric maximum likelihood estimator (NPMLE) seeks to find a discrete probability distribution $F$ over the parameter space $\Omega$ that maximizes the population likelihood:

$$\mathcal{L}(F) = \prod_{i=1}^{N} \int_{\Omega} p(\boldsymbol{y}_i | \boldsymbol{\theta}) \, dF(\boldsymbol{\theta})$$

where $p(\boldsymbol{y}_i | \boldsymbol{\theta})$ is the likelihood of subject $i$'s data given parameters $\boldsymbol{\theta}$.

### 1.2 The Discrete Support Point Representation

By the theory of Lindsay (1983), the NPMLE can be represented as a discrete distribution with at most $N$ support points:

$$\hat{F} = \sum_{j=1}^{K} w_j \delta_{\boldsymbol{\theta}_j}$$

where:

- $K \leq N$ is the number of support points
- $\boldsymbol{\theta}_j \in \Omega$ are the support point locations
- $w_j \geq 0$ with $\sum_{j=1}^{K} w_j = 1$ are the probability weights
- $\delta_{\boldsymbol{\theta}_j}$ is the Dirac delta function at $\boldsymbol{\theta}_j$

The log-likelihood becomes:

$$\ell(F) = \sum_{i=1}^{N} \ln\left(\sum_{j=1}^{K} w_j \, p(\boldsymbol{y}_i | \boldsymbol{\theta}_j)\right)$$

### 1.3 The $\Psi$ Matrix and Weight Optimization

Define the $N \times K$ probability matrix $\Psi$ with elements:

$$\Psi_{ij} = p(\boldsymbol{y}_i | \boldsymbol{\theta}_j)$$

The population likelihood under the discrete model is:

$$P(\boldsymbol{Y} | G) = \prod_{i=1}^{N} \sum_{j=1}^{K} w_j \Psi_{ij} = \prod_{i=1}^{N} (\Psi \boldsymbol{w})_i$$

where $\boldsymbol{w} = (w_1, \ldots, w_K)^T$.

Given fixed support points $\{\boldsymbol{\theta}_j\}_{j=1}^K$, the optimal weights are found by solving the convex optimization problem:

$$\max_{\boldsymbol{w}} \sum_{i=1}^{N} \ln\left((\Psi \boldsymbol{w})_i\right) \quad \text{subject to} \quad \boldsymbol{w} \geq 0, \; \boldsymbol{1}^T \boldsymbol{w} = 1$$

This is solved efficiently using the primal-dual interior point method of Burke (1988).

### 1.4 The D-Optimality Criterion

A fundamental tool in non-parametric estimation is the directional derivative of the likelihood with respect to adding probability mass at a candidate point $\boldsymbol{\theta}^*$. For a current distribution $G$ with $P(Y|G)_i = (\Psi \boldsymbol{w})_i$, the D-criterion at $\boldsymbol{\theta}^*$ is:

$$D(\boldsymbol{\theta}^*; G) = \sum_{i=1}^{N} \frac{p(\boldsymbol{y}_i | \boldsymbol{\theta}^*)}{P(Y|G)_i} - N$$

**Theorem (Lindsay, 1983):** At the NPMLE $\hat{F}$:

1. $D(\boldsymbol{\theta}; \hat{F}) \leq 0$ for all $\boldsymbol{\theta} \in \Omega$
2. $D(\boldsymbol{\theta}_j; \hat{F}) = 0$ for all support points $\boldsymbol{\theta}_j$ with $w_j > 0$

This provides both a measure of optimality and a criterion for identifying promising new support point locations.

### 1.5 NPAG: Non-Parametric Adaptive Grid

NPAG (Schumitzky, 1991; Yamada et al., 2021) uses systematic grid refinement:

**Algorithm (NPAG):**

1. Initialize with a grid of support points spanning $\Omega$
2. Compute $\Psi$ matrix and optimize weights via IPM
3. Remove points with negligible weight via QR rank-revealing factorization
4. Expand remaining points on an adaptive grid with spacing $\varepsilon$
5. If converged in objective, halve $\varepsilon$
6. Repeat until $\varepsilon < \varepsilon_{min}$ and $|P(Y|L)^{(k)} - P(Y|L)^{(k-1)}| < \theta_F$

### 1.6 NPOD: Non-Parametric Optimal Design

NPOD (Baek et al., 2006; D'Argenio et al., 2023) directly optimizes support point locations:

**Algorithm (NPOD):**

1. Initialize support points
2. Compute $\Psi$ matrix and optimize weights
3. Condensation: Remove low-weight points via QR factorization
4. For each support point, optimize its location by maximizing D-criterion using Nelder-Mead
5. Repeat until convergence

---

## 2. Mathematical Preliminaries

### 2.1 Error Model Specification

The observation model for subject $i$ at time $t_{ik}$ is:

$$y_{ik} = f(t_{ik}, \boldsymbol{\theta}_i, \boldsymbol{u}_i) + \epsilon_{ik}$$

where $f$ is the structural model, $\boldsymbol{u}_i$ is the dosing history, and $\epsilon_{ik} \sim \mathcal{N}(0, \sigma^2_{ik})$.

For the combined additive-proportional error model:

$$\sigma_{ik} = \sqrt{\gamma_0^2 + \gamma_1^2 \cdot f(t_{ik}, \boldsymbol{\theta}_i, \boldsymbol{u}_i)^2}$$

The likelihood for subject $i$ given parameters $\boldsymbol{\theta}$ is:

$$p(\boldsymbol{y}_i | \boldsymbol{\theta}) = \prod_{k=1}^{n_i} \frac{1}{\sqrt{2\pi}\sigma_{ik}} \exp\left(-\frac{(y_{ik} - f_{ik})^2}{2\sigma_{ik}^2}\right)$$

### 2.2 The Interior Point Method (Burke's Algorithm)

Given $\Psi$, we maximize $\ell(\boldsymbol{w}) = \sum_{i=1}^N \ln((\Psi\boldsymbol{w})_i)$ subject to $\boldsymbol{w} \geq 0, \boldsymbol{1}^T\boldsymbol{w} = 1$.

The KKT conditions yield the primal-dual system solved iteratively:

$$\begin{pmatrix} -H & A^T & I \\ A & 0 & 0 \\ S & 0 & W \end{pmatrix} \begin{pmatrix} \Delta\boldsymbol{w} \\ \Delta\nu \\ \Delta\boldsymbol{s} \end{pmatrix} = \begin{pmatrix} \nabla\ell - A^T\nu - \boldsymbol{s} \\ \boldsymbol{1}^T\boldsymbol{w} - 1 \\ \mu\boldsymbol{1} - SW\boldsymbol{1} \end{pmatrix}$$

where $H = \nabla^2\ell$, $A = \boldsymbol{1}^T$, $S = \text{diag}(\boldsymbol{s})$, $W = \text{diag}(\boldsymbol{w})$, and $\mu \to 0$ as convergence is achieved.

### 2.3 QR Rank-Revealing Factorization for Condensation

After weight optimization, support points with negligible contribution are removed via column-pivoted QR:

$$\Psi \Pi = QR$$

where $\Pi$ is a permutation matrix. Points are retained if:

$$\frac{|R_{jj}|}{\|\boldsymbol{r}_j\|_2} \geq \tau_{QR}$$

This ensures numerical stability and removes linearly dependent support points.

### 2.4 Convergence Criteria

Standard convergence criteria used across algorithms:

| Symbol               | Value     | Description                             |
| -------------------- | --------- | --------------------------------------- |
| $\theta_\varepsilon$ | $10^{-4}$ | Minimum grid spacing                    |
| $\theta_G$           | $10^{-4}$ | Objective function change threshold     |
| $\theta_F$           | $10^{-2}$ | $P(Y\|L)$ convergence criterion         |
| $\theta_D$           | $10^{-4}$ | Minimum distance between support points |
| $\theta_W$           | $10^{-3}$ | Weight stability threshold              |

---

## 3. NPSAH: Non-Parametric Simulated Annealing Hybrid

Two versions of NPSAH have been developed: the original NPSAH (v1) and an enhanced NPSAH2. We present both formulations.

---

### 3.1 NPSAH (Version 1)

#### 3.1.1 Algorithm Overview

NPSAH v1 combines three complementary strategies:

1. **NPAG's systematic grid exploration** for guaranteed coverage
2. **NPOD's D-optimal refinement** for efficient point placement
3. **Simulated Annealing** for global mode discovery and escaping local optima

#### 3.1.2 Algorithmic Phases

**Phase 1: Warm-up (Cycles $1$ to $C_{warmup}$)**

During warm-up (default $C_{warmup} = 5$), NPSAH uses pure NPAG-style adaptive grid expansion:

$$\Theta^{(k+1)} = \bigcup_{\boldsymbol{\theta}_j \in \Theta^{(k)}} \{\boldsymbol{\theta}_j + \varepsilon \cdot \boldsymbol{e}_m : m = 1,\ldots,d\} \cap \Omega$$

where $\boldsymbol{e}_m$ is the $m$-th standard basis vector and $\varepsilon$ is the current grid spacing.

**Phase 2: Hybrid Expansion (Cycles $> C_{warmup}$)**

The hybrid phase combines three expansion mechanisms:

**(a) D-Optimal Refinement**

For each support point $\boldsymbol{\theta}_j$ with weight $w_j$, perform local optimization:

$$\boldsymbol{\theta}_j^{new} = \arg\max_{\boldsymbol{\theta} \in \mathcal{N}(\boldsymbol{\theta}_j)} D(\boldsymbol{\theta}; G^{(k)})$$

The iteration count is adaptive based on importance:

$$
\text{MaxIter}_j = \begin{cases}
100 & \text{if } w_j / w_{max} > 0.1 \quad \text{(high importance)} \\
10 & \text{otherwise} \quad \text{(low importance)}
\end{cases}
$$

**(b) Sparse Grid Expansion**

In low-density regions, apply grid expansion with reduced spacing:

$$\varepsilon_{sparse} = 0.5 \cdot \varepsilon$$

**(c) Simulated Annealing Injection**

Inject $n_{SA}(T)$ random points according to a temperature-dependent schedule:

$$n_{SA}(T) = \lceil n_{SA}^{max} \cdot T \rceil$$

For each candidate point $\boldsymbol{\theta}^*$ sampled uniformly from $\Omega$:

$$
P(\text{accept } \boldsymbol{\theta}^*) = \begin{cases}
1 & \text{if } D(\boldsymbol{\theta}^*; G) > 0 \\
\exp\left(\frac{D(\boldsymbol{\theta}^*; G)}{T}\right) & \text{otherwise}
\end{cases}
$$

The temperature follows a geometric cooling schedule:

$$T^{(k+1)} = \alpha \cdot T^{(k)}, \quad \alpha = 0.95$$

with initial temperature $T^{(0)} = 1.0$ and minimum temperature $T_{min} = 0.01$.

#### 3.1.3 Multi-Criterion Convergence

NPSAH v1 employs a multi-criterion convergence test:

**Criterion 1: Objective Function Stability**

$$\max_{k-W < j \leq k} |\ell^{(j)} - \ell^{(j-1)}| < \theta_G$$

where $W = 3$ is the convergence window.

**Criterion 2: Global Optimality Check**

Sample $M = 500$ points uniformly from $\Omega$ and verify:

$$\max_{\boldsymbol{\theta}^* \in \text{samples}} D(\boldsymbol{\theta}^*; G^{(k)}) < \theta_{global}$$

where $\theta_{global} = 0.01$.

**Criterion 3: NPAG-style $P(Y|L)$ Criterion**

When $\varepsilon \leq \theta_\varepsilon$:

$$|P(Y|L)^{(k)} - P(Y|L)^{(k-1)}| < \theta_F$$

---

### 3.2 NPSAH2 (Version 2) — Enhanced Algorithm

NPSAH2 introduces significant improvements over v1, designed with three goals in mind:

- Better exploration of multimodal distributions
- Improved convergence criteria and bias-variance tradeoff
- Parallelized operations, memory efficiency, early termination

#### 3.2.1 Key Improvements over NPSAH v1

| Feature              | NPSAH v1                | NPSAH2                                               |
| -------------------- | ----------------------- | ---------------------------------------------------- |
| Temperature schedule | Fixed geometric cooling | Adaptive based on acceptance ratio                   |
| Elite preservation   | None                    | Top 3 points preserved across cycles                 |
| Exploration          | Grid + random SA        | LHS + local SA + gradient-informed                   |
| Restart mechanism    | None                    | Automatic restart after stagnation                   |
| Phase structure      | 2 phases                | 4 phases (Warmup, Hybrid, Exploitation, Convergence) |
| Condensation         | Fixed threshold         | Phase-adaptive threshold                             |

#### 3.2.2 Four-Phase Architecture

```
+----------+     +----------+     +-------------+     +-------------+
|  WARMUP  | --> |  HYBRID  | --> | EXPLOITATION| --> | CONVERGENCE |
| (1-3)    |     | (4-6)    |     | (T > 2*Tmin)|     | (T <= 2*Tmin)|
+----------+     +----------+     +-------------+     +-------------+
```

**Phase 1: Warmup (Cycles 1-3)**

- Latin Hypercube Sampling (LHS) for space-filling initial coverage
- NPAG-style adaptive grid expansion

**Phase 2: Hybrid (Cycles 4-6)**

- D-optimal refinement
- Local SA moves around high-weight points
- Sparse grid expansion
- Global SA injection
- Elite point re-injection

**Phase 3: Exploitation (While $T > 2 T_{min}$)**

- D-optimal refinement (high-weight points only)
- Light grid expansion
- Reduced SA activity

**Phase 4: Convergence (When $T \leq 2 T_{min}$)**

- Minimal expansion
- Focus on convergence verification

#### 3.2.3 Adaptive Temperature Schedule

NPSAH2 tracks the SA acceptance ratio over each cycle:

$$\hat{r}^{(k)} = \frac{a^{(k)}}{n^{(k)}}$$

where $a^{(k)}$ is accepted moves and $n^{(k)}$ is proposed moves.

**Adaptive Cooling Rate:**

$$
\alpha^{(k+1)} = \begin{cases}
\min(\alpha^{(k)} + 0.02, 0.98) & \text{if } \hat{r} < 0.125 \; \text{(too cold)} \\
\alpha^{(k)} & \text{if } 0.125 \leq \hat{r} \leq 0.375 \\
\max(\alpha^{(k)} - 0.02, 0.85) & \text{if } \hat{r} > 0.375 \; \text{(too hot)}
\end{cases}
$$

with base cooling rate $\alpha^{(0)} = 0.88$ and target acceptance ratio $r_{target} = 0.25$.

**Reheat Mechanism:**

When $\hat{r} < 0.1$ and $T < 0.5$:

$$T^{(k+1)} = T^{(k)} \cdot \beta_{reheat}, \quad \beta_{reheat} = 1.3$$

This prevents the algorithm from becoming stuck when temperature drops too quickly.

#### 3.2.4 Latin Hypercube Sampling (Warmup)

Generate $M_{LHS} = 30$ initial samples using stratified random sampling:

For dimension $m = 1, \ldots, d$:

1. Divide $[\theta_m^{min} + \delta_m, \theta_m^{max} - \delta_m]$ into $M_{LHS}$ equal intervals
2. Generate a random permutation $\pi_m$ of $\{0, 1, \ldots, M_{LHS}-1\}$
3. For sample $i$: $\theta_{i,m} = \theta_m^{min} + \delta_m + (\pi_m(i) + U_i) \cdot h_m$

where $h_m = (\theta_m^{max} - \theta_m^{min} - 2\delta_m) / M_{LHS}$, $U_i \sim \text{Uniform}[0, h_m]$, and $\delta_m$ is the boundary margin.

**Theorem 3.1 (LHS Variance Reduction):** For additive functions $f(\boldsymbol{\theta}) = \sum_m g_m(\theta_m)$, LHS achieves variance reduction over simple random sampling:

$$\text{Var}_{LHS}[\bar{f}] \leq \text{Var}_{SRS}[\bar{f}]$$

with equality only when $f$ is constant.

#### 3.2.5 Local SA Moves

For each high-importance support point ($w_j / w_{max} > 0.025$), generate local perturbations:

$$\boldsymbol{\theta}^* = \boldsymbol{\theta}_j + T \cdot 0.1 \cdot \text{diag}(\boldsymbol{r}) \cdot \boldsymbol{\delta}$$

where $\delta_m \sim \text{Uniform}[-1, 1]$ and $r_m = \theta_m^{max} - \theta_m^{min}$.

Accept with standard Metropolis criterion. This focuses exploration around promising regions.

#### 3.2.6 Elite Point Preservation

Maintain a buffer $\mathcal{E}$ of up to $E = 3$ elite points:

$$\mathcal{E} = \{(\boldsymbol{\theta}, D(\boldsymbol{\theta}; G), \text{age})\}$$

**Update rules:**

1. Age all elite points: $\text{age}_e \leftarrow \text{age}_e + 1$
2. Remove stale elites: $\mathcal{E} \leftarrow \{e \in \mathcal{E} : \text{age}_e < 20\}$
3. Add top-weight points if not already elite
4. Sort by D-criterion, keep top $E$

Elite points are re-injected at each cycle to prevent loss of good solutions during aggressive condensation.

#### 3.2.7 Phase-Adaptive Condensation

The $\lambda$-filter threshold adapts to the algorithm phase:

| Phase        | Divisor | Effect                           |
| ------------ | ------- | -------------------------------- |
| Warmup       | 1,000   | Conservative (keep more points)  |
| Hybrid       | 5,000   | Moderate                         |
| Exploitation | 10,000  | Aggressive (focus on top points) |
| Convergence  | 10,000  | Aggressive                       |

$$\text{threshold}^{(k)} = \frac{\max_j \lambda_j}{\text{divisor}(\text{phase})}$$

#### 3.2.8 Restart Mechanism

If no improvement in objective function for $C_{stag} = 15$ consecutive cycles:

1. Reset temperature: $T \leftarrow T^{(0)}$
2. Reset cooling rate: $\alpha \leftarrow \alpha^{(0)}$
3. Inject new LHS samples
4. Re-inject elite points
5. Reset stagnation counter

Maximum restarts: $R_{max} = 2$.

#### 3.2.9 Hierarchical D-Optimal Refinement

Three-tier iteration budget based on relative weight:

| Tier   | Criterion                         | Max Iterations |
| ------ | --------------------------------- | -------------- |
| High   | $w_j / w_{max} > 0.05$            | 80             |
| Medium | $0.005 < w_j / w_{max} \leq 0.05$ | 30             |
| Low    | $0.01 < w_j / w_{max} \leq 0.005$ | 10             |
| Skip   | $w_j / w_{max} \leq 0.01$         | 0              |

Only points above 1% relative weight are refined, saving computation.

### 3.3 Mathematical Justification

**Theorem 3.2 (Convergence of SA Component):** Under the adaptive cooling schedule with reheat, the simulated annealing component maintains ergodicity:

$$\lim_{k \to \infty} P(\boldsymbol{\theta}^{(k)} \in \mathcal{G}) = 1$$

where $\mathcal{G}$ is the set of global optima.

_Proof sketch:_ The reheat mechanism ensures that the temperature never remains at $T_{min}$ indefinitely when acceptance drops too low. This maintains the sufficient condition $\sum_k T^{(k)} = \infty$ even with adaptive cooling, guaranteeing convergence to global optima.

**Theorem 3.3 (Elite Preservation Stability):** The elite preservation mechanism ensures that:

$$\max_{j} D(\boldsymbol{\theta}_j^{(k)}; G^{(k)}) \geq \max_{j} D(\boldsymbol{\theta}_j^{(k-20)}; G^{(k-20)}) - \epsilon$$

for sufficiently small $\epsilon > 0$, preventing catastrophic loss of solution quality.

_Proof sketch:_ Elite points are preserved for 20 cycles and re-injected, so any high-D point discovered within the last 20 cycles remains in consideration. The bounded age ensures this property holds over finite horizons.

---

## 4. NPCAT: Non-Parametric Covariance-Adaptive Trajectory

### 4.1 Algorithm Overview

NPCAT introduces four key innovations:

1. **Fisher Information-guided sampling** for principled exploration
2. **Sobol quasi-random sequences** for deterministic space-filling coverage
3. **Hierarchical convergence state machine** for adaptive behavior
4. **Selective local refinement** with gradient-aware optimization

### 4.2 Fisher Information Estimation

The Fisher Information matrix for the population model is approximated by:

$$\mathcal{I}(\boldsymbol{\theta}) \approx \sum_{i=1}^{N} \mathbb{E}\left[\nabla_\theta \ln p(\boldsymbol{y}_i|\boldsymbol{\theta}) \nabla_\theta \ln p(\boldsymbol{y}_i|\boldsymbol{\theta})^T\right]$$

NPCAT uses an empirical diagonal approximation based on the weighted variance of support points:

$$\hat{\mathcal{I}}_{mm}^{-1} = \text{Var}_G[\theta_m] = \sum_{j=1}^K w_j (\theta_{jm} - \bar{\theta}_m)^2$$

where $\bar{\theta}_m = \sum_j w_j \theta_{jm}$ is the weighted mean of dimension $m$.

**Interpretation:** High variance (low Fisher Information) in dimension $m$ indicates parameter uncertainty and the need for additional exploration in that direction.

### 4.3 Convergence State Machine

NPCAT operates through three phases with well-defined transition criteria:

```
+--------------+         +--------------+         +--------------+
|  EXPLORING   | ------> |   REFINING   | ------> |  POLISHING   |
|              |         |              |         |              |
| High K(t)    |         | Moderate K(t)|         | K(t) = 0     |
| Fisher-guided|         | Global checks|         | Full refine  |
+--------------+         +--------------+         +--------------+
```

**Transition: Exploring → Refining**

Requires conjunctively:

1. Objective function stable for $W_E = 3$ cycles
2. Sufficient coverage: $K \geq 2d$ support points

**Transition: Refining → Polishing**

Requires conjunctively:

1. Objective stable for $W_R = 5$ cycles
2. Global optimality check passed
3. Weight distribution stable: $\max_j |w_j^{(k)} - w_j^{(k-1)}| / w_j^{(k)} < \theta_W$

### 4.4 Information-Guided Candidate Generation

At each cycle, generate $K(t)$ candidates from three sources:

$$K(t) = \max\left(K_{min}, \lfloor K_0 \cdot \rho^t \rfloor\right)$$

where $K_0 = 40$, $\rho = 0.95$, and $K_{min} = 4$.

**4.4.1 Fisher Information Candidates (60%)**

For each existing support point $\boldsymbol{\theta}_j$, generate candidates along high-variance dimensions:

$$\boldsymbol{\theta}^*_{jm\pm} = \boldsymbol{\theta}_j \pm \Delta_m \boldsymbol{e}_m$$

where the step size adapts to uncertainty:

$$\Delta_m = \min\left(\max\left(\sqrt{\hat{\mathcal{I}}_{mm}^{-1}} \cdot (\theta_m^{max} - \theta_m^{min}), 0.05 \cdot r_m\right), 0.3 \cdot r_m\right)$$

and $r_m = \theta_m^{max} - \theta_m^{min}$ is the range.

**4.4.2 D-Optimal Gradient Candidates (30%)**

For high-weight points ($w_j > w_{median}$), compute the D-criterion gradient:

$$\nabla D(\boldsymbol{\theta}_j) \approx \left(\frac{D(\boldsymbol{\theta}_j + \delta_m \boldsymbol{e}_m) - D(\boldsymbol{\theta}_j - \delta_m \boldsymbol{e}_m)}{2\delta_m}\right)_{m=1}^d$$

Move in the direction of steepest ascent:

$$\boldsymbol{\theta}^* = \boldsymbol{\theta}_j + \eta \cdot \text{sign}(\nabla D(\boldsymbol{\theta}_j))$$

**4.4.3 Boundary Candidates (10%)**

Sample points near parameter boundaries to ensure edge coverage:

$$
\theta_m^* \sim \begin{cases}
\text{Uniform}[\theta_m^{min}, \theta_m^{min} + 0.1 \cdot r_m] & \text{w.p. } 0.5 \\
\text{Uniform}[\theta_m^{max} - 0.1 \cdot r_m, \theta_m^{max}] & \text{w.p. } 0.5
\end{cases}
$$

### 4.5 Sobol Quasi-Random Global Optimality Check

Unlike Monte Carlo sampling, NPCAT uses Sobol low-discrepancy sequences for guaranteed uniform coverage:

$$\boldsymbol{s}_i = \text{Sobol}(i, d), \quad i = 1, \ldots, M_{Sobol}$$

where $M_{Sobol} = 256$ and each $\boldsymbol{s}_i \in [0,1]^d$.

Scale to parameter space:

$$\boldsymbol{\theta}_i = \boldsymbol{\theta}^{min} + \boldsymbol{s}_i \odot (\boldsymbol{\theta}^{max} - \boldsymbol{\theta}^{min})$$

The global check passes if:

$$\max_{i=1,\ldots,M_{Sobol}} D(\boldsymbol{\theta}_i; G) < \theta_D^{global}$$

**Theorem 4.1 (Star Discrepancy Bound):** The Sobol sequence satisfies:

$$D_N^* \leq C_d \frac{(\ln N)^d}{N}$$

This guarantees asymptotically better coverage than pseudo-random sampling with $D_N^* \sim N^{-1/2}$.

### 4.6 Selective Local Refinement

**Refining Phase:** Only refine points with $w_j \geq w_{median}$:

$$\boldsymbol{\theta}_j^{new} = \arg\min_{\boldsymbol{\theta} \in \Omega} -D(\boldsymbol{\theta}; G)$$

using Nelder-Mead with adaptive iterations:

$$\text{MaxIter} = 20 + 10 \cdot \ln(k)$$

**Polishing Phase:** Refine all points with doubled iteration budget.

### 4.7 Final Convergence Criterion

In the Polishing phase, convergence requires:

1. Objective stable for $W_P = 3$ cycles
2. Weights stable
3. $P(Y|L)$ criterion: $|P(Y|L)^{(k)} - P(Y|L)^{(k-1)}| < \theta_F$

---

## 5. NPOPT: Non-Parametric OPTimal Trajectory

### 5.1 Algorithm Overview

NPOPT synthesizes the best elements from NPSAH, NPCAT, and introduces:

1. **Adaptive SA with automatic reheat** to escape local optima
2. **Stratified Sobol initialization** for space-filling coverage
3. **Elite preservation** to prevent loss of good solutions
4. **Subject residual injection** for targeted mode discovery

### 5.2 Three-Phase Architecture

**Phase 1: Exploration (Cycles 1-3)**

- Stratified Sobol initialization: $M_{init} = 50$ points
- Sparse adaptive grid expansion
- Begin Fisher Information tracking

**Phase 2: Refinement (Cycles 4+)**

- Hierarchical D-optimal refinement
- Adaptive SA injection with reheat
- Fisher-guided expansion
- Subject residual injection
- Elite preservation
- Periodic global checks (every 3 cycles)

**Phase 3: Polishing (After global check passes)**

- Full D-optimal refinement
- No expansion
- Convergence monitoring

### 5.3 Adaptive Simulated Annealing with Reheat

NPOPT tracks the SA acceptance ratio over a rolling window:

$$\hat{r}^{(k)} = \frac{\sum_{t=k-W_{SA}}^{k} a_t}{\sum_{t=k-W_{SA}}^{k} n_t}$$

where $a_t$ is accepted points and $n_t$ is proposed points at cycle $t$, with $W_{SA} = 5$.

**Adaptive Cooling:**

The effective cooling rate adapts to acceptance:

$$
\alpha_{eff} = \begin{cases}
\alpha_{base} \cdot (1 + 0.1) & \text{if } \hat{r} > r_{target} \cdot 1.5 \\
\alpha_{base} & \text{if } r_{target}/2 < \hat{r} < r_{target} \cdot 1.5 \\
\alpha_{base} \cdot (1 - 0.1) & \text{if } \hat{r} < r_{target}/2
\end{cases}
$$

where $\alpha_{base} = 0.90$ and $r_{target} = 0.23$ (theoretically optimal acceptance ratio).

**Automatic Reheat:**

When $\hat{r} < r_{reheat} = 0.08$ (SA becoming ineffective):

$$T^{(k+1)} = \min(T^{(k)} \cdot \beta_{reheat}, T^{(0)})$$

where $\beta_{reheat} = 1.5$.

### 5.4 Hierarchical D-Optimal Refinement

Support points are categorized by relative weight:

| Category | Criterion                       | Iterations |
| -------- | ------------------------------- | ---------- |
| High     | $w_j/w_{max} > 0.10$            | 80         |
| Medium   | $0.01 < w_j/w_{max} \leq 0.10$  | 30         |
| Low      | $0.001 < w_j/w_{max} \leq 0.01$ | 10         |
| Skip     | $w_j/w_{max} \leq 0.001$        | 0          |

Optimization uses bounded Nelder-Mead with reflection at boundaries.

### 5.5 Subject Residual Injection

Identify the $M_{res} = 3$ subjects with lowest fit:

$$\mathcal{S}_{poor} = \arg\min_{|\mathcal{S}| = M_{res}} \sum_{i \in \mathcal{S}} P(Y|G)_i$$

For each poorly-fit subject $i \in \mathcal{S}_{poor}$, find the MAP estimate:

$$\boldsymbol{\theta}_i^{MAP} = \arg\max_{\boldsymbol{\theta} \in \Omega} p(\boldsymbol{y}_i | \boldsymbol{\theta})$$

using COBYLA with $N_{eval} = 30$ evaluations, initialized at the weighted centroid:

$$\bar{\boldsymbol{\theta}} = \sum_{j=1}^K w_j \boldsymbol{\theta}_j$$

Add $\boldsymbol{\theta}_i^{MAP}$ if $D(\boldsymbol{\theta}_i^{MAP}; G) > 0$ and distance constraint satisfied.

### 5.6 Elite Preservation

Maintain a buffer of $E = 5$ elite points:

$$\mathcal{E} = \{(\boldsymbol{\theta}, D(\boldsymbol{\theta}; G), k)\}$$

ordered by D-criterion. At each cycle:

1. Evaluate D-criterion for current high-weight points
2. Update $\mathcal{E}$ with any improvements
3. Ensure elite points are included in support point set

This prevents the accidental loss of good solutions during aggressive condensation.

### 5.7 Fisher-Guided Expansion

Generate $K_F = 20$ candidates along high-variance directions.

For the top $\lceil d/2 \rceil$ dimensions by variance, create candidates:

$$\boldsymbol{\theta}^* = \bar{\boldsymbol{\theta}} + U \cdot \sqrt{\hat{\mathcal{I}}_{mm}^{-1}} \cdot \boldsymbol{e}_m$$

where $U \sim \text{Uniform}(-1, 1)$.

### 5.8 Convergence Criteria

NPOPT converges when:

1. **Global check passes twice consecutively:**
   $$\max_{\boldsymbol{s} \in \text{Sobol}(256)} D(\boldsymbol{s}; G) < 0.008$$

2. **Weights stable:**
   $$\|\boldsymbol{w}^{(k)} - \boldsymbol{w}^{(k-1)}\|_\infty < \theta_W$$

3. **NPAG-style criterion met** (when $\varepsilon \leq \theta_\varepsilon$)

---

## 6. NPPSO: Non-Parametric Particle Swarm Optimization

### 6.1 Algorithm Overview

NPPSO introduces true swarm intelligence to non-parametric estimation:

1. **Particle swarm dynamics** with D-criterion fitness
2. **Collective learning** through personal and global bests
3. **Velocity-based exploration** enabling escape from local optima
4. **Hybrid integration** with SA, MAP, and grid components

### 6.2 Particle Swarm Formulation

Initialize a swarm of $S = 40$ particles $\{(\boldsymbol{x}_s, \boldsymbol{v}_s)\}_{s=1}^S$ where:

- $\boldsymbol{x}_s \in \Omega$ is position (parameter values)
- $\boldsymbol{v}_s \in \mathbb{R}^d$ is velocity

**Initialization:**

$$x_{s,m} \sim \text{Uniform}[\theta_m^{min} + \delta_m, \theta_m^{max} - \delta_m]$$
$$v_{s,m} \sim \text{Uniform}[-0.015 \cdot r_m, 0.015 \cdot r_m]$$

where $\delta_m = 0.001 \cdot r_m$ is a boundary margin.

### 6.3 Swarm Dynamics

Each particle maintains:

- Position $\boldsymbol{x}_s^{(k)}$
- Velocity $\boldsymbol{v}_s^{(k)}$
- Personal best position $\boldsymbol{p}_s^{(k)}$ and fitness $f_s^{(k)}$
- Global best position $\boldsymbol{g}^{(k)}$ (shared)

**Velocity Update:**

$$v_{s,m}^{(k+1)} = \omega^{(k)} v_{s,m}^{(k)} + c_1 r_1 (p_{s,m}^{(k)} - x_{s,m}^{(k)}) + c_2 r_2 (g_m^{(k)} - x_{s,m}^{(k)})$$

where:

- $\omega^{(k)}$ is adaptive inertia weight
- $c_1 = 2.0$ is cognitive acceleration
- $c_2 = 2.0$ is social acceleration
- $r_1, r_2 \sim \text{Uniform}[0,1]$ are random numbers

**Adaptive Inertia:**

$$\omega^{(k)} = \omega_{max} - (\omega_{max} - \omega_{min}) \cdot \frac{k}{k_{max}}$$

with $\omega_{max} = 0.9$, $\omega_{min} = 0.4$. This balances exploration (high $\omega$) early and exploitation (low $\omega$) later.

**Velocity Clamping:**

$$v_{s,m}^{(k+1)} = \text{clamp}(v_{s,m}^{(k+1)}, -v_{max,m}, v_{max,m})$$

where $v_{max,m} = 0.15 \cdot r_m$.

**Position Update with Boundary Reflection:**

$$x_{s,m}^{(k+1)} = x_{s,m}^{(k)} + v_{s,m}^{(k+1)}$$

If $x_{s,m}^{(k+1)} < \theta_m^{min} + \delta_m$:
$$x_{s,m}^{(k+1)} = \theta_m^{min} + \delta_m + (\theta_m^{min} + \delta_m - x_{s,m}^{(k+1)})$$
$$v_{s,m}^{(k+1)} \leftarrow -0.5 \cdot v_{s,m}^{(k+1)}$$

(Similar for upper boundary.)

### 6.4 D-Criterion Fitness Function

The fitness of particle $s$ is the D-criterion at its position:

$$\phi(\boldsymbol{x}_s) = D(\boldsymbol{x}_s; G^{(k)}) = \sum_{i=1}^{N} \frac{p(\boldsymbol{y}_i | \boldsymbol{x}_s)}{P(Y|G^{(k)})_i} - N$$

Particles seek regions where the D-criterion is positive (indicating potential for improvement).

### 6.5 Support Point Injection from Swarm

At each PSO cycle, inject support points from:

1. **Current positions** with $\phi(\boldsymbol{x}_s) > \tau \cdot \max_s \phi(\boldsymbol{x}_s)$ where $\tau = 0.5$
2. **Personal bests** satisfying the same criterion
3. **Global best** (always included if positive D)

### 6.6 Swarm Diversity Maintenance

Monitor swarm convergence via centroid distance:

$$\rho^{(k)} = \frac{1}{S} \sum_{s=1}^S \frac{\|\boldsymbol{x}_s^{(k)} - \bar{\boldsymbol{x}}^{(k)}\|}{\text{diam}(\Omega)}$$

where $\bar{\boldsymbol{x}} = S^{-1} \sum_s \boldsymbol{x}_s$ and $\text{diam}(\Omega) = \|\boldsymbol{\theta}^{max} - \boldsymbol{\theta}^{min}\|$.

If $\rho^{(k)} < \rho_{threshold} = 0.2$ (swarm converging):

- Reinject $\lfloor 0.25 \cdot S \rfloor$ worst-performing particles randomly

### 6.7 Hybrid Components

NPPSO integrates additional mechanisms:

**6.7.1 SA Injection**

Identical to NPSAH's SA component, providing stochastic exploration independent of swarm dynamics.

**6.7.2 Subject MAP Injection**

Target $M_{res} = 2$ poorly-fit subjects with COBYLA-based MAP estimation.

**6.7.3 Periodic D-Optimal Refinement**

Every 10 cycles, refine high-weight points ($w_j > 0.05 \cdot w_{max}$) using COBYLA.

**6.7.4 Sparse Grid Expansion**

Every 3 cycles, apply NPAG-style grid expansion with $\varepsilon_{sparse} = 0.5 \cdot \varepsilon$.

### 6.8 NPPSO Algorithm Summary

```
Algorithm: NPPSO
Input: Data Y, model f, parameter bounds Omega
Output: Support points Theta, weights w

1. Initialize: swarm S, theta from prior, eps = 0.2
2. For k = 1 to k_max:
   3. If k <= C_warmup: // NPAG warm-up
        Adaptive grid expansion
      Else: // PSO-driven
        4. Compute Psi matrix, run IPM -> w, lambda, objf
        5. Condensation via lambda-filter and QR
        6. Evaluate particle fitness: phi(x_s) = D(x_s; G)
        7. Update personal bests, global best
        8. PSO velocity and position update
        9. Inject high-fitness particles as support points
        10. SA injection with Metropolis criterion
        11. Subject MAP injection for poor fits
        12. Periodic D-optimal refinement
        13. Periodic sparse grid expansion
        14. Diversity maintenance (reinject if converging)
   15. Check convergence criteria
   16. If converged: break
17. Return Theta, w
```

### 6.9 Convergence Analysis

**Theorem 6.1 (PSO Convergence):** Under the standard PSO dynamics with $\omega < 1$ and bounded velocities, each particle's position sequence $\{\boldsymbol{x}_s^{(k)}\}$ converges to a point in the convex hull of $\{\boldsymbol{p}_s, \boldsymbol{g}\}$.

**Corollary:** The diversity maintenance mechanism (particle reinjection) ensures continued exploration, preventing premature convergence to suboptimal regions.

NPPSO's global optimality check (identical to NPSAH) provides the formal termination criterion.

---

## 7. Comparative Analysis

### 7.1 Expansion Strategy Comparison

| Algorithm | Primary Expansion       | Secondary Expansion           | Global Search        |
| --------- | ----------------------- | ----------------------------- | -------------------- |
| NPAG      | Adaptive grid           | None                          | Exhaustive grid      |
| NPOD      | D-optimal (Nelder-Mead) | None                          | None                 |
| NPSAH v1  | Adaptive grid (warm-up) | D-optimal + SA injection      | MC sampling          |
| NPSAH2    | LHS + Local SA          | D-optimal + Grid + Elite      | MC + Restart         |
| NPCAT     | Fisher-guided           | D-optimal gradient + Boundary | Sobol sequence       |
| NPOPT     | Fisher + Sobol init     | SA with reheat + Subject MAP  | Sobol sequence       |
| NPPSO     | PSO swarm               | SA + Subject MAP + Grid       | MC + PSO exploration |

### 7.2 Convergence Criteria Comparison

| Algorithm | Primary                            | Secondary                   | Tertiary            |
| --------- | ---------------------------------- | --------------------------- | ------------------- |
| NPAG      | $\varepsilon < \theta_\varepsilon$ | $\Delta P(Y\|L) < \theta_F$ | —                   |
| NPOD      | $\Delta\ell < \theta_F$            | —                           | —                   |
| NPSAH v1  | Multi-criterion                    | $\varepsilon$-based         | Global MC check     |
| NPSAH2    | Phase-based                        | Stagnation restart          | Early convergence   |
| NPCAT     | State machine                      | Global Sobol check          | $P(Y\|L)$ criterion |
| NPOPT     | $2\times$ global check             | Weight stability            | $P(Y\|L)$ criterion |
| NPPSO     | NPAG-style $\varepsilon$           | $P(Y\|L)$ criterion         | Global check        |

### 7.3 Computational Complexity

Let $N$ = subjects, $K$ = support points, $d$ = dimensions, $C$ = cycles.

| Algorithm | Per-Cycle Complexity                    | Notes                          |
| --------- | --------------------------------------- | ------------------------------ |
| NPAG      | $O(NK + K \cdot 2d)$                    | Grid expansion scales with $d$ |
| NPOD      | $O(NK + K \cdot \text{NM iterations})$  | Nelder-Mead dominates          |
| NPSAH     | $O(NK + K \cdot d + M_{SA} \cdot N)$    | SA injection adds overhead     |
| NPCAT     | $O(NK + K \cdot d + M_{Sobol} \cdot N)$ | Sobol check periodic           |
| NPOPT     | $O(NK + K \cdot d + S_{SA} \cdot N)$    | Similar to NPSAH               |
| NPPSO     | $O(NK + S \cdot N + K_{inj} \cdot N)$   | Swarm evaluation adds cost     |

### 7.4 Theoretical Properties

| Property                     | NPAG | NPOD | NPSAH v1 | NPSAH2 | NPCAT | NPOPT | NPPSO |
| ---------------------------- | ---- | ---- | -------- | ------ | ----- | ----- | ----- |
| Guaranteed coverage          | X    | -    | X        | X      | X     | X     | ~     |
| Local optima escape          | -    | -    | X        | XX     | ~     | X     | X     |
| Information-guided           | -    | X    | ~        | ~      | X     | X     | ~     |
| Deterministic                | X    | ~    | -        | -      | ~     | -     | -     |
| Adaptive resource allocation | -    | -    | X        | XX     | X     | X     | X     |
| Elite preservation           | -    | -    | -        | X      | -     | X     | X     |
| Restart mechanism            | -    | -    | -        | X      | -     | ~     | ~     |

(XX indicates enhanced capability)

### 7.5 Empirical Performance Summary

From the comparative benchmark with identical initial conditions:

| Algorithm | -2LL               | Support Points | Cycles | Time         |
| --------- | ------------------ | -------------- | ------ | ------------ |
| NPAG      | -386.38            | 45             | 227    | 5.57s        |
| NPOD      | -381.00            | 45             | 11     | 2.10s        |
| NPSAH     | -418.14 to -419.52 | 45-48          | 18-31  | 28.68-59.30s |
| NPCAT     | -419.50            | 45             | 29     | 21.87s       |
| NPOPT     | -419.55            | 43             | 15     | 26.75s       |
| NPPSO     | -425.73            | 47             | 119    | 22.15s       |

**Key Observations:**

1. **Best likelihood:** NPPSO achieves the best -2LL (-425.73), demonstrating the swarm's ability to discover superior solutions.

2. **Efficiency:** NPOD remains fastest due to direct D-optimal moves without expansion overhead.

3. **Robustness:** NPSAH, NPCAT, NPOPT cluster around -419, suggesting they find similar local optima.

4. **Novel algorithms outperform classics:** All four new algorithms achieve better likelihoods than NPAG/NPOD.

---

---

## Appendix A: Algorithm Pseudocode

### A.1 NPSAH Pseudocode

```
Algorithm: NPSAH
Input: Data Y, model f, parameter bounds Omega, C_warmup
Output: Support points Theta, weights w

1. Initialize: Theta from prior, eps = 0.2, T = 1.0
2. For k = 1 to k_max:
   3. Compute Psi matrix from (Y, Theta, f)
   4. lambda, objf <- Burke_IPM(Psi)
   5. Condensation: lambda-filter then QR(Psi)
   6. lambda, objf <- Burke_IPM(Psi)  // recompute after condensation
   7. w <- lambda
   8. Optimize error model parameters

   9. If k <= C_warmup:
        Theta <- AdaptiveGrid(Theta, eps)
      Else:
        // D-optimal refinement
        10. For each theta_j with importance w_j/max(w):
              theta_j <- NelderMead(D(.; G), theta_j, max_iters)
        // Sparse grid
        11. Theta <- AdaptiveGrid(Theta, eps/2)
        // SA injection
        12. n_inject <- ceil(10 * T)
            For i = 1 to n_inject:
              theta* <- Uniform(Omega)
              d* <- D(theta*; G)
              If d* > 0 or rand() < exp(d*/T):
                Theta <- Theta + {theta*}

   13. Evaluate convergence:
         - Track objf_history
         - Check multi-criterion convergence
   14. Update: T <- 0.95 * T
   15. If eps stable and delta_objf < theta_G: eps <- eps/2
   16. If converged: break

17. Return Theta, w
```

### A.2 NPCAT Pseudocode

```
Algorithm: NPCAT
Input: Data Y, model f, parameter bounds Omega
Output: Support points Theta, weights w

1. Initialize: Theta from prior, state = EXPLORING, K = 40
2. For k = 1 to k_max:
   3. Compute Psi, run IPM -> w, lambda, objf
   4. Condensation via lambda-filter and QR
   5. Update Fisher diagonal from weighted variance
   6. Store w_prev <- w

   7. Switch state:
      EXPLORING:
        8. Generate K candidates:
             - 60% Fisher-guided
             - 30% D-optimal gradient
             - 10% boundary
        9. Add valid candidates to Theta
        10. If objf stable AND K >= 2d: state <- REFINING

      REFINING:
        11. Selective refinement (median-weight threshold)
        12. Generate candidates (as above)
        13. If cycle mod 5 == 0: Sobol global check
        14. If objf stable AND global passed AND weights stable:
              state <- POLISHING

      POLISHING:
        15. Full refinement (all points)
        16. Check P(Y|L) criterion
        17. If all criteria met: state <- CONVERGED

   18. K <- max(4, K * 0.95)  // decay candidate count
   19. If converged: break

20. Return Theta, w
```

### A.3 NPOPT Pseudocode

```
Algorithm: NPOPT
Input: Data Y, model f, parameter bounds Omega
Output: Support points Theta, weights w

1. Initialize: Theta via Sobol, phase = EXPLORATION, T = 2.0
2. For k = 1 to k_max:
   3. Compute Psi, run IPM -> w, lambda, objf
   4. Condensation
   5. Update Fisher diagonal, elite points

   6. Switch phase:
      EXPLORATION (k <= 3):
        7. Sparse adaptive grid
        8. Initialize Fisher tracking
        9. If k > 3: phase <- REFINEMENT

      REFINEMENT:
        10. Hierarchical D-optimal refinement
        11. Adaptive SA injection with acceptance tracking
        12. Fisher-guided expansion
        13. Subject residual MAP injection (top 3 poor fits)
        14. Elite preservation
        15. If k mod 3 == 0: Sobol global check
        16. If 2 consecutive passes: phase <- POLISHING

      POLISHING:
        17. Full D-optimal refinement, no expansion
        18. Monitor convergence

   19. Adapt T based on acceptance ratio
   20. If converged: break

21. Return Theta, w
```

### A.4 NPPSO Pseudocode

```
Algorithm: NPPSO
Input: Data Y, model f, parameter bounds Omega
Output: Support points Theta, weights w

1. Initialize: Theta from prior, swarm S of 40 particles, eps = 0.2
2. For k = 1 to k_max:
   3. If k <= 3: // Warm-up
        AdaptiveGrid(Theta, eps)
      Else: // PSO phase
        4. Compute Psi, run IPM -> w, lambda, objf
        5. Condensation
        6. pyl <- Psi * w

        // PSO updates
        7. For each particle s:
             phi_s <- D(x_s; G)  // fitness
        8. Update personal bests p_s, global best g
        9. omega <- adaptive_inertia(k)
        10. For each particle s:
              v_s <- omega*v_s + c1*r1*(p_s - x_s) + c2*r2*(g - x_s)
              v_s <- clamp(v_s, -v_max, v_max)
              x_s <- x_s + v_s
              x_s <- reflect_boundary(x_s)

        // Injection to support points
        11. For particles with phi_s > 0.5 * max(phi):
              If dist(x_s, Theta) > theta_D: Theta <- Theta + {x_s}

        // Auxiliary mechanisms
        12. SA_injection()
        13. Subject_MAP_injection(worst 2)
        14. If k mod 10 == 0: D_optimal_refinement()
        15. If k mod 3 == 0: SparseGrid(Theta, eps/2)
        16. Elite_preservation()

        // Diversity
        17. If swarm_convergence() > 0.8:
              Reinject 25% of particles

        18. T_SA <- 0.95 * T_SA

   19. NPAG-style eps convergence check
   20. If converged and global_check_passed: break

21. Return Theta, w
```

---

## Appendix B: Implementation Notes

### B.1 Numerical Stability

All algorithms implement:

- Logarithmic likelihood computation to prevent underflow
- Coercion of NaN/Inf values in $\Psi$ to zero with warning
- Minimum distance constraints ($\theta_D = 10^{-4}$) to prevent support point collapse
- Condition number monitoring in QR factorization

### B.2 Parallelization

The following operations are parallelized across CPU cores:

- $\Psi$ matrix computation (across subjects)
- D-optimal refinement (across support points)
- Particle fitness evaluation in NPPSO (across particles)
- Fisher-guided candidate generation

### B.3 Random Number Generation

All stochastic algorithms use seeded random number generators for reproducibility:

- Default seed from settings or 42
- StdRng from the `rand` crate (ChaCha-based)

---

_Document prepared for peer review. For implementation details, see the PMcore source code repository._
