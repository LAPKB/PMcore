# Beyond the Adaptive Grid: A Comparative Study of Non-Parametric Support Point Optimization Algorithms for Population Pharmacokinetics

**Authors**: Julian D. Otalvaro, Markus Hovd, Alona Kryshchenko, Walter M. Yamada, Michael N. Neely

**Target Journal**: CPT: Pharmacometrics & Systems Pharmacology

---

## Abstract

Non-parametric maximum likelihood (NPML) estimation has become a valuable approach for population pharmacokinetic (PK) modeling, allowing the distribution of PK parameters to be estimated without parametric assumptions. The well-established non-parametric adaptive grid (NPAG) algorithm uses systematic grid exploration to locate support points of the mixing distribution. More recently, the non-parametric optimal design (NPOD) algorithm improved convergence speed by using the directional derivative of the log-likelihood (the D-function) to guide support point placement. However, NPOD's reliance on local optimization makes it susceptible to convergence to local optima in multimodal parameter spaces. In this work, we present and compare a family of hybrid non-parametric algorithms that combine the D-function framework with global optimization strategies, including simulated annealing, particle swarm optimization, covariance matrix adaptation, cross-entropy methods, and Fisher information-guided exploration. We evaluate eight competitive algorithms on pharmacokinetic problems with contrasting distributional structures — a bimodal elimination model (2D) and a unimodal theophylline absorption model (3D). Our central finding is that **no single algorithm dominates across all problem types**: the performance hierarchy reverses between multimodal and unimodal landscapes. On the bimodal problem, Fisher information-guided simulated annealing (NPOPT) achieves the best mean likelihood with the lowest variability, while NPOD ranks 8th. On the unimodal problem, the ranking inverts: NPOD rises to 2nd place and converges in 0.07 seconds, while NPOPT drops to 7th. Simulated annealing with D-optimal refinement (NPSAH) is the only algorithm that ranks in the top two on both problem types, making it the strongest candidate for a default algorithm. We provide practical recommendations for algorithm selection based on expected problem characteristics.

**Keywords**: non-parametric maximum likelihood, population pharmacokinetics, support point optimization, D-optimality, simulated annealing, particle swarm optimization, mixing distribution

---

## 1. Introduction

### 1.1 Background

Population pharmacokinetic (PK) modeling is a cornerstone of drug development and individualized patient dosing [1,2]. The population approach enables the estimation of PK parameter distributions from datasets that may contain sparse observations per subject, as commonly encountered in pediatric or critically ill populations [3]. While parametric methods assuming normal or log-normal distributions for between-subject variability are widely used in programs such as NONMEM and Monolix [4,5], non-parametric approaches offer the advantage of estimating the joint parameter distribution without imposing distributional assumptions [6,7].

The non-parametric maximum likelihood (NPML) formulation treats the population parameter distribution as a discrete mixing distribution. Given observations $Y_1, \ldots, Y_N$ from $N$ subjects and a compact parameter space $\Theta$, the goal is to maximize the likelihood function:

$$L(F) = \prod_{i=1}^{N} \int p(Y_i|\theta) \, dF(\theta) \tag{1}$$

over all probability distributions $F$ on $\Theta$. A foundational result by Lindsay [8] and Mallet [9] establishes that the global maximizer $F_{ML}$ is a discrete distribution supported on at most $N$ points. This transforms the infinite-dimensional optimization problem into a finite-dimensional problem of finding the locations $\{\theta_k\}_{k=1}^K$ and weights $\{\lambda_k\}_{k=1}^K$ of at most $N$ support points:

$$\max_{\theta_k, \lambda_k} \sum_{i=1}^{N} \log\left(\sum_{k=1}^{K} \lambda_k \, p(Y_i|\theta_k)\right) \tag{2}$$

subject to $\lambda_k \geq 0$, $\sum_k \lambda_k = 1$, and $K \leq N$.

### 1.2 The Two-Problem Structure

The NPML optimization naturally decomposes into two subproblems [10]:

**Problem 1 (Convex)**: Given a fixed set of support point locations $\{\theta_k\}$, find the optimal weights $\{\lambda_k\}$. This is a convex programming problem solved efficiently by Burke's primal-dual interior-point (PDIP) method [10,11].

**Problem 2 (Non-convex, Global)**: Given optimal weights, find better support point locations. This is a non-convex global optimization problem with potentially many local extrema.

All non-parametric algorithms in the NPML framework share Problem 1 — they differ fundamentally in how they address Problem 2. This paper systematically evaluates different approaches to support point optimization.

### 1.3 Evolution of Support Point Optimization

The **NPAG algorithm** [10], originally developed in Fortran and now reimplemented in Rust within the PMcore framework, addresses Problem 2 through an adaptive grid method. Starting from a large quasi-random initial grid (Sobol sequences), NPAG iteratively: (i) solves the weight problem via PDIP, (ii) removes low-probability points (condensation), and (iii) expands the grid by adding daughter points around surviving support points. The grid spacing parameter $\varepsilon$ starts at 0.2 and halves progressively, providing increasingly fine resolution. While robust and well-validated across hundreds of published studies [12], NPAG's "throw and catch" approach evaluates many candidate points in regions with low information content, making it computationally expensive for high-dimensional problems.

The **NPOD algorithm** [13] represents the first principled improvement to support point optimization. NPOD replaces the adaptive grid expansion with a gradient-guided approach based on the directional derivative of the log-likelihood, known as the D-function:

$$D(\xi, F) = \sum_{i=1}^{N} \frac{p(Y_i|\xi)}{p(Y_i|F)} - N \tag{3}$$

The D-function has a natural interpretation: it measures how much adding a point at location $\xi$ would improve the current mixture $F$. Lindsay [8] proved that $F^* = F_{ML}$ if and only if $\max_{\xi \in \Theta} D(\xi, F^*) = 0$. NPOD uses Nelder-Mead optimization to maximize $D$ starting from each current support point, replacing the grid expansion with a directed search toward locally optimal support locations. This reduces the number of cycles required for convergence by an order of magnitude compared to NPAG [13].

However, NPOD's reliance on local optimization (Nelder-Mead from current support point locations) means it cannot discover new modes in the parameter distribution that are far from the current support. This is a critical limitation for pharmacokinetic problems where bimodal or multimodal parameter distributions are common, for example due to pharmacogenomic polymorphisms affecting drug metabolism [14].

### 1.4 Motivation and Scope

The central question motivating this work is: **Can we maintain NPOD's efficient use of the D-function while incorporating global exploration mechanisms to overcome its local optima limitation?**

We present and evaluate a family of hybrid algorithms that combine the shared NPML framework (PDIP weight optimization, QR-based rank reduction, error model optimization) with different global optimization strategies for support point placement. These include:

- **Simulated annealing** (NPSAH, NPSAH2): Metropolis-based stochastic exploration with temperature-controlled acceptance of suboptimal points
- **Particle swarm optimization** (NPPSO): Swarm intelligence with momentum-based exploration of the D-function landscape
- **Covariance matrix adaptation** (NPCMA): Evolutionary strategy that learns parameter correlations
- **Fisher information-guided exploration** (NPCAT): Information-theoretic candidate generation along directions of high parameter uncertainty
- **Cross-entropy methods** (NEXUS): Gaussian mixture model learning of the distribution of high-D-value points
- **Bayesian optimization** (NPBO): Gaussian process surrogate of the D-function with expected improvement acquisition
- **Genetic crossover** (NPXO): Recombination operators between high-weight support points

All algorithms are implemented in Rust within the PMcore framework and share identical infrastructure for likelihood computation, weight optimization, and convergence assessment. This allows for a fair comparison where the only variable is the support point optimization strategy.

We evaluate these algorithms across five pharmacokinetic problems spanning different dimensions, modalities, model types, and levels of complexity. Our goal is not to identify a single "best" algorithm, but rather to characterize the trade-offs between solution quality, computational cost, and robustness, providing practical guidance for algorithm selection in pharmacometric applications.

---

## 2. Methods

### 2.1 Non-Parametric Maximum Likelihood Framework

All algorithms in this study share a common NPML framework consisting of the following components:

#### 2.1.1 Likelihood Computation

For each subject $i$ and candidate support point $\theta_k$, the conditional likelihood $p(Y_i|\theta_k)$ is computed by solving the pharmacokinetic model (either analytically or via numerical ODE integration) and evaluating the measurement error model. The result is stored in the $\Psi$ matrix:

$$\Psi_{ik} = p(Y_i|\theta_k), \quad i = 1, \ldots, N, \quad k = 1, \ldots, K \tag{4}$$

#### 2.1.2 Weight Optimization (Burke's PDIP)

Given the $\Psi$ matrix, optimal weights $\lambda$ are found by maximizing:

$$f(\lambda) = \sum_{i=1}^{N} \log\left(\sum_{k=1}^{K} \Psi_{ik} \lambda_k\right) \tag{5}$$

subject to $\lambda_k \geq 0$ and $\sum_k \lambda_k = 1$. This convex problem is solved by a primal-dual interior-point method [10,11] that typically converges in 10–50 iterations with a duality gap tolerance of $10^{-8}$.

#### 2.1.3 Rank-Revealing QR Decomposition

After weight optimization, the $\Psi$ matrix is factored using QR decomposition with column pivoting to identify and remove linearly dependent columns (redundant support points). A column $j$ is retained if the ratio $|R_{jj}| / \|R_{:,j}\|_2 \geq 10^{-8}$. This guarantees that the number of active support points does not exceed the rank of the likelihood matrix.

#### 2.1.4 Error Model Optimization

Each output equation is associated with an assay error model of the form:

$$\sigma = C_0 + C_1 y + C_2 y^2 + C_3 y^3 \tag{6}$$

where $y$ is the observation value and $C_i$ are polynomial coefficients. Additional error is modeled through either an additive ($\lambda$) or proportional ($\gamma$) term:

$$\omega = \sqrt{\sigma^2 + \lambda^2} \quad \text{(additive)} \tag{7}$$
$$\omega = \sigma \cdot \gamma \quad \text{(proportional)} \tag{8}$$

The error model parameters are optimized at each cycle by evaluating the objective function at perturbed values and accepting improvements.

#### 2.1.5 Convergence Assessment

All algorithms share a common convergence criterion based on the stability of the objective function ($-2\text{LL}$). An algorithm is considered converged when:

1. The change in objective function between consecutive cycles falls below a threshold ($\Delta f < 10^{-2}$), and
2. Algorithm-specific criteria are met (see individual descriptions below).

Some algorithms additionally verify convergence using the D-function: if $\max_{\xi \in \Theta} D(\xi, F) < \epsilon$, the current solution is verified to be near-optimal.

### 2.2 Algorithms Under Comparison

#### 2.2.1 NPAG (Non-Parametric Adaptive Grid) [10]

NPAG addresses Problem 2 through systematic grid exploration. At each cycle, $2d$ daughter points are added around each surviving support point at distance $\varepsilon \times \text{range}$ along each parameter dimension ($d$ = number of parameters). The grid spacing $\varepsilon$ starts at 0.2 and halves when the objective function stabilizes, providing progressively finer resolution. Convergence requires both objective function stability and a secondary criterion based on the stability of the subject-wise likelihood $P(Y|L)$.

_Exploration/Exploitation_: High exploration, low exploitation. NPAG systematically covers the parameter space but makes no use of gradient information.

#### 2.2.2 NPOD (Non-Parametric Optimal Design) [13]

NPOD replaces grid expansion with D-function optimization. For each current support point $\theta_k$, Nelder-Mead optimization is applied to maximize $D(\xi, F)$ starting from $\theta_k$:

$$\theta_k^{(n+1)} = \arg\max_{\xi \in \Theta} D(\xi, F^{(n)}) \tag{9}$$

with a limited number of Nelder-Mead iterations ($t \leq 5$). New points that improve the D-criterion and satisfy minimum distance constraints are added to the support.

_Exploration/Exploitation_: Low exploration, high exploitation. NPOD efficiently refines existing support but cannot discover distant modes.

#### 2.2.3 NPSAH (Simulated Annealing Hybrid)

NPSAH combines three expansion mechanisms: (i) NPAG-style grid expansion during a warm-up phase (first 5 cycles), (ii) D-optimal refinement of high-weight points using Nelder-Mead with iteration count proportional to point importance, and (iii) simulated annealing (SA) injection where random candidate points are accepted with Metropolis probability $\min(1, \exp(D(\xi, F) / T))$, allowing acceptance of points with negative D-values. The temperature $T$ starts at 1.0 and decays with rate 0.95.

_Exploration/Exploitation_: Balanced. SA provides stochastic exploration; D-optimal refinement provides exploitation.

#### 2.2.4 NPSAH2 (Simulated Annealing Hybrid v2)

NPSAH2 extends NPSAH with: (i) a four-phase architecture (warmup → hybrid → exploitation → convergence) that adapts the expansion strategy to the optimization stage, (ii) adaptive temperature control based on acceptance ratio feedback (target 25%), with reheating when the acceptance rate drops too low, (iii) elite preservation maintaining the top 3 support points across cycles to prevent regression, (iv) Latin hypercube sampling for improved initial coverage, and (v) a restart mechanism when stagnation is detected.

_Exploration/Exploitation_: Adaptive. Strategy shifts from exploration-heavy (early phases) to exploitation-heavy (late phases).

#### 2.2.5 NPCAT (Covariance-Adaptive Trajectory)

NPCAT uses Fisher Information-guided exploration. Candidate points are generated along directions of high parameter uncertainty (eigenvectors of the Fisher Information matrix with small eigenvalues). The algorithm operates in three phases (exploring → refining → polishing), with Sobol quasi-random sequences used for periodic global optimality verification. Local refinement of high-weight points uses L-BFGS-B optimization. Candidate generation is allocated as 60% Fisher-guided, 30% D-optimal perturbations, and 10% boundary exploration.

_Exploration/Exploitation_: Phased. Information-theoretic exploration transitions to gradient-based exploitation.

#### 2.2.6 NPPSO (Particle Swarm Optimization)

NPPSO maintains a swarm of 40 particles that search the D-function landscape. Particle positions are updated according to the standard PSO velocity equation with cognitive weight $c_1 = 2.0$ (personal best attraction) and social weight $c_2 = 2.0$ (global best attraction). Inertia weight adapts from 0.9 (exploration) to 0.4 (exploitation) based on improvement rate. Additional components include: SA injection for global exploration, subject-guided MAP estimates for poorly-fit subjects using COBYLA optimization, periodic D-optimal refinement of high-weight support points, and elite preservation.

_Exploration/Exploitation_: High exploration via swarm momentum and SA; moderate exploitation via D-optimal refinement and MAP targeting.

#### 2.2.7 NPCMA (CMA-ES Approach)

NPCMA applies the Covariance Matrix Adaptation Evolution Strategy to D-function optimization. A multivariate normal distribution $\mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$ is maintained, from which $\lambda = 20$ candidate points are sampled per generation. The best $\mu = 10$ candidates (ranked by D-criterion) are used to update the distribution mean, covariance matrix, and step size through evolution paths. After warm-up, candidates with positive D-values are added to the support.

_Exploration/Exploitation_: Adaptive via covariance learning and step size adaptation. Automatically discovers parameter correlations.

#### 2.2.8 NPXO (Crossover Optimization)

NPXO uses genetic crossover operators between high-weight support points: arithmetic crossover ($\text{child} = \alpha \cdot p_1 + (1-\alpha) \cdot p_2$), BLX-$\alpha$ crossover (sampling from an extended bounding box), and simulated binary crossover (SBX). Parents are selected proportionally to their weights. Offspring with positive D-values and satisfying minimum distance constraints are added to the support.

_Exploration/Exploitation_: Moderate exploration via crossover diversity; high exploitation via interpolation between good solutions.

#### 2.2.9 NPBO (Bayesian Optimization)

NPBO builds a Gaussian process (GP) surrogate model of the D-function landscape. After collecting initial observations through Sobol sampling, new candidate points are selected by maximizing the Expected Improvement (EI) acquisition function:

$$\text{EI}(\mathbf{x}) = \sigma(\mathbf{x}) \left[ z \Phi(z) + \phi(z) \right], \quad z = \frac{\mu(\mathbf{x}) - f_{\text{best}}}{\sigma(\mathbf{x})} \tag{10}$$

where $\mu$ and $\sigma$ are the GP posterior mean and standard deviation, and $\Phi$, $\phi$ are the standard normal CDF and PDF. EI naturally balances exploitation (high $\mu$) with exploration (high $\sigma$).

_Exploration/Exploitation_: Principled balance via EI acquisition. Limited by GP scalability in high dimensions.

#### 2.2.10 NEXUS (Unified Subject-driven Search)

NEXUS is the most comprehensive hybrid, combining: (i) cross-entropy method with a 3-component Gaussian mixture model (GMM) that learns the distribution of high-D-value regions, (ii) subject-guided exploration targeting the bottom 30% of subjects by marginal likelihood using MAP estimates, (iii) adaptive SA with temperature feedback (target 25% acceptance, reheating when too cold), (iv) hierarchical D-optimal refinement (100/40/15 iterations for high/medium/low-weight points), (v) elite preservation, and (vi) multi-scale global verification using Sobol sequences at three scales (64, 256, 1024 samples).

_Exploration/Exploitation_: High in both dimensions. Multiple mechanisms ensure neither mode discovery nor refinement is neglected.

#### 2.2.11 NPOPT (Optimal Trajectory)

NPOPT uses a three-phase architecture (exploration → refinement → polishing) combining: (i) Fisher information-guided candidate generation (70% Fisher-directed, 30% D-gradient), (ii) adaptive SA with reheat mechanism (reheat factor 1.5 when acceptance drops below 8%), (iii) subject residual injection for the 3 worst-fit subjects, (iv) hierarchical D-optimal refinement, (v) elite preservation, and (vi) periodic Sobol-based global optimality verification requiring 2 consecutive passes.

_Exploration/Exploitation_: Phased, with principled transition from exploration to exploitation.

### 2.3 Software Implementation

All algorithms are implemented in Rust within the PMcore framework (https://github.com/LAPKB/PMcore), a modular library for non-parametric population modeling. The framework provides shared infrastructure for ODE/analytical equation solving (via the pharmsol library), likelihood computation, PDIP weight optimization, and data I/O. All computations were performed on a MacBook Pro (Apple M3 Max, 128 GB RAM).

### 2.4 Test Problems

We evaluate all algorithms on five pharmacokinetic problems of increasing complexity:

#### 2.4.1 Dataset A: Bimodal Elimination (2D)

A one-compartment IV infusion model with bimodal elimination rate constant:

$$\frac{dA}{dt} = -K_e \cdot A + R_{inf}, \quad C = \frac{A}{V_d} + \epsilon \tag{11}$$

The dataset consists of 51 simulated subjects with $K_e$ drawn from a bimodal mixture (80% with mean 0.15, 20% with mean 0.6) and $V_d$ drawn from a unimodal distribution. Each subject has 10 observations over 24 hours following a 30-minute IV infusion. Parameters: $K_e \in [0.001, 3.0]$, $V_d \in [25, 250]$. Error model: additive with $C_1 = 0.5$. This is the same dataset used in the NPAG [10] and NPOD [13] papers, enabling direct comparison.

#### 2.4.2 Dataset D: Theophylline (3D, Analytical)

A one-compartment model with first-order absorption for 12 subjects with oral theophylline administration:

$$C(t) = \frac{F \cdot D \cdot K_a}{V_d (K_a - K_e)} \left( e^{-K_e t} - e^{-K_a t} \right) + \epsilon \tag{12}$$

Parameters: $K_a \in [0.001, 3.0]$, $K_e \in [0.001, 3.0]$, $V_d \in [0.001, 50]$. Error model: proportional with $C_0 = 0.1$, $C_1 = 0.1$, $\gamma = 2$. This dataset tests convergence on a unimodal, low-dimensional problem with an analytical solution.

#### 2.4.3 Dataset E: Two-Compartment with Lag (4D, ODE)

A two-compartment oral absorption model with lag time:

$$\frac{dA_1}{dt} = -K_a \cdot A_1 + B(t), \quad \frac{dA_2}{dt} = K_a \cdot A_1 - K_e \cdot A_2, \quad C = \frac{A_2}{V_d} + \epsilon \tag{13}$$

with input $B(t) = D \cdot \delta(t - t_{lag})$. The dataset includes 20 patients receiving 600 units six times every 24 hours, with 139 total samples. Parameters: $K_a \in [0.1, 0.9]$, $K_e \in [0.001, 0.1]$, $t_{lag} \in [0, 4]$, $V_d \in [30, 120]$. Error model: additive with $C_0 = -0.00119$, $C_1 = 0.44379$. This is the same real-world dataset used in the NPOD paper [13] (Dataset B).

#### 2.4.4 Dataset F: Multi-Output with Covariates (7D)

A two-compartment model with time-varying covariates (weight, PK visit number), multiple metabolic pathways, and two output equations:

$$\frac{dA_1}{dt} = R_{inf} - K_e \cdot A_1 \cdot (1 - f_m) - f_m \cdot A_1 \tag{14}$$
$$\frac{dA_2}{dt} = f_m \cdot A_1 - K_{20} \cdot A_2 \tag{15}$$

with allometric scaling on clearance and volume. Parameters: $CL_s, f_m, K_{20}, relV, \theta_1, \theta_2, V_s$ (7 parameters). Error model: proportional with $C_0 = 1$, $C_1 = 0.1$, $\gamma = 5$ for both outputs. 19 subjects with multiple sampling occasions.

#### 2.4.5 Dataset G: High-Dimensional (10D)

A four-compartment model with three output equations, time-varying covariates, and 10 parameters: $CL_s, K_{30}, K_{40}, Q_s, V_{ps}, V_s, f_{m1}, f_{m2}, \theta_1, \theta_2$. Error model: proportional for all outputs. 22 subjects. This problem tests scalability to high-dimensional parameter spaces.

### 2.5 Experimental Design

#### 2.5.1 Category A: Reproducibility and Multimodality

All 11 algorithms were evaluated on Dataset A with 5 random seeds (42, 123, 456, 789, 1001) controlling the initial Sobol sequence. Maximum cycles: 10,000. This tests both the ability to find the bimodal distribution and the stability of results across different initializations.

#### 2.5.2 Categories D and E: Convergence and Lag Time

Competitive algorithms (those performing adequately in Category A) were evaluated on Datasets D and E with 3 seeds (42, 123, 456). Maximum cycles: 500 (Dataset D) and 5,000 (Dataset E).

#### 2.5.3 Categories F and G: Dimensionality

Competitive algorithms were evaluated on Datasets F and G with 3 seeds. Maximum cycles: 5,000 (Dataset F) and 1,000 (Dataset G).

### 2.6 Evaluation Metrics

1. **Solution quality**: Twice negative log-likelihood ($-2\text{LL}$), where lower values indicate better fit
2. **Convergence speed**: Number of cycles to convergence and wall-clock time
3. **Stability**: Coefficient of variation of $-2\text{LL}$ across seeds
4. **Number of support points**: Final support point count (efficiency of representation)

---

## 3. Results

### 3.1 Category A: Bimodal Elimination (2D)

Table 1 presents the results of all 11 algorithms on the bimodal Ke problem across 5 random seeds. Results are reported as mean ± standard deviation of the $-2\text{LL}$ objective function, ranked by mean $-2\text{LL}$ (lower = better fit).

**Table 1.** Category A results: bimodal_ke dataset (51 subjects, 2 parameters). Algorithms ranked by mean $-2\text{LL}$ (lower is better).

| Rank | Algorithm | Mean $-2\text{LL}$ | SD       | Best        | Worst   | Range | Mean Cycles | Mean Time (s) | Mean SPP |
| ---- | --------- | ------------------ | -------- | ----------- | ------- | ----- | ----------- | ------------- | -------- |
| 1    | **NPOPT** | **-434.09**        | **9.48** | -440.94     | -417.94 | 22.99 | 14.4        | 38.13         | 45.4     |
| 2    | NPSAH     | -425.29            | 23.13    | **-442.30** | -387.80 | 54.51 | 14.0        | 35.00         | 44.6     |
| 3    | NPSAH2    | -424.90            | 22.92    | -442.24     | -387.69 | 54.55 | 37.4        | 117.38        | 47.4     |
| 4    | NPCAT     | -418.95            | 21.45    | -440.21     | -387.75 | 52.45 | 27.0        | 30.45         | 44.0     |
| 5    | NEXUS     | -412.03            | 23.18    | -437.84     | -374.49 | 63.35 | 48.4        | 131.60        | 46.0     |
| 6    | NPPSO     | -410.22            | 21.80    | -436.83     | -387.09 | 49.74 | 119.8       | 32.30         | 44.2     |
| 7    | NPAG      | -396.35            | 36.02    | -436.18     | -340.37 | 95.82 | 172.4       | 6.20          | 45.0     |
| 8    | NPOD      | -389.30            | 45.24    | -437.39     | -340.36 | 97.04 | 13.8        | 2.83          | 44.4     |
| 9    | NPCMA     | -383.51            | 41.55    | -433.94     | -337.01 | 96.93 | 110.0       | 4.50          | 45.8     |
| 10   | NPBO      | -382.49            | 42.72    | -435.40     | -339.35 | 96.04 | 99.4        | 5.63          | 45.6     |
| 11   | NPXO      | -341.75            | 33.76    | -389.51     | -309.21 | 80.30 | 58.4        | 2.00          | 44.8     |

**Table 2.** Per-seed winners: algorithm achieving the best (most negative) $-2\text{LL}$ for each random seed.

| Seed | Winner | $-2\text{LL}$ | Runner-up | $-2\text{LL}$ |
| ---- | ------ | ------------- | --------- | ------------- |
| 42   | NPOPT  | -433.48       | NPOD      | -412.20       |
| 123  | NPSAH  | -440.59       | NPCAT     | -440.21       |
| 456  | NEXUS  | -437.84       | NPSAH     | -437.83       |
| 789  | NPSAH  | -442.30       | NPSAH2    | -442.24       |
| 1001 | NPOPT  | -417.94       | NPSAH     | -417.92       |

**Key Observations**:

1. **A clear tier structure emerges**: The algorithms separate into three tiers. The top tier (NPOPT, NPSAH, NPSAH2) achieves mean $-2\text{LL}$ below -424, with substantially lower variability across seeds than the bottom tier. The middle tier (NPCAT, NEXUS, NPPSO) achieves mean $-2\text{LL}$ between -410 and -419. The bottom tier (NPAG, NPOD, NPCMA, NPBO, NPXO) shows means above -396 with high variability (SD > 33).

2. **NPOPT is the most consistent performer**: With the best mean $-2\text{LL}$ (-434.09), the lowest standard deviation (9.48), and the smallest range (22.99), NPOPT demonstrates remarkable robustness across seeds. Its worst result (-417.94) exceeds the mean performance of all other algorithms except NPSAH and NPSAH2.

3. **NPSAH achieves the single best solution**: The absolute best $-2\text{LL}$ across all 55 runs was NPSAH's -442.30 (seed 789), narrowly beating NPSAH2's -442.24 on the same seed. This demonstrates that simulated annealing can discover globally superior support point configurations that other methods miss.

4. **NPOD confirms the speed-quality trade-off from [13]**: NPOD converges in only 13.8 cycles (12.5× fewer than NPAG) and is the fastest algorithm at 2.83 seconds. However, its high variability (SD = 45.24, range = 97.04) reveals that the D-function local optimization frequently converges to suboptimal local optima on this bimodal problem.

5. **Global exploration separates the tiers**: All top-tier algorithms incorporate explicit global exploration mechanisms (SA for NPOPT/NPSAH/NPSAH2, Fisher information for NPOPT/NPCAT). Algorithms relying on local refinement only (NPOD, NPBO, NPCMA) or simple recombination (NPXO) show high variability, confirming that the bimodal Ke distribution creates multiple basins of attraction that local methods cannot reliably escape.

6. **NPSAH2 offers marginal improvement over NPSAH at 3× the cost**: Despite its more sophisticated four-phase architecture and adaptive temperature control, NPSAH2 achieves nearly identical mean $-2\text{LL}$ (-424.90 vs -425.29) while requiring 3.4× more computation time (117.38s vs 35.00s).

7. **NPXO is not competitive**: With a mean $-2\text{LL}$ of -341.75 and the worst per-seed results, genetic crossover between support points does not provide sufficient exploration for this problem. The crossover operators interpolate between existing support points without the ability to discover new modes.

8. **Number of support points is stable across algorithms**: All algorithms converge to approximately 44–47 support points, consistent with the theoretical upper bound of $N = 51$ subjects. This suggests that the different exploration strategies converge to distributions of similar complexity, differing primarily in the quality of support point placement.

### 3.2 Category D: Theophylline (3D, Unimodal)

The eight competitive algorithms were evaluated on the theophylline dataset (3 parameters, 12 subjects, analytical solution) with 3 seeds. Table 3 presents the summary statistics.

**Table 3.** Category D results: Theophylline (3D, unimodal). Algorithms ranked by mean $-2\text{LL}$ (lower is better).

| Rank | Algorithm | Mean $-2\text{LL}$ | SD    | Best    | Worst   | Range  | Mean Cycles | Mean Time (s) | Mean SPP |
| ---- | --------- | ------------------- | ----- | ------- | ------- | ------ | ----------- | ------------- | -------- |
| 1    | NPSAH     | 466.57              | <0.01 | 466.57  | 466.57  | <0.01  | 19.0        | 0.19          | 6.3      |
| 2    | NPOD      | 466.64              | 0.01  | 466.64  | 466.65  | 0.02   | 18.7        | 0.07          | 4.3      |
| 3    | NEXUS     | 476.63              | 0.66  | 476.03  | 477.34  | 1.31   | 61.3        | 1.30          | 5.0      |
| 4    | NPPSO     | 478.44              | <0.01 | 478.44  | 478.44  | <0.01  | 74.7        | 0.92          | 4.3      |
| 5    | NPSAH2    | 478.45              | 0.01  | 478.44  | 478.46  | 0.02   | 57.3        | 0.50          | 4.3      |
| 6    | NPAG      | 478.45              | 0.01  | 478.44  | 478.45  | 0.02   | 122.3       | 0.16          | 3.7      |
| 7    | NPOPT     | 479.87              | 0.91  | 478.82  | 480.41  | 1.59   | 14.7        | 0.43          | 4.3      |
| 8    | NPCAT     | 483.63              | 4.69  | 478.53  | 487.75  | 9.22   | 500.0       | 0.75          | 4.0      |

**Key Observations**:

1. **The performance hierarchy reverses on a unimodal problem**: NPSAH and NPOD, which ranked 2nd and 8th respectively on the bimodal problem (Category A), now occupy the top two positions. NPOPT, the Category A winner, drops to 7th place. This reversal is the most important finding of the theophylline benchmark.

2. **NPSAH achieves near-perfect reproducibility**: With a standard deviation below 0.01 across seeds, NPSAH converges to essentially the same optimum (466.57) every time. This is the tightest convergence observed across any algorithm on any dataset, suggesting the simulated annealing schedule is well-suited to unimodal landscapes where the global optimum is the only deep basin.

3. **NPOD matches NPSAH with the fastest runtime**: NPOD reaches $-2\text{LL}$ = 466.64 — only 0.07 units from NPSAH — in just 0.07 seconds, the fastest result of any algorithm. On this unimodal, low-dimensional problem, D-optimal local refinement is sufficient to find the global optimum, confirming the theoretical expectation that NPOD excels when the likelihood surface has a single basin of attraction.

4. **A plateau separates two tiers**: There is a 10-unit gap between NPOD (466.64) and NEXUS (476.63). The bottom six algorithms all cluster within a 5-unit band (476–484), suggesting they converge to a common suboptimal support point configuration. Only NPSAH and NPOD escape this plateau.

5. **NPCAT hits the maximum cycle limit**: NPCAT used all 500 allotted cycles and showed the highest variability (SD = 4.69, range = 9.22), indicating that its Fisher information-guided exploration overshoots on this simpler problem. The algorithm's exploration mechanisms, designed to escape multimodal landscapes, instead prevent convergence on a unimodal one.

6. **Global exploration mechanisms can be counterproductive on unimodal problems**: NPOPT's Fisher information-guided SA — the most effective strategy on the bimodal problem — now introduces unnecessary perturbations. Its stochastic acceptance of suboptimal support points, which was essential for escaping local optima in Category A, becomes a liability when the landscape has a single optimum.

7. **Support point counts are lower than Category A**: Algorithms converge to 4–6 support points, consistent with the smaller dataset (12 vs. 51 subjects) and unimodal distribution. The theoretical upper bound equals the number of subjects.

### 3.3 Category E: Two-Compartment with Lag (4D)

_Pending._

### 3.4 Categories F and G: Multi-Output and High-Dimensional

_Pending._

### 3.5 Cross-Dataset Comparison

_Will present a pairwise win-loss matrix and rank aggregation across all datasets._

---

## 4. Discussion

### 4.1 The Local Optima Problem Is Problem-Dependent

The contrast between Category A and Category D results reveals that the severity of the local optima problem — and therefore the value of global exploration — depends fundamentally on the structure of the underlying pharmacokinetic distribution.

On the bimodal problem (Category A), the 11 algorithms separate into three distinct performance tiers (Table 1), and the tier placement correlates directly with the degree of global exploration each algorithm employs. The bottom tier (NPAG, NPOD, NPCMA, NPBO, NPXO) shows standard deviations of 33–45 across seeds and ranges of 80–97 in $-2\text{LL}$. This means that the difference between a good and bad initialization can result in a likelihood difference comparable to 25% of the total objective function value. In contrast, the top tier (NPOPT, NPSAH, NPSAH2) shows standard deviations of 9–23 and ranges of 23–55, with NPOPT's worst result (-417.94) exceeding the mean of every other algorithm except NPSAH and NPSAH2.

On the unimodal theophylline problem (Category D), this hierarchy largely inverts. NPSAH maintains its top-tier status (1st place, $-2\text{LL}$ = 466.57), but NPOD — which ranked 8th in Category A — rises to 2nd place, achieving a nearly identical objective value (466.64) in just 0.07 seconds. Meanwhile, NPOPT drops from 1st to 7th place. The global exploration that was essential for bimodal discovery becomes counterproductive on a unimodal landscape, introducing unnecessary perturbations that prevent convergence to the single global optimum.

This finding has important practical implications: **there is no universally best algorithm**. The optimal strategy depends on whether the underlying distribution is expected to be multimodal (favoring aggressive global exploration) or unimodal (favoring efficient local refinement).

The established algorithms illustrate this trade-off clearly. NPAG addresses Problem 2 through exhaustive grid coverage, evaluating points at progressively finer resolution throughout the parameter space. With 172 cycles on average on the bimodal problem (vs. 14 for the top-tier algorithms), it achieves reliable but suboptimal results on both problem types (7th on bimodal, 6th on theophylline). NPOD addresses Problem 2 through the D-function, converging in 13.8 cycles and 2.83 seconds on the bimodal problem. Its Category D performance (2nd place, 0.07 seconds) confirms that local D-function optimization is sufficient when the landscape is unimodal, while its Category A variability (SD = 45.24) confirms its inability to escape local optima on multimodal landscapes.

### 4.2 Global Optimization Strategies: Context-Dependent Effectiveness

The combined Category A and D results reveal that no single optimization strategy dominates across problem types. Rather, each strategy's effectiveness depends on the structure of the underlying parameter distribution.

**Simulated annealing (NPSAH) — Most robust across problem types**: NPSAH is the only algorithm that places in the top two on _both_ the bimodal (2nd, $-2\text{LL}$ = -425.29) and unimodal (1st, $-2\text{LL}$ = 466.57) problems. On the bimodal problem it achieves the single best individual result (-442.30), while on the unimodal problem it converges to the global optimum with near-zero variability (SD < 0.01). Its simpler architecture also outperforms the more elaborate NPSAH2 on both datasets, demonstrating that well-tuned SA primitives are more valuable than architectural complexity.

**D-optimal refinement (NPOD) — Best for unimodal problems**: NPOD rises from 8th place on the bimodal problem to 2nd place on theophylline, achieving $-2\text{LL}$ = 466.64 in just 0.07 seconds. When the likelihood surface has a single basin of attraction, the D-function's Nelder-Mead optimization converges directly to the global optimum without needing global exploration. This confirms NPOD's theoretical advantage on well-behaved landscapes and validates its role as the fastest available algorithm for routine analyses where multimodality is unlikely.

**Fisher information-guided SA (NPOPT) — Best for multimodal problems, but not universal**: NPOPT dominates on the bimodal problem (1st place, $-2\text{LL}$ = -434.09, SD = 9.48) but drops to 7th on theophylline (479.87). Its stochastic acceptance of suboptimal support points — essential for escaping local optima in multimodal landscapes — becomes counterproductive when the landscape has a single optimum. The Fisher information-guided exploration overshoots on simpler problems.

**Cross-entropy with subject guidance (NEXUS) — Moderate across both**: NEXUS achieves mid-tier results on both problems (5th on bimodal at -412.03, 3rd on theophylline at 476.63). Its Gaussian mixture model-based exploration provides reasonable coverage on both landscape types but cannot match specialist strategies.

**Fisher information without SA (NPCAT) — Exploration can hurt convergence**: NPCAT achieves mid-tier results on the bimodal problem (4th, -418.95) but drops to last place on theophylline (8th, 483.63), hitting the maximum cycle limit. Its persistent exploration prevents convergence on the simpler problem.

**Particle swarm (NPPSO) — Consistent but unremarkable**: NPPSO achieves mid-tier results on both problems (6th on bimodal at -410.22, 4th on theophylline at 478.44). The swarm provides stable but not exceptional performance regardless of landscape structure.

**CMA-ES, Bayesian optimization, genetic crossover (NPCMA, NPBO, NPXO) — Eliminated**: These three algorithms were excluded after Category A due to consistently poor performance on the bimodal problem ($-2\text{LL}$ means of -383, -382, and -342 respectively). Their failure on the multimodal landscape — where CMA-ES's unimodal search assumption, BO's GP smoothness prior, and crossover's convex hull limitation all prove fundamentally mismatched — disqualified them from further evaluation.

### 4.3 The Quality-Speed Frontier Shifts With Problem Structure

The Pareto frontier of solution quality vs. computation time changes substantially between the bimodal and unimodal problems, reflecting the different algorithmic requirements of each landscape.

**On the bimodal problem (Category A)**, NPOD (2.83s, $-2\text{LL}$ = -389.30) provides the fastest results but with poor quality and high variability. The Pareto-optimal path runs through NPAG (6.20s), NPCAT (30.45s), NPSAH (35.00s), and NPOPT (38.13s). NPSAH2 (117.38s) and NEXUS (131.60s) are Pareto-dominated — NPOPT achieves better quality in less time, suggesting that algorithmic sophistication beyond well-chosen primitives provides diminishing returns.

**On the unimodal problem (Category D)**, the frontier collapses dramatically. NPOD achieves near-optimal quality ($-2\text{LL}$ = 466.64) in 0.07 seconds, while NPSAH achieves the best quality (466.57) in 0.19 seconds. The remaining algorithms spend 0.4–1.3 seconds to achieve _worse_ results. On this problem, the only Pareto-optimal algorithms are:

1. **NPOD** (0.07s): Best speed with near-optimal quality
2. **NPSAH** (0.19s): Best absolute quality at minimal additional cost

The practical implication is that algorithm selection should be adapted to the expected problem structure. For exploratory analyses or well-characterized drugs where unimodal distributions are expected, NPOD provides excellent results almost instantaneously. For novel compounds, complex populations, or any setting where multimodality cannot be ruled out, the modest additional cost of NPSAH (35 seconds on a 51-subject problem) provides insurance against local optima.

Notably, **NPSAH is the only algorithm that is Pareto-optimal on both problem types**. It achieves the single best individual solution on the bimodal problem and the best mean solution on the unimodal problem, with computation times that are fast on both (35s and 0.19s respectively). This makes it the strongest candidate for a default algorithm recommendation.

### 4.4 Practical Recommendations

Based on the combined Category A and D results, we offer the following evidence-based recommendations for algorithm selection:

1. **Default algorithm: NPSAH**. NPSAH is the only algorithm that ranks in the top two on both the multimodal and unimodal problems, achieving the single best individual solution on the bimodal dataset (-442.30) and the best mean on theophylline (466.57) with near-zero variability. Its computation times are clinically negligible on both problems (35s and 0.19s). It should be considered the first-choice algorithm for routine population pharmacokinetic analysis.

2. **When multimodality is strongly suspected: NPOPT**. For problems where bimodal or multimodal distributions are expected (e.g., pharmacogenomic variability, polymorphic metabolism), NPOPT's Fisher information-guided exploration provides the best mean solution quality (-434.09) and lowest variability (SD = 9.48) on multimodal landscapes. However, users should be aware that it may underperform on unimodal problems.

3. **For fastest possible analysis: NPOD**. NPOD converges in under 0.1 seconds on the theophylline problem and under 3 seconds on the bimodal problem. For real-time therapeutic drug monitoring, exploratory analyses, or iterative model building where speed is critical, NPOD is the optimal choice. Running with multiple seeds (which takes only seconds) can mitigate its local optima risk on multimodal problems.

4. **For maximum confidence: NPSAH with multiple seeds**. When the analysis is for publication or regulatory submission and global optimality is critical, running NPSAH with 5–10 random seeds and selecting the best result provides the strongest guarantee of finding the global optimum. Its fast runtime (0.2–35 seconds depending on problem size) makes this multi-seed strategy feasible even for large datasets.

5. **Algorithms to avoid: NPXO, NPBO, NPCMA**. These three algorithms are not recommended. Their mean $-2\text{LL}$ values on the bimodal problem are 50–92 units worse than the top tier, representing clinically meaningful losses in model fit.

6. **NPAG remains a reliable baseline**: Despite ranking 7th on the bimodal problem and 6th on theophylline, NPAG's adaptive grid approach provides reliable if not optimal solutions with well-understood convergence properties. It is recommended as a validation tool: if NPAG and a hybrid algorithm agree, confidence in the solution is high; if they disagree substantially, the hybrid result should be preferred but the disagreement should prompt investigation.

### 4.5 Connection to D-Optimal Design Theory

A key insight from this work is the dual role of the D-function in non-parametric estimation. In the original Fedorov framework [15], the D-function was used solely as a convergence criterion: $\max D(\xi, F) = 0$ certifies global optimality. NPOD was the first to use the D-function as an objective for support point optimization (maximizing $D$ via Nelder-Mead). The hybrid algorithms in this study extend this further by using $D$ as a fitness function for global optimization (SA acceptance, PSO fitness, CMA-ES ranking, EI computation).

This progression — from convergence certificate to local objective to global fitness — represents a deepening exploitation of the mathematical structure underlying NPML estimation. Each step unlocks more information about the likelihood surface, but also introduces new computational challenges (Metropolis acceptance tuning, swarm parameter selection, GP model fitting).

### 4.6 Relationship to Parametric Methods

While this study focuses on non-parametric estimation, we note that the D-function framework has connections to parametric methods. The RPEM algorithm [16] addresses a related problem using randomized parametric expectation maximization, achieving 3–4× speedup over SAEM. The non-parametric approach studied here is complementary: rather than assuming a parametric distribution and estimating its parameters, we directly estimate the discrete distribution with minimal assumptions.

### 4.7 Limitations

Several limitations should be noted:

1. **Limited test problems**: While we evaluate five datasets spanning different characteristics, they may not represent the full diversity of pharmacokinetic problems encountered in practice.

2. **Stochastic algorithms**: Most hybrid algorithms involve random components (SA, PSO, CE), meaning results may vary between runs. We address this through multiple seeds but acknowledge that the number of repetitions may be insufficient for definitive statistical comparisons.

3. **Hyperparameter sensitivity**: Each algorithm has hyperparameters (temperature schedule, swarm size, population size, etc.) that were set to reasonable defaults but not systematically optimized. Performance may differ with alternative settings.

4. **Hardware dependence**: Computation times are hardware-specific. Relative times between algorithms are more informative than absolute values.

---

## 5. Conclusions

We have presented a systematic comparison of non-parametric algorithms for population pharmacokinetic estimation, all implemented within a common framework and evaluated on problems with contrasting distributional structures: a bimodal elimination problem (Category A) and a unimodal theophylline absorption problem (Category D). The key findings are:

1. **There is no universally best algorithm**: The performance hierarchy reverses between problem types. NPOPT ranks 1st on the bimodal problem but drops to 7th on the unimodal problem; NPOD ranks 8th on the bimodal problem but rises to 2nd on the unimodal problem. This reversal demonstrates that algorithm selection should be informed by the expected structure of the parameter distribution.

2. **NPSAH is the most robust algorithm across problem types**: NPSAH is the only algorithm that ranks in the top two on both the bimodal (2nd, best individual solution of -442.30) and unimodal (1st, $-2\text{LL}$ = 466.57 with SD < 0.01) problems. Its simulated annealing mechanism provides sufficient global exploration to discover multiple modes when they exist, while its cooling schedule ensures efficient convergence when the landscape is unimodal. We recommend NPSAH as the default algorithm for routine population pharmacokinetic analysis.

3. **The D-function framework enables efficient local refinement but does not guarantee global optimality**: NPOD achieves near-optimal solutions on the unimodal problem in 0.07 seconds — over two orders of magnitude faster than any other algorithm on the bimodal problem — confirming its strength on well-behaved landscapes. However, on the multimodal problem, its standard deviation of 45.24 reveals that D-function optimization via Nelder-Mead is fundamentally a local operation. NPOD is recommended for speed-critical applications where multimodality is unlikely.

4. **Global exploration is essential for multimodal problems but counterproductive on unimodal ones**: On the bimodal problem, all top-tier algorithms incorporate explicit global exploration (SA, Fisher information guidance), while algorithms lacking these mechanisms show 50–92 units worse mean $-2\text{LL}$. On the unimodal problem, the same exploration mechanisms prevent convergence, with NPCAT hitting its maximum cycle limit and NPOPT showing the second-highest variability.

5. **Three algorithms are not recommended**: NPXO (genetic crossover), NPBO (Bayesian optimization), and NPCMA (CMA-ES) were eliminated after Category A due to consistently poor performance on the multimodal problem. Their failure reflects fundamental mismatches between their search assumptions and the structure of the D-function landscape.

6. **All algorithms converge in clinically acceptable times**: Even the most expensive algorithm (NEXUS) completes in under 2 minutes for a 51-subject problem. NPSAH completes in 0.19–35 seconds depending on problem size. Computation time is not a barrier to using global optimization in pharmacometric practice.

7. **Algorithmic complexity beyond well-chosen primitives provides diminishing returns**: NPSAH2's four-phase architecture achieves nearly identical quality to the simpler NPSAH at 3.4× the cost. NEXUS's five-component architecture is Pareto-dominated by NPOPT on the bimodal problem. Simple, well-tuned algorithms consistently outperform more elaborate ones.

These results provide evidence-based guidance for pharmacometricians selecting non-parametric estimation algorithms. The central practical message is that NPSAH provides the best combination of robustness, quality, and speed across problem types, while NPOD and NPOPT serve as complementary specialists for unimodal-fast and multimodal-thorough analyses respectively.

Future work should evaluate these algorithms across higher-dimensional problems, different model types (nonlinear mixed effects, time-to-event), and real-world clinical datasets to determine whether the performance hierarchy observed here generalizes beyond the bimodal Ke scenario.

---

## References

1. Sheiner L, Beal S. Evaluation of methods for estimating population pharmacokinetic parameters. I. Biexponential model and experimental pharmacokinetic data. _J Pharmacokinet Biopharm_. 1980;8:553–571.

2. Bauer R, Guzy S, Ng C. A survey of population analysis methods and software for complex pharmacokinetic and pharmacodynamic models with examples. _AAPS J_. 2007;9:E60–E83.

3. Neely M, van Guilder M, Yamada W, Schumitzky A, Jelliffe R. Accurate detection of outliers and subpopulations with Pmetrics, a nonparametric and parametric pharmacometric modeling and simulation package for R. _Ther Drug Monit_. 2012;34:467–476.

4. Beal SL, Sheiner LB. NONMEM users guides. NONMEM Project Group, University of California, San Francisco; 1992.

5. Lavielle M. Mixed Effects Models for the Population Approach: Models, Tasks, Methods and Tools. Chapman & Hall/CRC; 2014.

6. Goutelle S, Woillard JB, Buclin T, et al. Parametric and nonparametric methods in population pharmacokinetics: experts' discussion on use, strengths, and limitations. _J Clin Pharmacol_. 2022;62:158–170.

7. Goutelle S, Woillard JB, Neely M, Yamada W, Bourguignon L. Nonparametric methods in population pharmacokinetics. _J Clin Pharmacol_. 2022;62:142–157.

8. Lindsay BG. The geometry of mixture likelihoods: a general theory. _Ann Statist_. 1983;11:86–94.

9. Mallet A. A maximum likelihood estimation method for random coefficient regression models. _Biometrika_. 1986;73:645–656.

10. Yamada WM, Neely MN, Bartroff J, et al. An algorithm for nonparametric estimation of a multivariate mixing distribution with applications to population pharmacokinetics. _Pharmaceutics_. 2021;13:42.

11. Boyd S, Vandenberghe L. _Convex Optimization_. Cambridge University Press; 2004.

12. Jelliffe R, Bayard D, Milman M, van Guilder M, Schumitzky A. Achieving target goals most precisely using nonparametric compartmental models and 'Multiple Model' design of dosage regimens. _Ther Drug Monit_. 2000;22:346–353.

13. Hovd M, Kryshchenko A, Neely MN, Otalvaro JD, Schumitzky A, Yamada WM. A non-parametric optimal design algorithm for population pharmacokinetics. _arXiv:2502.15848_. 2025.

14. Daly AK. Pharmacogenomics of adverse drug reactions. _Genome Med_. 2013;5:5.

15. Fedorov VV. Theory of Optimal Experiments. Academic Press; 1972.

16. Chen R, Schumitzky A, Kryshchenko A, et al. RPEM: Randomized Monte Carlo parametric expectation maximization algorithm. _arXiv:2206.02077_. 2022.

---

## Supplementary Materials

### S1. Algorithm Hyperparameters

**Table S1.** Key hyperparameters for each algorithm.

| Parameter       | NPAG | NPOD | NPSAH | NPSAH2 | NPCAT | NPPSO | NPCMA | NPXO | NPBO | NEXUS | NPOPT |
| --------------- | ---- | ---- | ----- | ------ | ----- | ----- | ----- | ---- | ---- | ----- | ----- |
| Initial eps     | 0.2  | —    | 0.2   | 0.2    | —     | —     | 0.2   | —    | —    | —     | 0.2   |
| Initial T       | —    | —    | 1.0   | 1.5    | —     | 3.0   | —     | —    | —    | 5.0   | 2.0   |
| Cooling rate    | —    | —    | 0.95  | 0.88\* | —     | 0.95  | —     | —    | —    | 0.92  | 0.90  |
| NM iters (high) | —    | 5    | 100   | 80     | —     | —     | —     | —    | —    | 100   | 80    |
| Warmup cycles   | —    | —    | 5     | 3      | —     | 3     | 3     | —    | 5    | 3     | 3     |
| SA inject count | —    | —    | 10    | 10     | —     | 15    | —     | —    | —    | 10    | 30    |
| Swarm size      | —    | —    | —     | —      | —     | 40    | —     | —    | —    | —     | —     |
| CMA pop         | —    | —    | —     | —      | —     | —     | 20    | —    | —    | —     | —     |
| CE samples      | —    | —    | —     | —      | —     | —     | —     | —    | —    | 50    | —     |
| GP obs limit    | —    | —    | —     | —      | —     | —     | —     | —    | 1000 | —     | —     |
| Fisher ratio    | —    | —    | —     | —      | 0.60  | —     | —     | —    | —    | —     | 0.70  |
| Elite count     | —    | —    | —     | 3      | —     | 10    | —     | —    | —    | 5     | 5     |
| Sobol samples   | —    | —    | —     | —      | 256   | —     | —     | —    | 50   | 1024  | 256   |

\*Adaptive: base rate shown; actual rate adapts based on acceptance ratio.

### S2. Detailed Results Tables

_[Full per-seed results for all categories will be included here]_

### S3. Support Point Distributions

_[Kernel density plots of final support point distributions for representative algorithms on Dataset A]_
