# Visual Algorithm Flowcharts

## NPAGFULL vs NPAGFULL11 Comparison

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            NPAGFULL                                       │
│                      (Full Optimization)                                  │
└──────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Prior Density  │
                    │  (N support pts)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Read Patient   │
                    │     Data        │
                    └────────┬────────┘
                             │
                ┌────────────▼───────────┐
                │   Initialize:          │
                │   - ICYCLE = 0         │
                │   - resolve = 0.2      │
                │   - eps = 0.2          │
                └────────────┬───────────┘
                             │
              ┌──────────────▼──────────────┐
              │  CYCLE LOOP (1..MAXCYC)     │
              │  ┌─────────────────────┐    │
              │  │   1. EVALUATION     │    │
              │  │   Calculate P(y|θᵢ) │    │
              │  │   for all θᵢ        │    │
              │  └──────────┬──────────┘    │
              │             │                │
              │  ┌──────────▼──────────┐    │
              │  │  2. GAMMA OPTIMIZE  │    │
              │  │  Try γ×(1±δ)        │    │
              │  │  Keep best γ        │    │
              │  └──────────┬──────────┘    │
              │             │                │
              │  ┌──────────▼──────────┐    │
              │  │  3. IPM (emint)     │    │
              │  │  Optimize weights   │    │
              │  │  Get objective fn   │    │
              │  └──────────┬──────────┘    │
              │             │                │
              │  ┌──────────▼──────────┐    │
              │  │  4. CONDENSATION    │    │
              │  │  - Drop low prob pts│    │
              │  │  - QR filter        │    │
              │  └──────────┬──────────┘    │
              │             │                │
              │  ┌──────────▼──────────┐    │
              │  │  5. CHECK CONVERGE  │    │
              │  │  |Δobjf| < tol?     │    │
              │  └──────────┬──────────┘    │
              │             │                │
              │       ┌─────┴─────┐          │
              │       │           │          │
              │    Yes│           │No        │
              │   ┌───▼──┐    ┌───▼────┐    │
              │   │DONE  │    │EXPAND  │    │
              │   │EXIT  │    │ GRID   │    │
              │   └──────┘    └───┬────┘    │
              │                   │          │
              │            ┌──────┴─────┐   │
              │            │  Continue  │   │
              │            │  Next Cycle│   │
              │            └────────────┘   │
              └──────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Return: 1 pt   │
                    │  (MAP estimate) │
                    └─────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                           NPAGFULL11                                      │
│                      (Bayesian Filtering Only)                            │
└──────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Prior Density  │
                    │  (N support pts)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Read Patient   │
                    │     Data        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Set MAXCYC = 0 │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │      SINGLE EVALUATION      │
              │  ┌─────────────────────┐    │
              │  │  Calculate P(y|θᵢ)  │    │
              │  │  for each θᵢ        │    │
              │  └──────────┬──────────┘    │
              │             │                │
              │  ┌──────────▼──────────┐    │
              │  │  Bayes' Rule        │    │
              │  │  P(θᵢ|y) ∝         │    │
              │  │  P(y|θᵢ) × P(θᵢ)   │    │
              │  └──────────┬──────────┘    │
              │             │                │
              │  ┌──────────▼──────────┐    │
              │  │  Integration        │    │
              │  │  P(y) = ΣP(y|θᵢ)P(θᵢ)│  │
              │  └──────────┬──────────┘    │
              │             │                │
              │  ┌──────────▼──────────┐    │
              │  │  Normalize          │    │
              │  │  P(θᵢ|y)/P(y)       │    │
              │  └──────────┬──────────┘    │
              │             │                │
              │  ┌──────────▼──────────┐    │
              │  │  FILTER             │    │
              │  │  Keep if:           │    │
              │  │  P(θᵢ|y) >          │    │
              │  │  1e-100 × max       │    │
              │  └──────────┬──────────┘    │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Return: M pts  │
                    │  (M typically   │
                    │   5-50 points)  │
                    └─────────────────┘


                          KEY DIFFERENCE

    ┌────────────────────────────┬────────────────────────────┐
    │        NPAGFULL            │        NPAGFULL11          │
    ├────────────────────────────┼────────────────────────────┤
    │  Cycles: 1 to MAXCYC       │  Cycles: 0 (hardcoded)     │
    │  Method: Full optimization │  Method: Bayesian only     │
    │  Output: 1 refined point   │  Output: M filtered points │
    │  Time: Minutes             │  Time: Seconds             │
    │  Accuracy: High (optimized)│  Accuracy: Moderate (grid) │
    │  Uncertainty: Lost         │  Uncertainty: Preserved    │
    └────────────────────────────┴────────────────────────────┘
```

---

## BestDose Algorithm Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         BESTDOSE ALGORITHM                                 │
│                    (Optimal Dose Determination)                            │
└───────────────────────────────────────────────────────────────────────────┘


         INPUT                                  PROCESS

    ┌─────────────┐                    ┌────────────────────────┐
    │   Prior     │                    │  STAGE 1: POSTERIOR    │
    │  Density    │                    │      CALCULATION       │
    │ (NPAG file) │────┐               │                        │
    └─────────────┘    │               │  Step 1.1: Filter      │
                       │               │  ┌──────────────────┐  │
    ┌─────────────┐    │               │  │  NPAGFULL11      │  │
    │   Patient   │    ├──────────────►│  │  Bayesian Filter │  │
    │   "Past"    │    │               │  └────────┬─────────┘  │
    │  (optional) │────┘               │           │            │
    └─────────────┘                    │      N → M points      │
                                       │      (M ≈ 5-50)        │
    ┌─────────────┐                    │           │            │
    │   Future    │                    │  Step 1.2: Refine      │
    │  Template   │                    │  ┌────────▼─────────┐  │
    │  + Targets  │                    │  │  For each of M:  │  │
    └─────┬───────┘                    │  │  Call NPAGFULL   │  │
          │                            │  │  (full optimize) │  │
          │                            │  └────────┬─────────┘  │
          │                            │           │            │
          │                            │      M refined points  │
          │                            │      with probs from   │
          │                            │      NPAGFULL11        │
          │                            └────────────┬───────────┘
          │                                         │
          │                            ┌────────────▼───────────┐
          │                            │  STAGE 2: OPTIMIZE     │
          │                            │       DOSES            │
          │                            │                        │
          └───────────────────────────►│  For candidate dose d: │
                                       │                        │
                                       │  ┌──────────────────┐  │
                                       │  │ COST FUNCTION    │  │
                                       │  │                  │  │
                                       │  │ For each θᵢ of M:│  │
                                       │  │   Simulate(θᵢ,d) │  │
                                       │  │   pred[i] = y(θᵢ)│  │
                                       │  │                  │  │
                                       │  │ Variance =       │  │
                                       │  │   Σ P(θᵢ|past) × │  │
                                       │  │   Σⱼ(tgt-pred)²  │  │
                                       │  │                  │  │
                                       │  │ Bias² =          │  │
                                       │  │   Σⱼ(tgt-ȳⱼ)²    │  │
                                       │  │   ȳⱼ=ΣP(θᵢ)predⱼ │  │
                                       │  │                  │  │
                                       │  │ Cost =           │  │
                                       │  │   (1-λ)Var+λBias²│  │
                                       │  └────────┬─────────┘  │
                                       │           │            │
                                       │  ┌────────▼─────────┐  │
                                       │  │ NELDER-MEAD      │  │
                                       │  │ Simplex Search   │  │
                                       │  │ (~100-1000 iters)│  │
                                       │  └────────┬─────────┘  │
                                       │           │            │
                                       │    Find dose* that     │
                                       │    minimizes Cost      │
                                       └────────────┬───────────┘
                                                    │
                                       ┌────────────▼───────────┐
                                       │  STAGE 3: PREDICT      │
                                       │       OUTPUT           │
                                       │                        │
                                       │  With optimal dose*:   │
                                       │  ┌──────────────────┐  │
                                       │  │ Recalculate psi  │  │
                                       │  │ Run IPM (burke)  │  │
                                       │  │ Get weights w    │  │
                                       │  └────────┬─────────┘  │
                                       │           │            │
                                       │  ┌────────▼─────────┐  │
                                       │  │ Generate:        │  │
                                       │  │ - Predictions    │  │
                                       │  │ - Intervals      │  │
                                       │  │ - Metrics        │  │
                                       │  └────────┬─────────┘  │
                                       └────────────┬───────────┘
                                                    │
         OUTPUT                                     │
                                                    │
    ┌───────────────────────────────────────────────▼─────────┐
    │                    RESULTS                               │
    │  ┌────────────────┐  ┌────────────────┐                │
    │  │ Optimal Dose(s)│  │  Predictions   │                │
    │  │  - Amount(s)   │  │  - Time course │                │
    │  │  - Route(s)    │  │  - Intervals   │                │
    │  └────────────────┘  └────────────────┘                │
    │  ┌────────────────┐  ┌────────────────┐                │
    │  │   Metrics      │  │  Diagnostic    │                │
    │  │  - Variance    │  │  - Status      │                │
    │  │  - Bias²       │  │  - Iterations  │                │
    │  │  - Combined    │  │  - Convergence │                │
    │  └────────────────┘  └────────────────┘                │
    └───────────────────────────────────────────────────────┘
```

---

## Cost Function Visualization

```
                     BestDose Cost Function

    Cost = (1 - λ) × Variance + λ × Bias²


    VARIANCE TERM                      BIAS TERM
    (Patient-Specific)                 (Population-Level)

    ┌─────────────────┐               ┌─────────────────┐
    │  For each θᵢ:   │               │  Population     │
    │  ┌───────────┐  │               │  mean ȳⱼ:       │
    │  │ P(θᵢ|past)│  │               │  ┌───────────┐  │
    │  │    ×       │  │               │  │ Σᵢ P(θᵢ) │  │
    │  │ Σⱼ(error)²│  │               │  │    ×      │  │
    │  └───────────┘  │               │  │ pred(θᵢ)  │  │
    │                 │               │  └───────────┘  │
    │  Weighted by    │               │                 │
    │  posterior      │               │  Weighted by    │
    │  probability    │               │  prior prob.    │
    │                 │               │                 │
    │  "How uncertain │               │  "How far from  │
    │   am I about    │               │   population    │
    │   THIS patient?"│               │   average?"     │
    └─────────────────┘               └─────────────────┘
            │                                  │
            │                                  │
            └──────────┬───────────────────────┘
                       │
              ┌────────▼────────┐
              │   BALANCE via λ │
              │                 │
              │  λ=0: Full      │
              │       individual│
              │                 │
              │  λ=0.5: Mixed   │
              │                 │
              │  λ=1: Full      │
              │       population│
              └─────────────────┘


    EXAMPLE: λ = 0 (Default)
    ┌────────────────────────────────────┐
    │  Cost = Variance only              │
    │                                    │
    │  Optimize for THIS patient         │
    │  Ignores population average        │
    │  Risk: May suggest extreme doses   │
    │  Use: Good patient-specific data   │
    └────────────────────────────────────┘

    EXAMPLE: λ = 0.5
    ┌────────────────────────────────────┐
    │  Cost = 0.5×Variance + 0.5×Bias²   │
    │                                    │
    │  Balance individual & population   │
    │  Moderate personalization          │
    │  Risk: Compromise solution         │
    │  Use: Safety-conscious dosing      │
    └────────────────────────────────────┘

    EXAMPLE: λ = 1
    ┌────────────────────────────────────┐
    │  Cost = Bias² only                 │
    │                                    │
    │  Target population average         │
    │  Ignores patient specifics         │
    │  Risk: Ignores real data           │
    │  Use: No/poor patient data         │
    └────────────────────────────────────┘
```

---

## Support Point Evolution in BestDose

```
    PRIOR                NPAGFULL11         NPAGFULL          FINAL
   (Input)              (Filter)           (Refine)         (Output)

     θ₁  θ₂                θ₁                                  θ₁'
     ●   ●                 ●                                   ●
   θ₃ θ₄ θ₅              θ₂                θ₁' ──→          θ₂'
   ● ●   ●               ●                 ●                  ●
   θ₆ θ₇ θ₈              θ₃                θ₂' ──→          θ₃'
   ● ●   ●               ●                 ●                  ●
     θ₉                  θ₄                θ₃' ──→          θ₄'
     ●                   ●                 ●                  ●

  1000 points         5 points         5 optimizations      5 refined
  from NPAG           compatible       (full NPAGFULL       points with
  population          with past        each)                NPAGFULL11
  analysis            data                                  probabilities

  P(θᵢ)              P(θᵢ|past)        θᵢ → θᵢ'            P(θᵢ'|past)
  uniform            Bayesian          local                from step 2
  or from            posterior         optima               (preserved)
  prior run


  ┌─────────────┐    ┌─────────────┐   ┌─────────────┐    ┌─────────────┐
  │ Broad       │    │ Compatible  │   │ Refined     │    │ Ready for   │
  │ coverage    │───►│ with        │──►│ to local    │───►│ dose        │
  │ of          │    │ patient     │   │ optima      │    │ optimization│
  │ population  │    │             │   │             │    │             │
  └─────────────┘    └─────────────┘   └─────────────┘    └─────────────┘
      Fast               Fast              Slow               Fast
   (pre-computed)     (~seconds)        (~minutes)         (per eval)
```

---

## Nelder-Mead Simplex Visualization

```
    2D Example (optimizing 2 doses)

    Initial Simplex:

        d₂
        ▲
        │    ●3          Initial guess: (d₁₀, d₂₀)
        │   ╱ ╲         Vertices: 3 points for 2D
    d₂₀ ●1   ●2
        │              Point 1: (d₁₀, d₂₀)
        │              Point 2: (d₁₀+δ, d₂₀)
        └──────►       Point 3: (d₁₀, d₂₀+δ)
         d₁₀  d₁


    Iteration Process:

    Step 1: Evaluate cost at each vertex
            Rank: Best, Good, Worst

    Step 2: Reflect worst point through centroid
            If improved, replace worst

    Step 3: Expand or contract based on improvement

    Step 4: If stuck, shrink simplex toward best


    Cost Function Contours:

        d₂
        ▲
        │     ╱─────────╲
        │    ╱    ●3*    ╲    * = high cost
        │   │   ╱─●2──╲   │
        │   │  ● 1─→●  │  │   → = simplex moves
        │   │   ╲──●──╱   │       toward optimum
        │    ╲    ●★     ╱    ★ = optimal dose
        │     ╲─────────╱
        └─────────────────►
                        d₁


    Convergence:
    ┌─────────────────────────────────┐
    │ When simplex becomes small      │
    │ around optimal point            │
    │                                 │
    │   d₂                            │
    │   ▲                             │
    │   │    ●●                       │
    │   │     ●★  ← All vertices      │
    │   │         near optimum        │
    │   └────►                        │
    │       d₁                        │
    │                                 │
    │ Return: center of simplex       │
    └─────────────────────────────────┘
```

---

## Algorithm Timing Breakdown

```
                    BESTDOSE TIMING
                    (Typical Case)

    Total: 1-5 minutes

    ┌──────────────────────────────────────────────┐
    │ Stage 1: Posterior (10-60 seconds)           │
    ├──────────────────────────────────────────────┤
    │                                              │
    │  NPAGFULL11  ▓▓░░░░░░░░░░░░░░░░░░░░  5-10s  │
    │              (1 evaluation pass)             │
    │                                              │
    │  NPAGFULL    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  5-50s    │
    │  (×M points) (M full optimizations)          │
    │                                              │
    └──────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────┐
    │ Stage 2: Optimization (30-300 seconds)       │
    ├──────────────────────────────────────────────┤
    │                                              │
    │  Nelder-Mead iterations (100-1000)          │
    │                                              │
    │  Per iteration:                              │
    │  ├─ Set doses                   < 1ms        │
    │  ├─ For each support point (M): ~500ms      │
    │  │   └─ ODE integration         ~500ms/M    │
    │  └─ Calculate cost              < 1ms        │
    │                                              │
    │  Cost per iteration ≈ 500ms                 │
    │  Total: 500ms × 100-1000 = 50-500s          │
    │                                              │
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      │
    │  (Majority of computation time)              │
    │                                              │
    └──────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────┐
    │ Stage 3: Predictions (< 1 second)            │
    ├──────────────────────────────────────────────┤
    │                                              │
    │  Calculate psi    ▓░░░░░░░░░░░░░  < 500ms   │
    │  Burke (IPM)      ▓░░░░░░░░░░░░░  < 100ms   │
    │  Predictions      ▓░░░░░░░░░░░░░  < 400ms   │
    │                                              │
    └──────────────────────────────────────────────┘


    SPEEDUP with Parallelization (Rust):

    Sequential (Fortran):
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  300s

    4-core parallel (Rust):
    ▓▓▓▓▓▓▓▓▓  75s
    (3.5-4× faster)

    8-core parallel (Rust):
    ▓▓▓▓  40s
    (7-8× faster)
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     DATA STRUCTURES                           │
└──────────────────────────────────────────────────────────────┘

Prior           Past              Future           Optimal
Density         Data              Template         Doses
   │              │                   │               │
   ▼              ▼                   │               │
┌─────┐       ┌──────┐               │               │
│Theta│       │Data  │               │               │
│N×P  │       │T×O   │               │               │
└──┬──┘       └──┬───┘               │               │
   │             │                    │               │
   ├─────────────┴────────────────────┤               │
   │         Stage 1                  │               │
   │      NPAGFULL11                  │               │
   ▼                                  │               │
┌────────┐                            │               │
│Filtered│                            │               │
│Theta   │                            │               │
│M×P     │────────────────────────────┤               │
└────┬───┘                            │               │
     │         Stage 1                │               │
     │      NPAGFULL (×M)             │               │
     ▼                                │               │
┌─────────┐                           │               │
│Refined  │                           │               │
│Theta    │                           │               │
│M×P      │                           │               │
│+Probs   │                           │               │
└────┬────┘                           │               │
     │                                │               │
     ├────────────────────────────────┴───────┐       │
     │            Stage 2                     │       │
     │         Cost Function                  │       │
     │                                        │       │
     ▼                                        ▼       │
┌────────┐     For each candidate      ┌──────────┐  │
│ Theta  │────► dose, simulate with ───►│ Targets  │  │
│ M×P    │      all M support points    │ N×1      │  │
└────────┘                               └────┬─────┘  │
                                              │        │
     ┌────────────────────────────────────────┘        │
     │                                                 │
     ▼                                                 │
┌─────────────┐                                       │
│ Predictions │                                       │
│ M×N         │                                       │
└──────┬──────┘                                       │
       │                                              │
       │  Calculate Variance + Bias²                 │
       ▼                                              │
   ┌──────┐                                          │
   │ Cost │◄─────────────────────────────────────────┤
   │Scalar│      Nelder-Mead                         │
   └──┬───┘      Optimization                        │
      │                                               │
      └───────────────────────────────────────────────┤
                                                      │
      ┌───────────────────────────────────────────────┘
      │           Stage 3
      │         Predictions
      ▼
┌────────────┐
│  Optimal   │
│   Doses    │
│   D×1      │
└─────┬──────┘
      │
      ▼
┌─────────────┐
│ Final Psi   │
│ N×M         │
└──────┬──────┘
       │
       │  Burke (IPM)
       ▼
┌─────────────┐
│  Weights    │
│   M×1       │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Predictions  │
│Time series  │
│+ Intervals  │
└─────────────┘


Legend:
N = Number of prior support points
M = Number of posterior support points (M << N)
P = Number of parameters
T = Number of time points (past data)
O = Number of observations (past data)
D = Number of doses to optimize
```

This visual documentation provides intuitive understanding of the complex algorithms!
