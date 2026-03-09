# Comprehensive Experiment Design for Algorithm Comparison Paper

## 1. Philosophy: Impartial Evaluation

**Key Principle**: No algorithm is universally best. Each algorithm has strengths and weaknesses that emerge under different conditions:

- **Problem dimensionality** (2 params vs 10+ params)
- **Distribution shape** (unimodal vs multimodal)
- **Sample size** (sparse vs rich data)
- **Model complexity** (analytical vs complex ODE)
- **Special features** (lag times, IOV, covariates)

Our experiments must be designed to reveal these trade-offs, not to crown a single winner.

---

## 2. Available Datasets and Models

### 2.1 Dataset Inventory (from PMcore examples)

| Dataset          | Model Type               | Parameters          | Subjects | Obs/Subj | Expected Behavior | Key Challenge            |
| ---------------- | ------------------------ | ------------------- | -------- | -------- | ----------------- | ------------------------ |
| **bimodal_ke**   | 1-comp IV                | 2 (ke, v)           | 51       | ~10      | Bimodal ke        | Multimodality            |
| **theophylline** | 1-comp oral (analytical) | 3 (ka, ke, v)       | 12       | ~11      | Unimodal          | Standard reference       |
| **two_eq_lag**   | 2-comp oral + lag        | 4 (ka, ke, tlag, v) | 20       | ~7       | Moderate          | Lag identifiability      |
| **drusano**      | 5-comp PK-PD             | 24                  | 9        | ~30      | Very complex      | High dimensionality      |
| **neely**        | 4-comp + metabolites     | 10                  | 22       | ~18      | Hard              | Multi-output, covariates |
| **meta**         | 2-comp + metabolite      | 7                   | 19       | ~12      | Moderate          | Multi-output, covariates |

### 2.2 Dataset Characteristics Matrix

| Dataset      | Dims        | Multimodal? | Correlations?      | Identifiability | Covariate Effects |
| ------------ | ----------- | ----------- | ------------------ | --------------- | ----------------- |
| bimodal_ke   | Low (2)     | Yes         | Low                | High            | None              |
| theophylline | Low (3)     | No          | Moderate           | High            | None              |
| two_eq_lag   | Low (4)     | Unknown     | Moderate (ka-tlag) | Moderate (tlag) | None              |
| drusano      | High (24)   | Unknown     | High (PD params)   | Low             | Yes (IC)          |
| neely        | Medium (10) | Unknown     | Moderate           | Moderate        | Yes (wt, pkvisit) |
| meta         | Medium (7)  | Unknown     | Moderate           | Moderate        | Yes (wt, pkvisit) |

---

## 3. Experiment Categories

### 3.1 Category A: Reproducibility & Stability

**Goal**: Assess algorithm robustness across different random seeds

**Design**:

- Dataset: bimodal_ke (simple, known bimodal)
- Algorithms: All 11
- Seeds: 5 different (e.g., 42, 123, 456, 789, 1001)
- Metrics: Mean -2LL, SD of -2LL, % runs finding both modes

**Rationale**: Some algorithms (especially stochastic ones like NPPSO, NPSAH) may have high variance. This test reveals stability.

**Expected Outcomes**:

- NPAG: Very stable (deterministic grid)
- NPOD: Moderate variance (deterministic after init)
- SA-based (NPSAH, NPSAH2): Some variance from temperature schedule
- NPPSO: Higher variance from swarm randomness
- NPCMA: Moderate variance from sampling

### 3.2 Category B: Scalability with Dimensionality

**Goal**: Test how algorithms scale as parameters increase

**Design**:
| Test | Dataset | Parameters | Expected Winner |
|------|---------|------------|-----------------|
| B1 | bimodal_ke | 2 | SA-based (can explore) |
| B2 | theophylline | 3 | All should work well |
| B3 | two_eq_lag | 4 | Test lag handling |
| B4 | meta | 7 | NPPSO, NEXUS (scale better) |
| B5 | neely | 10 | NPPSO, NEXUS (scale better) |
| B6 | drusano | 24 | NPAG (safe), NPPSO (scalable) |

**Metrics**: -2LL, time, cycles, support points

**Expected Trade-offs**:

- NPBO: GP complexity grows O(n³), may struggle with high dims
- NPCMA: Covariance matrix grows O(d²), may struggle >10 params
- NPPSO: Swarm scales well with subjects
- NEXUS: CE-based, less affected by dimensionality

### 3.3 Category C: Multimodality Detection

**Goal**: Test ability to find multiple modes in the distribution

**Design**:

- Dataset: bimodal_ke (known bimodal in ke)
- Algorithms: All 11
- Analysis:
  - Count support points in each mode
  - Check if both modes are represented with >5% weight
  - Plot marginal distributions

**Mode Detection Criteria**:

```
Mode 1: ke ∈ [0.05, 0.15] (slow eliminators)
Mode 2: ke ∈ [0.25, 0.40] (fast eliminators)

Success = both modes have ≥2 support points with weight >1%
```

**Expected Outcomes**:

- NPOD: May miss secondary mode (local optimizer)
- NPAG: Should find both (grid coverage)
- SA-based: Should find both (global exploration)
- NPPSO: Should find both (swarm diversity)
- NPCMA: May converge to one mode

### 3.4 Category D: Convergence Speed

**Goal**: Measure time-to-quality trade-offs

**Design**:

- Dataset: theophylline (clean, fast to run)
- Algorithms: All 11
- Metrics at various cycle counts:
  - After 5 cycles
  - After 10 cycles
  - After 25 cycles
  - After 50 cycles
  - After 100 cycles
  - At convergence

**Analysis**: Plot -2LL vs cycles (or time) for each algorithm

**Expected Outcomes**:

- NPOD: Fast initial improvement, early plateau
- NPSAH2: Slower start, best final quality
- NPAG: Many cycles but steady improvement
- NPPSO: Fast exploration, gradual refinement

### 3.5 Category E: Lag Time Estimation

**Goal**: Test ability to estimate absorption lag times (identifiability challenge)

**Design**:

- Dataset: two_eq_lag (4 params including tlag)
- Algorithms: NPAG, NPOD, NPSAH, NPSAH2, NPPSO, NPCAT, NEXUS
- Metrics:
  - -2LL
  - Recovered tlag distribution
  - Correlation between ka and tlag estimates

**Rationale**: Lag time creates flat likelihood regions where different (ka, tlag) combinations produce similar predictions. Tests optimization robustness.

**Expected Challenges**:

- Local optimizers (NPOD) may get stuck
- Global searchers (NPSAH, NPPSO) should explore the ridge

### 3.6 Category F: High-Dimensional Stress Test

**Goal**: Evaluate algorithms on complex, high-dimensional problem

**Design**:

- Dataset: drusano (24 parameters, 5 outputs)
- Algorithms: NPAG, NPOD, NPSAH, NPPSO, NEXUS
- Max cycles: 1000 (or 1 hour timeout)
- Seeds: 3

**Metrics**:

- -2LL achieved
- Time per cycle
- Total time
- Memory usage (if trackable)
- Number of support points

**Expected Outcomes**:

- Many algorithms may struggle
- NPAG: Safe but slow
- NPOD: Fast but may get stuck
- NPPSO: Best hope for global exploration
- NEXUS: CE may help navigate high-dim space

### 3.7 Category G: Multi-Output Models

**Goal**: Test algorithms on models with multiple observed outputs

**Design**:

- Dataset: neely (3 outputs: parent + 2 metabolites) or meta (2 outputs)
- Algorithms: All
- Metrics:
  - Overall -2LL
  - Per-output fit quality
  - Covariate effect recovery

**Rationale**: Multi-output models have more complex likelihood surfaces. Tests if algorithms balance fit across outputs.

---

## 4. Statistical Analysis Plan

### 4.1 Primary Metrics

| Metric            | Description                   | Lower/Higher Better   |
| ----------------- | ----------------------------- | --------------------- |
| **-2LL**          | Negative twice log-likelihood | Lower (more negative) |
| **Cycles**        | Iterations to convergence     | Lower                 |
| **Time**          | Wall-clock seconds            | Lower                 |
| **NSP**           | Number of support points      | Context-dependent     |
| **Mode Coverage** | Fraction of true modes found  | Higher                |

### 4.2 Statistical Tests

**Pairwise Comparisons**:

- Wilcoxon signed-rank test (paired, non-parametric)
- Paired t-test (if normality holds)
- Multiple seed results as replicates

**Multiple Comparison Correction**:

- Bonferroni or Benjamini-Hochberg for multiple algorithms
- Report adjusted p-values

**Effect Size**:

- Cohen's d for -2LL differences
- Percentage improvement over baseline (NPAG)

### 4.3 Visualization Plan

1. **Box plots**: -2LL by algorithm (across seeds)
2. **Convergence curves**: -2LL vs cycles (or time)
3. **Heatmaps**: Algorithm × Dataset performance matrix
4. **Radar charts**: Multi-dimensional comparison (speed, quality, stability)
5. **Marginal distributions**: Compare estimated distributions to true (if known)

---

## 5. Implementation Plan

### 5.1 Benchmark Script Structure

```rust
// examples/paper_benchmarks.rs
struct BenchmarkConfig {
    name: String,
    dataset: String,
    algorithms: Vec<Algorithm>,
    seeds: Vec<u64>,
    max_cycles: usize,
    timeout_secs: u64,
}

struct BenchmarkResult {
    algorithm: String,
    seed: u64,
    dataset: String,
    objf: f64,
    cycles: usize,
    time_secs: f64,
    n_support_points: usize,
    theta: Vec<Vec<f64>>,
    weights: Vec<f64>,
}
```

### 5.2 Execution Order

**Phase 1: Quick Tests** (can run in parallel)

1. Category A (bimodal_ke, 5 seeds, all algorithms) - ~30 min
2. Category C (bimodal_ke, mode detection) - use Phase 1 results
3. Category D (theophylline, convergence) - ~20 min

**Phase 2: Moderate Tests** 4. Category B tests B1-B4 - ~2 hours 5. Category E (two_eq_lag) - ~1 hour 6. Category G (meta or neely) - ~1 hour

**Phase 3: Stress Tests** 7. Category B test B5-B6 (high-dim) - ~4+ hours 8. Category F (drusano full) - ~8+ hours

### 5.3 Resource Estimates

| Test Category       | Estimated Time | Parallelizable |
| ------------------- | -------------- | -------------- |
| A (reproducibility) | 30-60 min      | Yes (by seed)  |
| B (scalability)     | 4-6 hours      | Partially      |
| C (multimodality)   | Uses A results | N/A            |
| D (convergence)     | 20-30 min      | Yes            |
| E (lag time)        | 1-2 hours      | Yes            |
| F (stress)          | 8+ hours       | Limited        |
| G (multi-output)    | 1-2 hours      | Yes            |

**Total: ~15-20 hours of computation**

---

## 6. Expected Findings & Hypotheses

### 6.1 Primary Hypotheses

**H1**: SA-based algorithms (NPSAH, NPSAH2) will achieve better -2LL on multimodal problems (bimodal_ke) than gradient-based (NPOD).

**H2**: NPOD will be fastest for unimodal, low-dimensional problems (theophylline).

**H3**: Algorithm stability (variance across seeds) will be inversely related to exploration intensity.

**H4**: NPPSO and NEXUS will scale better to high-dimensional problems than NPCMA and NPBO.

**H5**: No algorithm will be best across all datasets - trade-offs will emerge.

### 6.2 Anticipated Trade-off Matrix

| Scenario                     | Likely Best  | Likely Worst       |
| ---------------------------- | ------------ | ------------------ |
| Fast approximation           | NPOD         | NPSAH2 (slow)      |
| Best quality (no time limit) | NPSAH2       | NPXO               |
| Multimodal                   | NPPSO, NPSAH | NPOD               |
| High-dimensional             | NPPSO, NEXUS | NPBO, NPCMA        |
| Most stable                  | NPAG         | NPPSO (stochastic) |
| Best speed-quality           | NPOD, NPSAH  | NEXUS              |

---

## 7. Paper Narrative Framework

### 7.1 Story Arc

1. **Introduction**: NP estimation importance, current limitations
2. **Background**: NPAG as gold standard, NPOD as first optimization
3. **Methods**: Introduce new algorithms (SA, PSO, CMA, BO, CE)
4. **Experiments**: Fair comparison across diverse scenarios
5. **Results**: Trade-offs revealed, no single winner
6. **Discussion**: When to use which algorithm
7. **Recommendations**: Decision tree for practitioners

### 7.2 Key Messages

- **Message 1**: NPOD improves speed but sacrifices global exploration
- **Message 2**: SA-based hybrids (NPSAH, NPSAH2) recover global exploration while maintaining efficiency
- **Message 3**: Different algorithms excel in different scenarios
- **Message 4**: Algorithm choice should be guided by problem characteristics
- **Message 5**: Implementation in PMcore makes these algorithms accessible

---

## 8. Immediate Action Items

### 8.1 Create Benchmark Infrastructure

```bash
# Create benchmark runner
touch examples/paper_benchmarks/mod.rs
touch examples/paper_benchmarks/category_a.rs
touch examples/paper_benchmarks/category_b.rs
# etc.
```

### 8.2 Run Initial Quick Tests

1. **First**: Category A (bimodal_ke, 5 seeds) - establishes baseline
2. **Second**: Category D (theophylline convergence) - quick diagnostic
3. **Third**: Category E (two_eq_lag) - lag time challenge

### 8.3 Data Collection Format

CSV output for each run:

```csv
experiment,dataset,algorithm,seed,cycles,time_secs,objf,n_spp,converged
A1,bimodal_ke,NPAG,42,326,9.98,-347.93,46,true
A1,bimodal_ke,NPOD,42,13,3.03,-375.22,45,true
...
```

---

## 9. Appendix: Algorithm Quick Reference

| Algorithm | Type        | Global Search        | Local Refinement | Expected Strength     |
| --------- | ----------- | -------------------- | ---------------- | --------------------- |
| NPAG      | Grid        | Systematic expansion | None             | Baseline, stable      |
| NPOD      | D-optimal   | None                 | Nelder-Mead      | Fast, unimodal        |
| NPSAH     | SA+D-opt    | SA injection         | Adaptive NM      | Balanced              |
| NPSAH2    | SA+D-opt    | 4-phase SA+LHS       | Hierarchical NM  | Best quality          |
| NPCAT     | Categorical | Unknown              | Unknown          | To investigate        |
| NPPSO     | PSO         | Swarm                | COBYLA           | Scalable              |
| NPCMA     | CMA-ES      | Covariance           | Evolution paths  | Correlated params     |
| NPXO      | Crossover   | Genetic              | None             | Fast but poor         |
| NPBO      | Bayesian    | GP+EI                | None             | Low-dim only          |
| NEXUS     | CE+Subject  | Cross-entropy        | Hierarchical NM  | Convergence guarantee |
| NPOPT     | Phased      | SA+Fisher            | Hierarchical NM  | Phased approach       |

---

_Document version: 1.0_
_Created: January 2026_
_Purpose: Guide comprehensive algorithm comparison experiments_
