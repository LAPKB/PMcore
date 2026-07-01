# SDE IOV Verification — Reproduction Guide

Reproduces the synthetic verification of the SDE IOV sigma optimization pipeline.

## Files

```
generate_data.py    — Python script: generate synthetic SDE data
data.csv            — Output of generate_data.py (20 subjects, ske=0.08)
plot.py             — Python script: generate verification plot
surface.csv         — High-precision likelihood surface sweep (from Rust)
optimizer.csv       — NelderMead recovery results (from Rust)
verification.png    — Output of plot.py
```

## Steps to Reproduce

### 1. Generate Synthetic Data

```bash
cd examples/iov_synthetic
python3 generate_data.py
# → data.csv (140 rows, 20 subjects)
```

Requires only Python stdlib (no dependencies).
Ground truth: ke=0.3, v=10.0, ske=0.08.

### 2. Run the Verification Pipeline

```bash
cd ../..  # back to PMcore root
cargo run --release --example iov_verify
# → surface.csv, optimizer.csv
```

This runs:

- Stage 1: NPAG ODE fit (finds support points)
- Stage 2: SDE IOV optimization at 50/100/200/500 particles
- Re-evaluation on shared high-precision surface (500p × 10 resamples)

Expected output (approximate):

```
  SPs: 2, OBJF: ~186
  Top SP: ke≈0.29, v≈10.0
  Surface peak: ske≈0.06, LL≈-97.6
  NM 50p:   ske≈0.055, LL≈-97.6
  NM 100p:  ske≈0.062, LL≈-97.7
  NM 200p:  ske≈0.056, LL≈-97.7
  NM 500p:  ske≈0.056, LL≈-97.6
  Truth ske=0.080: LL≈-98.1 (ratio estimator bias Δ≈0.5)
```

### 3. Generate the Plot

```bash
cd examples/iov_synthetic
python3 -m venv .venv
.venv/bin/pip install matplotlib numpy
.venv/bin/python plot.py
# → verification.png
```

The plot shows:

- **Left**: High-precision likelihood surface (500p × 10 resamples) with
  optimizer results marked on the same surface
- **Right**: Recovery error by particle count

### Interpretation

All NelderMead results cluster on the same statistical plateau (ΔLL < 0.15
from grid peak). The optimizer reliably finds the likelihood maximum.

The ratio estimator bias in the particle filter likelihood appears as a
systematic shift of the apparent peak ~0.015 below the true value (0.08).
This bias is O(1/M) in particle count M and is inherent to the product-form
particle filter likelihood estimator — the optimizer cannot fix it.

## Re-running with Different Parameters

Edit the constants at the top of `generate_data.py`:

- `KE_TRUE`, `V_TRUE`, `SKE_TRUE` — true parameter values
- `N_SUBJECTS` — number of subjects
- `OBS_TIMES` — observation schedule
- `SEED` — random seed for reproducibility
