#!/usr/bin/env python3
"""Plot SDE IOV verification — fair comparison on the same high-precision surface.

All points (surface sweep AND optimizer results) are evaluated with the same
SDE (500 particles × 10 resamples), making them directly comparable.

Requires: matplotlib
Data:   examples/iov_synthetic/surface.csv
        examples/iov_synthetic/optimizer.csv
Output: examples/iov_synthetic/verification.png
"""

import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib/numpy not available. Run: pip install matplotlib numpy")
    raise SystemExit(1)

# ═══════════════════════════ Load data ═══════════════════════════

ske_surface, ll_surface = [], []
with open(os.path.join(HERE, "surface.csv")) as f:
    for row in csv.DictReader(f):
        ske_surface.append(float(row["ske"]))
        ll_surface.append(float(row["log_likelihood"]))

n_particles = []
with open(os.path.join(HERE, "optimizer.csv")) as f:
    data = list(csv.DictReader(f))
    p_list   = [int(row["particles"]) for row in data]
    ske_list = [float(row["ske_opt"]) for row in data]
    ll_list  = [float(row["re_eval_ll"]) for row in data]
    conv_list = [row["converged"] == "true" for row in data]
    iters_list = [int(row["iters"]) for row in data]

TRUE_SKE = 0.08  # ground truth

# Grid peak for reference
best_i = max(range(len(ll_surface)), key=lambda i: ll_surface[i])
grid_peak_ske = ske_surface[best_i]
grid_peak_ll  = ll_surface[best_i]

# Re-evaluate truth at same precision (from output)
truth_ll = -98.20  # from the verify run

# ═══════════════════════════ Plot ═══════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
fig.suptitle(
    f"SDE IOV Verification — Fair Comparison\n"
    f"(ke=0.293, v=10, {len(data)} optimizers, all evaluated on same 500p×10 surface)",
    fontsize=13, fontweight="bold",
)

# ── LEFT: Likelihood surface (high-precision) ──

ax1.plot(ske_surface, ll_surface, "o-", color="#1565C0", markersize=4,
         linewidth=1.5, label=f"Surface ({len(ske_surface)} points, 500p × 10 resamples)")
ax1.axvline(x=TRUE_SKE, color="gray", linestyle=":", linewidth=2, alpha=0.6,
            label=f"True ske = {TRUE_SKE}")

# Grid peak
ax1.plot(grid_peak_ske, grid_peak_ll, "D", color="#1565C0", markersize=12,
         markeredgewidth=1.5, markeredgecolor="white",
         label=f"Grid peak: ske={grid_peak_ske:.4f}, LL={grid_peak_ll:.2f}")

# Truth at same evaluation precision
ax1.plot(TRUE_SKE, truth_ll, "*", color="gray", markersize=14,
         label=f"Truth at 500p×10: LL={truth_ll:.2f}")

# Optimizer results — all on the SAME surface
colors = ["#E91E63", "#9C27B0", "#FF9800", "#4CAF50"]
for i, (p, ske, ll) in enumerate(zip(p_list, ske_list, ll_list)):
    ax1.plot(ske, ll, "s", color=colors[i], markersize=10,
             markeredgewidth=1.5, markeredgecolor="white",
             label=f"NM {p}p → ske={ske:.4f}, LL={ll:.2f}")

ax1.set_xlabel("ske (diffusion coefficient)", fontsize=11)
ax1.set_ylabel("Log-likelihood", fontsize=11)
ax1.set_title("High-Precision Likelihood Surface\n(all points evaluated with 500 particles × 10 resamples)", fontsize=11)
ax1.legend(fontsize=8, loc="lower left", ncol=2)
ax1.grid(True, alpha=0.25)

# ── RIGHT: Error by particle count ──

errors = [abs(s - TRUE_SKE) for s in ske_list]
pct_errors = [abs(s - TRUE_SKE) / TRUE_SKE * 100 for s in ske_list]
colors_bar = ["#4CAF50" if c else "#FF9800" for c in conv_list]

bars = ax2.bar(range(len(p_list)), errors, tick_label=[str(p) for p in p_list],
               color=colors_bar, edgecolor="black", linewidth=0.5, width=0.55)
ax2.axhline(y=0, color="gray", linewidth=0.5)

# Add the ratio estimator bias floor as a shaded region
ax2.axhspan(0.02, 0.03, alpha=0.08, color="blue", label="Ratio estimator bias ~0.02–0.03")

for i, (ske, err, conv, iters) in enumerate(zip(ske_list, errors, conv_list, iters_list)):
    y = err + 0.005
    ax2.text(i, y, f"ske={ske:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax2.text(i, y + 0.008, f"({pct_errors[i]:.0f}%)", ha="center", fontsize=7, color="#666")
    status = f"✓{iters} iters" if conv else f"{iters}/{40}"
    ax2.text(i, y + 0.014, status, ha="center", fontsize=7, color="#888")

ax2.set_xlabel("Number of particles used during optimization", fontsize=11)
ax2.set_ylabel("|ske_recovered − 0.08|", fontsize=11)
ax2.set_title(f"Recovery Error (on same 500p×10 surface)", fontsize=11)
ax2.set_ylim(0, max(errors) * 1.35)
ax2.grid(True, alpha=0.25, axis="y")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4CAF50", label="Converged"),
    Patch(facecolor="#FF9800", label="Max iters"),
]
ax2.legend(handles=legend_elements, fontsize=9)

# ── Summary bar ──
deltas = [f"{grid_peak_ll - ll:.3f}" for ll in ll_list]
summary = (
    f"  Grid peak: ske={grid_peak_ske:.4f}  LL={grid_peak_ll:.2f}  "
    f"|  Truth: ske=0.080  LL={truth_ll:.2f}  (Δ={grid_peak_ll - truth_ll:.3f})  "
    f"|  Optimizer ΔLL from grid: [{', '.join(deltas)}]"
)
fig.text(0.5, 0.01, summary, ha="center", fontsize=8, family="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.94])
out = os.path.join(HERE, "verification.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
print(f"  Grid peak:   ske={grid_peak_ske:.4f}  LL={grid_peak_ll:.2f}")
print(f"  Truth:       LL={truth_ll:.2f}  (Δ={grid_peak_ll - truth_ll:.3f})")
for p, ske, ll, d in zip(p_list, ske_list, ll_list, deltas):
    print(f"  NM {p:>3}p:  ske={ske:.4f}  LL={ll:.2f}  ΔLL={d}")
