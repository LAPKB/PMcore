
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

HERE = "/Users/siel/code/LAPKB/PMcore/examples/iov_synthetic"

particles, peaks, biases = [], [], []
with open(f"{HERE}/calibration.csv") as f:
    for row in csv.DictReader(f):
        particles.append(int(row["particles"]))
        peaks.append(float(row["grid_peak"]))
        biases.append(float(row["bias"]))

TRUE = 0.08
plateau_mean = -0.0176

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Ratio Estimator Bias Calibration — SDE Particle Filter", fontsize=13, fontweight="bold")

# Left: bias vs M
ax1.axhline(y=0, color="gray", linewidth=0.5)
ax1.axhline(y=plateau_mean, color="red", linestyle="--", linewidth=1.5, alpha=0.6,
            label=f"Plateau bias = {plateau_mean:.4f} ({abs(plateau_mean)/TRUE*100:.0f}%)")
ax1.plot(particles, biases, "o-", color="#1565C0", markersize=8, linewidth=2)
for p, b in zip(particles, biases):
    ax1.text(p, b + (0.002 if b > 0 else -0.004), f"{b:+.4f}", ha="center", fontsize=8)
ax1.set_xlabel("Number of particles M", fontsize=12)
ax1.set_ylabel("Bias = grid_peak − 0.08", fontsize=12)
ax1.set_title(f"Bias vs Particle Count (N={20} subjects)", fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)
ax1.set_xscale("log")
ax1.set_xticks(particles)
ax1.set_xticklabels([str(p) for p in particles])

# Right: grid peak vs M
ax2.axhline(y=TRUE, color="green", linestyle="--", linewidth=2, label=f"True skee = {TRUE}")
ax2.axhline(y=TRUE + plateau_mean, color="red", linestyle=":", linewidth=1.5, alpha=0.6,
            label=f"Plateau = {TRUE + plateau_mean:.4f}")
ax2.plot(particles, peaks, "s-", color="#E65100", markersize=8, linewidth=2)
for p, pk in zip(particles, peaks):
    ax2.text(p, pk + 0.003, f"{pk:.4f}", ha="center", fontsize=8)
ax2.set_xlabel("Number of particles M", fontsize=12)
ax2.set_ylabel("Grid peak sike", fontsize=12)
ax2.set_title(f"Grid Peak vs Particle Count", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)
ax2.set_xscale("log")
ax2.set_xticks(particles)
ax2.set_xticklabels([str(p) for p in particles])

# Summary
summary = (
    f"Bias plateaus at {plateau_mean:+.4f} ({abs(plateau_mean)/TRUE*100:.0f}%) for M >= 100. "
    f"Finite-subject limited (N={20}), not particle-count limited. "
    f"M > 100 buys nothing."
)
fig.text(0.5, 0.01, summary, ha="center", fontsize=9, family="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 0.94])
plt.savefig(f"{HERE}/calibration.png", dpi=150)
print(f"Saved calibration.png")
print(f"Plateau bias: {plateau_mean:+.4f} ({abs(plateau_mean)/TRUE*100:.0f}%)")
