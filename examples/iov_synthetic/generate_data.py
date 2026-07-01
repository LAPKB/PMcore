#!/usr/bin/env python3
"""Generate synthetic SDE IOV data with known ground truth.

Produces a Pmetrics-format CSV file with subjects simulated from a
1-compartment IV bolus model with Ornstein-Uhlenbeck process on ke.

Ground truth:
  ke  = 0.3    (mean reversion target)
  v   = 10.0   (volume of distribution)
  ske = 0.08   (diffusion coefficient — the parameter we want to recover)

Each subject gets an independent SDE trajectory via Euler-Maruyama.
Observations have 2% proportional measurement noise.

Output: data.csv (Pmetrics format)

Usage:
  python3 generate_synthetic_data.py
  # Produces data.csv in the current directory
"""

import math
import random
import csv
import os

# ═══════════════════════════════════════════════════════════════════════
# Ground truth
# ═══════════════════════════════════════════════════════════════════════
KE_TRUE = 0.3
V_TRUE = 10.0
SKE_TRUE = 0.08
N_SUBJECTS = 20
DOSE = 100.0
OBS_TIMES = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]  # hours post-dose
SEED = 42

# ═══════════════════════════════════════════════════════════════════════
# Box-Muller normal random number generator
# ═══════════════════════════════════════════════════════════════════════
def normal(mean=0.0, std=1.0):
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z

# ═══════════════════════════════════════════════════════════════════════
# SDE simulation: 1-cmt IV bolus with OU process on ke
# ═══════════════════════════════════════════════════════════════════════
def simulate_subject(ke0, v, ske, dose, obs_times):
    """Euler-Maruyama simulation of:
    
    d(ke_latent) = -(ke_latent - ke0)·dt + ske·dW     (OU process)
    d(central)   = -ke_latent·central·dt                (drug elimination)
    
    Initial: ke_latent(0) = ke0, central(0) = dose
    
    Observations: central(t) / v + N(0, 0.02·central/v)
    """
    dt = 0.01
    t_max = max(obs_times) + 0.5
    n_steps = int(t_max / dt)

    ke_latent = ke0
    central = 0.0
    observations = []
    obs_idx = 0

    for step in range(n_steps):
        t = step * dt

        # Apply bolus at t=0
        if step == 0:
            central += dose

        # Euler-Maruyama step for OU process
        dW = normal(0, math.sqrt(dt))
        ke_latent += (-ke_latent + ke0) * dt + ske * dW

        # ODE-like step for concentration
        central += -ke_latent * central * dt

        # Record observation at designated times
        if obs_idx < len(obs_times) and abs(t - obs_times[obs_idx]) < dt / 2:
            conc = central / v
            conc_obs = conc + normal(0, 0.02 * conc)
            observations.append((t, max(conc_obs, 0.01)))
            obs_idx += 1

    return observations


# ═══════════════════════════════════════════════════════════════════════
# Main: generate data and write CSV
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    random.seed(SEED)

    all_data = []
    for subj_id in range(1, N_SUBJECTS + 1):
        obs = simulate_subject(KE_TRUE, V_TRUE, SKE_TRUE, DOSE, OBS_TIMES)
        for t, conc in obs:
            all_data.append({
                "ID": subj_id, "EVID": 0, "TIME": round(t, 4),
                "DUR": ".", "DOSE": ".", "ADDL": ".", "II": ".",
                "INPUT": ".", "OUT": round(conc, 6), "OUTEQ": 1,
                "C0": ".", "C1": ".", "C2": ".", "C3": ".", "BLOCK": 1,
            })
        # Dose event
        all_data.append({
            "ID": subj_id, "EVID": 1, "TIME": 0.0, "DUR": 0.0,
            "DOSE": DOSE, "ADDL": ".", "II": ".", "INPUT": 1,
            "OUT": ".", "OUTEQ": ".",
            "C0": ".", "C1": ".", "C2": ".", "C3": ".", "BLOCK": 1,
        })

    # Sort: ID ascending, TIME ascending, EVID descending (doses before obs)
    all_data.sort(key=lambda x: (x["ID"], x["TIME"], -x["EVID"]))

    fieldnames = [
        "ID", "EVID", "TIME", "DUR", "DOSE", "ADDL", "II",
        "INPUT", "OUT", "OUTEQ", "C0", "C1", "C2", "C3", "BLOCK",
    ]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"Generated {len(all_data)} rows ({N_SUBJECTS} subjects)")
    print(f"Ground truth: ke={KE_TRUE}, v={V_TRUE}, ske={SKE_TRUE}")
    print(f"Saved to {out_path}")
