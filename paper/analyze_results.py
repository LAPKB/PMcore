#!/usr/bin/env python3
"""Analyze paper benchmark results from Category A (bimodal_ke, 5 seeds, all algorithms)."""

import csv
import os
import statistics
import json

RESULTS_FILE = "../examples/paper_benchmarks/results_1769776808.csv"
OUTPUT_DIR = "../examples/paper_benchmarks/output/bimodal_ke"

def load_results():
    results = {}
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            alg = row["algorithm"]
            if alg not in results:
                results[alg] = {"objf": [], "time": [], "cycles": [], "nspp": [], "seeds": []}
            results[alg]["objf"].append(float(row["objf"]))
            results[alg]["time"].append(float(row["time_secs"]))
            results[alg]["cycles"].append(int(row["cycles"]))
            results[alg]["nspp"].append(int(row["n_spp"]))
            results[alg]["seeds"].append(int(row["seed"]))
    return results


def print_summary(results):
    sorted_algs = sorted(results.keys(), key=lambda a: statistics.mean(results[a]["objf"]))

    print("=" * 110)
    print(
        f"{'Algorithm':<10} | {'Mean -2LL':>12} | {'SD':>8} | {'Best':>12} | {'Worst':>12} | {'Range':>8} | {'Mean Time':>10} | {'Mean Cyc':>9} | {'Mean SPP':>9}"
    )
    print("-" * 110)
    for alg in sorted_algs:
        d = results[alg]
        m = statistics.mean(d["objf"])
        sd = statistics.stdev(d["objf"]) if len(d["objf"]) > 1 else 0
        best = min(d["objf"])  # lower is better
        worst = max(d["objf"])
        rng = worst - best
        tm = statistics.mean(d["time"])
        cyc = statistics.mean(d["cycles"])
        spp = statistics.mean(d["nspp"])
        print(
            f"{alg:<10} | {m:>12.2f} | {sd:>8.2f} | {best:>12.2f} | {worst:>12.2f} | {rng:>8.2f} | {tm:>10.2f}s | {cyc:>9.1f} | {spp:>9.1f}"
        )

    print()

    # Efficiency analysis
    npag_mean = statistics.mean(results["NPAG"]["objf"])
    print("EFFICIENCY ANALYSIS (improvement over NPAG baseline):")
    print(
        f"{'Algorithm':<10} | {'Mean -2LL':>12} | {'Δ vs NPAG':>10} | {'Time(s)':>8} | {'Δ-2LL/sec':>12} | {'Speed-Quality':>15}"
    )
    print("-" * 80)
    for alg in sorted_algs:
        m = statistics.mean(results[alg]["objf"])
        t = statistics.mean(results[alg]["time"])
        delta = m - npag_mean
        rate = delta / t if t > 0 else 0
        # Classify efficiency
        if delta < -50 and t < 50:
            cat = "HIGH"
        elif delta < -30 and t < 100:
            cat = "MEDIUM"
        elif delta < 0 and t < 20:
            cat = "FAST-DECENT"
        elif delta < -50:
            cat = "SLOW-BEST"
        elif delta >= -5:
            cat = "NPAG-LEVEL"
        else:
            cat = "LOW"
        print(f"{alg:<10} | {m:>12.2f} | {delta:>10.2f} | {t:>8.2f} | {rate:>12.4f} | {cat:>15}")

    print()

    # Stability analysis (CV)
    print("STABILITY ANALYSIS (Coefficient of Variation):")
    print(f"{'Algorithm':<10} | {'CV(-2LL)':>10} | {'Interpretation':>20}")
    print("-" * 50)
    stability_order = sorted(
        results.keys(),
        key=lambda a: abs(statistics.stdev(results[a]["objf"]) / statistics.mean(results[a]["objf"]) * 100)
        if len(results[a]["objf"]) > 1
        else 0,
    )
    for alg in stability_order:
        d = results[alg]
        if len(d["objf"]) > 1:
            cv = abs(statistics.stdev(d["objf"]) / statistics.mean(d["objf"]) * 100)
            if cv < 3:
                interp = "Very Stable"
            elif cv < 5:
                interp = "Stable"
            elif cv < 8:
                interp = "Moderate"
            else:
                interp = "Variable"
            print(f"{alg:<10} | {cv:>10.2f}% | {interp:>20}")


def analyze_support_points(results):
    """Analyze theta (support point) distributions to check multimodality detection."""
    print()
    print("=" * 80)
    print("MULTIMODALITY DETECTION ANALYSIS")
    print("bimodal_ke: True distribution has TWO ke modes")
    print("  Mode 1 (slow): ke ~ 0.05-0.15")
    print("  Mode 2 (fast): ke ~ 0.25-0.45")
    print("=" * 80)

    for alg in sorted(results.keys()):
        seeds = results[alg]["seeds"]
        mode1_found = 0
        mode2_found = 0
        total_runs = len(seeds)

        for seed in seeds:
            theta_path = os.path.join(OUTPUT_DIR, f"{alg}_seed{seed}", "theta.csv")
            if not os.path.exists(theta_path):
                continue

            # Read theta file
            ke_vals = []
            weights = []
            with open(theta_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ke_vals.append(float(row.get("ke", 0)))
                        weights.append(float(row.get("prob", row.get("w", 0))))
                    except (ValueError, KeyError):
                        pass

            if not ke_vals:
                continue

            # Check mode detection
            mode1_weight = sum(w for ke, w in zip(ke_vals, weights) if 0.03 <= ke <= 0.18)
            mode2_weight = sum(w for ke, w in zip(ke_vals, weights) if 0.20 <= ke <= 0.50)

            if mode1_weight > 0.02:
                mode1_found += 1
            if mode2_weight > 0.02:
                mode2_found += 1

        both_pct = 0
        if total_runs > 0:
            both_count = min(mode1_found, mode2_found)
            both_pct = both_count / total_runs * 100

        print(
            f"  {alg:<10}: Mode1 found {mode1_found}/{total_runs}, Mode2 found {mode2_found}/{total_runs}, Both modes: {both_pct:.0f}%"
        )


def analyze_theta_detail(results):
    """Detailed look at the support point distributions for best seed per algorithm."""
    print()
    print("=" * 80)
    print("SUPPORT POINT DISTRIBUTION DETAIL (best seed per algorithm)")
    print("=" * 80)

    for alg in sorted(results.keys()):
        # Find best seed
        best_idx = results[alg]["objf"].index(min(results[alg]["objf"]))
        best_seed = results[alg]["seeds"][best_idx]
        best_objf = results[alg]["objf"][best_idx]

        theta_path = os.path.join(OUTPUT_DIR, f"{alg}_seed{best_seed}", "theta.csv")
        if not os.path.exists(theta_path):
            print(f"\n  {alg} (seed {best_seed}, -2LL={best_objf:.2f}): No theta file found")
            continue

        ke_vals = []
        v_vals = []
        weights = []
        with open(theta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ke_vals.append(float(row.get("ke", 0)))
                    v_vals.append(float(row.get("v", 0)))
                    weights.append(float(row.get("prob", row.get("w", 0))))
                except (ValueError, KeyError):
                    pass

        if not ke_vals:
            continue

        # Classify support points by ke mode
        low_ke = [(ke, v, w) for ke, v, w in zip(ke_vals, v_vals, weights) if ke < 0.18]
        high_ke = [(ke, v, w) for ke, v, w in zip(ke_vals, v_vals, weights) if ke >= 0.18]

        total_w = sum(weights) if sum(weights) > 0 else 1

        print(f"\n  {alg} (seed {best_seed}, -2LL={best_objf:.2f}, {len(ke_vals)} spp):")
        if low_ke:
            w_sum = sum(w for _, _, w in low_ke)
            ke_mean = sum(ke * w for ke, _, w in low_ke) / w_sum if w_sum > 0 else 0
            print(f"    Mode 1 (slow ke): {len(low_ke)} spp, weight={w_sum / total_w:.1%}, mean ke={ke_mean:.4f}")
        else:
            print(f"    Mode 1 (slow ke): NOT FOUND")
        if high_ke:
            w_sum = sum(w for _, _, w in high_ke)
            ke_mean = sum(ke * w for ke, _, w in high_ke) / w_sum if w_sum > 0 else 0
            print(f"    Mode 2 (fast ke): {len(high_ke)} spp, weight={w_sum / total_w:.1%}, mean ke={ke_mean:.4f}")
        else:
            print(f"    Mode 2 (fast ke): NOT FOUND")

        # Show top 5 support points by weight
        sorted_spp = sorted(zip(ke_vals, v_vals, weights), key=lambda x: -x[2])
        print(f"    Top 5 support points by weight:")
        for i, (ke, v, w) in enumerate(sorted_spp[:5]):
            mode = "slow" if ke < 0.18 else "fast"
            print(f"      #{i+1}: ke={ke:.4f} ({mode}), v={v:.2f}, w={w/total_w:.3%}")


def pairwise_ranking(results):
    """For each pair of algorithms, count how many seeds one beats the other."""
    print()
    print("=" * 80)
    print("PAIRWISE WIN/LOSS MATRIX (row beats column in N/5 seeds)")
    print("=" * 80)

    algs = sorted(results.keys(), key=lambda a: statistics.mean(results[a]["objf"]))
    n_seeds = len(results[algs[0]]["objf"])

    # Header
    header = f"{'':>10} |"
    for a in algs:
        header += f" {a[:5]:>5}"
    print(header)
    print("-" * (13 + 6 * len(algs)))

    for a1 in algs:
        row = f"{a1:>10} |"
        for a2 in algs:
            if a1 == a2:
                row += "     -"
            else:
                wins = sum(1 for o1, o2 in zip(results[a1]["objf"], results[a2]["objf"]) if o1 < o2)
                row += f"   {wins}/5"
            
        print(row)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    results = load_results()
    print_summary(results)
    pairwise_ranking(results)
    analyze_support_points(results)
    analyze_theta_detail(results)
