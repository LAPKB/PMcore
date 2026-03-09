#!/usr/bin/env python3
import csv
from collections import defaultdict
import statistics

data = defaultdict(list)
with open('/Users/siel/code/LAPKB/PMcore/examples/paper_benchmarks/results_1770330847.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        alg = row['algorithm']
        data[alg].append({
            'objf': float(row['objf']),
            'cycles': int(row['cycles']),
            'time': float(row['time_secs']),
            'spp': int(row['n_spp']),
            'seed': int(row['seed'])
        })

print('=' * 120)
print(f'{"Algorithm":<10} {"Mean -2LL":>12} {"SD":>8} {"Best":>12} {"Worst":>12} {"Range":>8} {"Mean Cyc":>10} {"Mean Time":>10} {"Mean SPP":>10}')
print('=' * 120)

ranked = sorted(data.items(), key=lambda x: statistics.mean([r['objf'] for r in x[1]]))
for alg, runs in ranked:
    objfs = [r['objf'] for r in runs]
    cycles = [r['cycles'] for r in runs]
    times = [r['time'] for r in runs]
    spps = [r['spp'] for r in runs]
    m = statistics.mean(objfs)
    sd = statistics.stdev(objfs)
    best = min(objfs)
    worst = max(objfs)
    print(f'{alg:<10} {m:>12.2f} {sd:>8.2f} {best:>12.2f} {worst:>12.2f} {worst-best:>8.2f} {statistics.mean(cycles):>10.1f} {statistics.mean(times):>10.2f} {statistics.mean(spps):>10.1f}')

print()
print('Per-seed best algorithm:')
for seed in [42, 123, 456, 789, 1001]:
    best_alg = None
    best_objf = float('inf')
    for alg, runs in data.items():
        for r in runs:
            if r['seed'] == seed and r['objf'] < best_objf:
                best_objf = r['objf']
                best_alg = alg
    print(f'  Seed {seed}: {best_alg} ({best_objf:.4f})')

print()
print('Win count (best -2LL per seed):')
wins = defaultdict(int)
for seed in [42, 123, 456, 789, 1001]:
    best_alg = None
    best_objf = float('inf')
    for alg, runs in data.items():
        for r in runs:
            if r['seed'] == seed and r['objf'] < best_objf:
                best_objf = r['objf']
                best_alg = alg
    wins[best_alg] += 1
for alg, w in sorted(wins.items(), key=lambda x: -x[1]):
    print(f'  {alg}: {w} wins')

print()
print('Efficiency ratio (mean -2LL / mean time):')
for alg, runs in ranked:
    objfs = [r['objf'] for r in runs]
    times = [r['time'] for r in runs]
    m = statistics.mean(objfs)
    t = statistics.mean(times)
    print(f'  {alg:<10} {m/t:>10.2f} -2LL/sec')
