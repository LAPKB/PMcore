#!/usr/bin/env python3
import csv
from collections import defaultdict
import statistics

data = defaultdict(list)
with open('/Users/siel/code/LAPKB/PMcore/examples/paper_benchmarks/results_1770333982.csv') as f:
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
    sd = statistics.stdev(objfs) if len(objfs) > 1 else 0
    best = min(objfs)
    worst = max(objfs)
    print(f'{alg:<10} {m:>12.4f} {sd:>8.4f} {best:>12.4f} {worst:>12.4f} {worst-best:>8.4f} {statistics.mean(cycles):>10.1f} {statistics.mean(times):>10.2f} {statistics.mean(spps):>10.1f}')
