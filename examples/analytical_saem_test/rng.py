import random
import csv

trial_iterations: int = 1000
seed = 632545

random.seed(seed)

header = ['ka', 'ke', 'v', 'trial_id']
data = []

for i in range(trial_iterations):
    data.append([random.uniform(0.0, 1.83*2), random.uniform(0.0, 30.5*2), random.uniform(0.0, 0.075*2), i])

with open('examples/analytical_saem_test/random_init.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # 1. Write the header row
    writer.writerow(header)
    
    # 2. Write all data rows at once
    writer.writerows(data)