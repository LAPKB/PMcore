import csv

file = "examples/analytical_saem_test/logs/pmcore.log"
csv_file = "examples/analytical_saem_test/logs/pmcore_converted.csv"

iteration_list: list[int] = []
objf_list: list[float] = []
ka_list: list[float] = []
ke_list: list[float] = []
v_list: list[float] = []

with open(file, 'r') as file:
    for line in file:
        iteration_list.append(int(line.split("iteration: ")[1].split(" objf: ")[0]))
        objf_list.append(float(line.split("objf: ")[1].split(" pop mu: ")[0]))
        mu: list[float] = [float(i) for i in line.split("pop mu: ")[1][1:-2].split(", ")]
        ka_list.append(mu[0])
        ke_list.append(mu[1])
        v_list.append(mu[2])

rows = zip(iteration_list, objf_list, ka_list, ke_list, v_list)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['Iteration', 'Objf', 'Ka', 'Ke', 'Vol']) # Header
    writer.writerows(rows)