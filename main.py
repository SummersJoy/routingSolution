import numpy as np
import time
import os
from utils.io.rxml import read_xml, get_file_path
from utils.manipulate import reformat_depot, get_dist_mat
from utils.aorr.tripattr import get_max_route_len, fill_zero
from utils.aorr.triprepr import label2route
from utils.algorithm.memetic.ga.gaoperator import optimize, get_new_ind, split, get_initial_solution
from utils.algorithm.memetic.ga.gautils import generation_management
from utils.visualize.sol import plot_sol

# parameters
pm = 0.05
size = 30
max_dist = 10000
alpha = 30000
beta = 10000
delta = 0.5
rho = 16  # number of restarts
max_agl = 22.5  # angle threshold
root = "C:\\Users\\shiyao\\Downloads"
dataset = "christofides"
# dataset = "golden"
instance = "CMT03"
# instance = "Golden_02"
filename = get_file_path(root, dataset, instance)
cached_pool_path = f"./data/cache/{dataset}/{instance}_pool.npy"
cached_fitness_path = f"./data/cache/{dataset}/{instance}_fitness.npy"
cx, cy, q, w, depot = read_xml(filename)
cx = reformat_depot(cx)
cy = reformat_depot(cy)
c = np.round(get_dist_mat(cx, cy), 3)
n = len(cx) - 1
d = np.zeros(n)
q = fill_zero(n, q)
d = fill_zero(n, d)
max_route_len = get_max_route_len(q, w)

heuristic_sol = np.empty((3, n + 1), dtype=int)
for i in range(3):
    ind = get_new_ind(n)
    heuristic_sol[i] = ind
h_sol = heuristic_sol
# maintain a set of good solutions
sol_size = 10000
sol_pool = np.empty((sol_size, n + 1), dtype=int)
fit_pool = np.empty(sol_size)
sol_spaced = np.zeros(500000, dtype=int)
best_val = np.inf
best_sol = None
sol_count = 0
# todo: split algorithm does not match with the solution after mutation!
if os.path.exists(cached_pool_path) and os.path.exists(cached_fitness_path):
    cached_pool = np.loadtxt(cached_pool_path)
    cached_fitness = np.loadtxt(cached_fitness_path)
    sol_count = len(cached_pool)
    sol_pool[:sol_count] = cached_pool
    fit_pool[:sol_count] = cached_fitness
    initial_sol, initial_fit = generation_management(size, sol_count, sol_pool, fit_pool, delta, n)
else:
    initial_sol, initial_fit, _ = get_initial_solution(n, size, q, d, c, w, max_dist, delta, heuristic_sol)

for i in range(1000):
    max_agl = 45. if i % 10 else 180.
    alpha = 9000 if i % 31 else 30000
    beta = 3000 if i % 31 else 10000
    pool, ind_fit = optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl,
                             initial_sol, initial_fit)
    # s = pool[0]
    # label, fitness = split(n, s, q, d, c, w, max_dist)
    # trip = label2route(n, label, s, max_route_len)
    print(ind_fit[0])
    for idx, fit in enumerate(ind_fit):
        if fit < best_val:
            if not sol_spaced[int(fit / delta)]:
                sol_pool[sol_count] = pool[idx]
                fit_pool[sol_count] = ind_fit[idx]
                sol_spaced[int(fit / delta)] = 1
                sol_count += 1
        elif not sol_spaced[int(fit / delta)]:
            sol_pool[sol_count] = pool[idx]
            fit_pool[sol_count] = ind_fit[idx]
            sol_spaced[int(fit / delta)] = 1
            sol_count += 1
    if ind_fit[0] < best_val:
        best_val = ind_fit[0]
        best_sol = pool[0]
        print(f"Incumbent solution found : {np.round(best_val, 3)}")
    else:
        heuristic_sol = pool[:3]
        print(f"Not improved in iteration {i}, current best:{np.round(best_val, 3)}")
    if sol_count > 500:
        initial_sol, initial_fit = generation_management(size, sol_count, sol_pool, fit_pool, delta, n)
        rand_sol, rand_fit, _ = get_initial_solution(n, size, q, d, c, w, max_dist, delta, heuristic_sol)
        initial_sol[(size - 10):-1] = rand_sol[(size - 10):-1]
        initial_fit[(size - 10):-1] = rand_fit[(size - 10):-1]
    else:
        initial_sol, initial_fit, _ = get_initial_solution(n, size, q, d, c, w, max_dist, delta, heuristic_sol)
    np.savetxt(cached_pool_path, sol_pool[:sol_count])
    np.savetxt(cached_fitness_path, fit_pool[:sol_count])

label, val = split(n, best_sol, q, d, c, w, max_dist)
trip = label2route(n, label, best_sol, max_route_len)
plot_sol(cx, cy, trip, val, instance)
