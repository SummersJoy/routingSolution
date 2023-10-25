import numpy as np
from utils.io.rvrplib import read_vrp, parse_vrplib
from utils.io.rjson import config_feature, config_ga
from utils.algorithm.memetic.ga.gaoperator import optimize, get_new_ind, split, get_initial_solution
from utils.aorr.tripattr import get_max_route_len, fill_zero
from utils.visualize.sol import plot_sol
from utils.aorr.triprepr import label2route
from utils.run_cached import run

dataset = "Vrp-Set-X"
root = f"D:/vrplib/{dataset}"
instance_name = "X-n129-k18"
fmt = "vrp"
cached_pool_path = f"./data/cache/{dataset}/{instance_name}_pool.npy"
cached_fitness_path = f"./data/cache/{dataset}/{instance_name}_fitness.npy"
instance, solution = read_vrp(root, instance_name, fmt)
cx, cy, q, w, c, n, d = parse_vrplib(instance)
do_sa, temp, factor, perturbation_size, div_lim = config_feature("./param/feature.json")
# parameters
pm, size, max_dist, alpha, beta, delta, rho, max_agl = config_ga("./param/ga.json")
max_route_len = get_max_route_len(q, w)

best_sol = run(dataset, instance_name, cached_pool_path, cached_fitness_path, pm, size, max_dist, cx, cy, q, w, c, n, d,
               delta, do_sa, temp, factor, perturbation_size, div_lim, max_route_len)
label, val = split(n, best_sol, q, d, c, w, max_dist)
trip = label2route(n, label, best_sol, max_route_len)
plot_sol(cx, cy, trip, val, instance_name)
