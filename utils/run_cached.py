import numpy as np
import json
import os
import time
from utils.algorithm.memetic.ga.gaoperator import optimize, get_new_ind, split, get_initial_solution
from utils.algorithm.memetic.ga.gautils import generation_management
from utils.algorithm.div.divcq import divide_conquer
from utils.algorithm.memetic.localsearch.neighbor import get_angle


def run(dataset, instance, cached_pool_path, cached_fitness_path, pm, size, max_dist, cx, cy, q, w, c, n, d, delta,
        do_sa, temp, factor, perturbation_size, div_lim, max_route_len):
    bks = json.load(open("param/bks.json"))
    angle = get_angle(cx, cy)
    angle_max = max(angle)
    heuristic_sol = np.empty((3, n + 1), dtype=int)
    for i in range(3):
        ind = get_new_ind(n)
        heuristic_sol[i] = ind
    h_sol = heuristic_sol
    # maintain a set of good solutions
    sol_size = 100000
    sol_pool = np.empty((sol_size, n + 1), dtype=int)
    fit_pool = np.empty(sol_size)
    sol_spaced = np.zeros(50000000, dtype=int)
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
    print(instance)
    instance_bks = bks[dataset][instance]
    start = time.perf_counter()
    for i in range(1000):
        max_agl = angle_max / 16. if i % 11 else angle_max / 8.
        alpha = 9000 if i % 101 else 30000
        beta = 3000 if i % 101 else 10000
        pool, ind_fit = optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl,
                                 do_sa, temp, factor, initial_sol, initial_fit)
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
            print(
                f"Instance {instance}, Incumbent solution found in iteration {i}, "
                f"obj_val : {np.round(best_val, 3)}, bks: {instance_bks}, "
                f"bks gap: {np.round((best_val - instance_bks) * 100 / instance_bks, 3)}%, "
                f"time elapsed: {time.perf_counter() - start}")
        else:
            heuristic_sol = pool[:3]
            print(f"Instance {instance}, Not improved in iteration {i}, current best:{np.round(best_val, 3)}, "
                  f"bks gap: {np.round((best_val - instance_bks) * 100 / instance_bks, 3)}%, "
                  f"this iteration: {np.round(ind_fit[0], 3)}, time elapsed: {time.perf_counter() - start}")
        if sol_count > 300:
            if not i % 10:
                div_sol, div_val = divide_conquer(instance, cached_pool_path, cached_fitness_path, cx, cy, n, q, d, c,
                                                  w, max_dist, max_route_len, size, pm, alpha, beta, delta, max_agl,
                                                  do_sa, temp, factor, perturbation_size, div_lim)
                sol_pool[sol_count] = div_sol
                fit_pool[sol_count] = div_val
                sol_count += 1
                if div_val < best_val:
                    best_val = div_val
                    best_sol = div_sol
            initial_sol, initial_fit = generation_management(size, sol_count, sol_pool, fit_pool, delta, n)
            rand_sol, rand_fit, _ = get_initial_solution(n, size, q, d, c, w, max_dist, delta, heuristic_sol)
            initial_sol[(size - 10):-1] = rand_sol[(size - 10):-1]
            initial_fit[(size - 10):-1] = rand_fit[(size - 10):-1]
        else:
            initial_sol, initial_fit, _ = get_initial_solution(n, size, q, d, c, w, max_dist, delta, heuristic_sol)
        np.savetxt(cached_pool_path, sol_pool[:sol_count])
        np.savetxt(cached_fitness_path, fit_pool[:sol_count])
    return best_sol
