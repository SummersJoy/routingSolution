import numpy as np
from numba import njit, int32
from utils.algorithm.memetic.ga.gautils import fill_chromosome, get_new_ind, descend


@njit(fastmath=True, cache=True)
def lox(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    TSP like crossover operator
    """
    n = len(p1)
    pos1 = np.random.randint(0, n)
    pos2 = np.random.randint(0, n)
    c1 = np.empty(n, dtype=int32)
    c2 = np.empty(n, dtype=int32)
    while pos2 == pos1:
        pos2 = np.random.randint(0, n)
    i = min(pos1, pos2)
    j = max(pos1, pos2)
    c1[i:j] = p1[i:j]
    c2[i:j] = p2[i:j]
    fill_chromosome(p1, p2, c1, i, j, n)
    fill_chromosome(p2, p1, c2, i, j, n)
    return c1, c2


@njit(fastmath=True, cache=True)
def get_initial_solution(n, size, q, d, c, w, max_load, delta, heuristic_sol):
    res = np.empty((size, n + 1), dtype=int32)
    num_heu_sol = len(heuristic_sol)
    ind_fitness = np.empty(size)
    res[:num_heu_sol] = heuristic_sol
    for i in range(num_heu_sol):
        _, fitness = split(n, heuristic_sol[i], q, d, c, w, max_load)
        ind_fitness[i] = fitness
    restart = 0
    for i in range(num_heu_sol, size):
        s = get_new_ind(n)
        _, fitness = split(n, s, q, d, c, w, max_load)
        while True:
            well_spaced = True
            for j in range(i):
                if abs(fitness - ind_fitness[j]) < delta:
                    well_spaced = False
                    break
            if well_spaced:
                ind_fitness[i] = fitness
                break
            else:
                restart += 1
                s = get_new_ind(n)
                _, fitness = split(n, s, q, d, c, w, max_load)
        res[i] = s

    return res, ind_fitness, restart


@njit(fastmath=True, cache=True)
def split(n: int, s: np.ndarray, q: np.ndarray, d: np.ndarray, c: np.ndarray, w: float, max_load: float) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Bellman algorithm to split a permutation of route into several feasible sub-routes with minimum travel distance
    """
    v = np.empty(n + 1)
    p = np.empty(n + 1, dtype=int32)
    v[0] = 0.
    v[1:] = np.inf
    for i in range(1, n + 1):
        load = 0.
        cost = 0.
        j = i
        while True:
            sj = s[j]
            sj_prev = s[j - 1]
            load += q[sj]
            if i == j:
                cost = c[0, sj] + d[sj] + c[sj, 0]
            else:
                cost = cost - c[sj_prev, 0] + c[sj_prev, sj] + d[sj] + c[sj, 0]
            if load <= w and cost <= max_load:
                if v[i - 1] + cost < v[j]:
                    v[j] = v[i - 1] + cost
                    p[j] = i - 1
                j += 1
            if j > n or load > w or cost > max_load:
                break
    return p, v[-1]


@njit(fastmath=True, cache=True)
def binary_tournament_selection(population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Tournament selection on a sorted population
    """
    n = len(population)
    id1 = np.random.randint(0, n)
    id2 = np.random.randint(0, n)
    father = id1 if id1 < id2 else id2
    id3 = np.random.randint(0, n)
    id4 = np.random.randint(0, n)
    mother = id3 if id3 < id4 else id4
    while father == mother:
        id3 = np.random.randint(0, n)
        id4 = np.random.randint(0, n)
        mother = id3 if id3 < id4 else id4
    return population[father], population[mother]


@njit(fastmath=True, cache=True)
def mutation(n, c, fitness, trip_dmd, q, w, lookup, neighbor, trip_num, lookup_prev, lookup_next, dpf):
    prev = 0.
    while True:
        gain = descend(n, c, trip_dmd, q, w, lookup, neighbor, trip_num, lookup_prev, lookup_next, dpf)
        if gain < 0. or abs(gain - prev) < 1e-4:
            break
        else:
            fitness -= gain
            prev = gain
    return fitness


@njit(fastmath=True, cache=True)
def decoding(trip: np.ndarray, n) -> np.ndarray:
    """
    decode a trip into chromosome
    """
    res = np.empty(n + 1, dtype=int32)
    res[0] = 0
    count = 1
    for route in trip[::-1, :]:
        for cust in route:
            res[count] = cust
            if cust == 0:
                break
            count += 1
    return res


@njit(fastmath=True)
def optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl, h_sol):
    # todo: add artificial penalty function to objective and increase the penalty factor
    print("compiled")
    pool, ind_fit, restart = get_initial_solution(n, size, q, d, c, w, max_dist, delta, h_sol)
    ordered_idx = np.argsort(ind_fit)
    pool = pool[ordered_idx, :]
    ind_fit = ind_fit[ordered_idx]
    neighbor = neighbourhood_gen(cx, cy, max_agl)
    space_hash = np.zeros(500000)
    for sol in pool:
        _, fitness = split(n, sol, q, d, c, w, max_dist)
        hash_idx = int(fitness / delta)
        space_hash[hash_idx] = 1.
    a = 0
    b = 0
    mid = size // 2
    while a != alpha and b != beta:
        # todo: duplicated p1 and p2
        p1, p2 = binary_tournament_selection(pool)  # 910 ns ± 12.7 ns
        child1, child2 = lox(p1[1:], p2[1:])  # 2.82 µs ± 12.9 ns
        child = child1 if np.random.random() < 0.5 else child2  # 266 ns ± 8.91 ns
        child = fill_zero(n, child)
        label, val = split(n, child, q, d, c, w, max_dist)  # 10.6 µs ± 83 ns
        trip = label2route(n, label, child, max_route_len)  # 1.18 µs ± 24.2 ns
        k = np.random.randint(mid, size)
        modified_fitness = np.concatenate((ind_fit[:k], ind_fit[k + 1:]))  # 1.15 µs ± 13.4 ns
        if np.random.random() < pm:
            trip_num = get_trip_num(trip)  # 800 ns ± 6.77 ns
            trip_dmd = get_trip_dmd(trip, q, trip_num)  # 867 ns ± 3.24 ns
            f = val
            lookup = trip_lookup(trip, n)  # 800 ns ± 5.58 ns
            lookup_prev, lookup_next = trip_lookup_precedence(trip, trip_num, n)  # 1.17 µs ± 30.2 ns
            mutation(n, c, val, trip_dmd, q, w, lookup, neighbor, trip_num, lookup_prev, lookup_next)
            # retriv trip
            trip = lookup2trip(lookup, n, max_route_len, len(trip))  # 2.54 µs ± 18.9 ns
            chromosome = decoding(trip, n)  # 732 ns ± 11.6 ns
            _, fitness = split(n, chromosome, q, d, c, w, max_dist)
            is_spaced = check_spaced(space_hash, fitness, delta)  # 187 ns ± 1.46 ns
            if is_spaced:
                child = chromosome
                val = fitness
            else:
                val = f
                is_spaced = check_spaced(space_hash, val, delta)
        else:
            is_spaced = check_spaced(space_hash, val, delta)
        if is_spaced:
            space_hash[int(ind_fit[k] / delta)] = 0.  # remove hashed value from spack_hash
            a += 1
            idx = bisect(modified_fitness, val) + 1  # 432 ns ± 4.61 ns
            if idx == k:
                pool[k] = child
                ind_fit[k] = val
            elif idx < k:
                pool = np.concatenate(
                    (pool[:idx, :], child.reshape((1, n + 1)), pool[idx:k, :], pool[(k + 1):, :]))  # 2.75 µs ± 22.1 ns
                # pool_cpy = pool.copy()
                # pool_cpy[idx + 1: k + 1] = pool_cpy[idx:k]
                # pool_cpy[idx] = child
                ind_fit = np.concatenate((ind_fit[:idx], val * np.ones(1), ind_fit[idx:k], ind_fit[(k + 1):]))
            else:
                idx += 1
                pool = np.concatenate((pool[:k, :], pool[(k + 1):idx, :], child.reshape((1, n + 1)), pool[idx:, :]))
                # todo: efficiency test
                # pool[k:idx, :] = pool[k + 1:idx + 1, :]
                # pool[idx + 1] = child
                ind_fit = np.concatenate((ind_fit[:k], ind_fit[(k + 1):idx], val * np.ones(1), ind_fit[idx:]))
            if idx == 0:  # incumbent solution found
                b = 0
            else:  # stall
                b += 1
    return pool, ind_fit