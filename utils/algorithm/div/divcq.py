import numpy as np
from utils.algorithm.memetic.ga.gaoperator import split, optimize, decoding
from utils.algorithm.memetic.ga.gautils import get_new_ind
from utils.aorr.triprepr import label2route
from utils.aorr.tripattr import get_trip_len, fill_zero


def divide_conquer(instance, cached_pool_path, cached_fitness_path, cx, cy, n, q, d, c, w, max_dist, max_route_len,
                   size, pm, alpha, beta, delta, max_agl, do_sa, temp, factor, perturbation_size, div_lim):
    """
    re-optimize a subset of large instance
    """

    cached_sol = np.loadtxt(cached_pool_path).astype(int)
    cached_fit = np.loadtxt(cached_fitness_path)
    best_idx = np.argmin(cached_fit)
    best_sol = cached_sol[best_idx]
    best_fit = cached_fit[best_idx]
    print(f"initial fitness: {best_fit}")
    label, initial_fit = split(n, best_sol, q, d, c, w, max_dist)
    print(f"actual initial fit: {initial_fit}")
    trip = label2route(n, label, best_sol, max_route_len)
    for idx in range(div_lim):
        route_id = np.random.choice(len(trip), perturbation_size, replace=False)
        trip_len_ori = get_trip_len(c, trip[route_id])
        seq = get_subp(n, route_id, trip)
        sub_n, sub_q, sub_c, sub_d, sub_cx, sub_cy = get_subpar(seq, q, c, d, cx, cy)

        initial_sol, initial_fit = get_random_initial_sol(size, sub_n, sub_q, sub_d, sub_c, w, max_dist)

        pool, ind_fit = optimize(sub_cx, sub_cy, max_route_len, sub_n, sub_q, sub_d, sub_c, w, max_dist, size, pm,
                                 alpha, beta, delta, max_agl, do_sa, temp, factor, initial_sol, initial_fit)
        if ind_fit[0] < trip_len_ori - 1e-6:
            print(f"Instance {instance}, Incumbent solution found by divide and conquer, "
                  f"gain: {trip_len_ori - ind_fit[0]}")
            sol = pool[0]
            label, val = split(sub_n, sol, sub_q, sub_d, sub_c, w, max_dist)
            assert val == ind_fit[0]
            sub_trip_inc = label2route(sub_n, label, sol, max_route_len)
            for j in range(len(sub_trip_inc)):
                sub_trip_inc[j] = seq[sub_trip_inc[j]]
            assert np.sum(sub_trip_inc) == np.sum(seq)
            assert abs(get_trip_len(c, sub_trip_inc) - ind_fit[0]) < 1e-4
            if len(sub_trip_inc) > perturbation_size:
                empty_idx = []
                nonempty_idx = []
                for k, v in enumerate(sub_trip_inc[:, 0]):
                    if not v:
                        empty_idx.append(k)
                    else:
                        nonempty_idx.append(k)
                num_trim = perturbation_size - len(nonempty_idx)
                if num_trim >= 0:
                    for k in range(num_trim):
                        nonempty_idx.append(empty_idx[k])
                    sub_trip_inc = sub_trip_inc[nonempty_idx]
                    trip[route_id] = sub_trip_inc
                else:
                    trip[route_id] = sub_trip_inc[:perturbation_size]
                    trip = np.concatenate((trip, sub_trip_inc[perturbation_size:]))
            # seek for better split
            sol = decoding(trip, n)
            best_sol = sol
            label, best_fit = split(n, sol, q, d, c, w, max_dist)
            assert np.sum(sol) == np.sum(np.arange(n + 1))
    return best_sol, best_fit
    # label, val = split(n, sol, q, d, c, w, max_dist)
    # trip = label2route(n, label, sol, max_route_len)
    # print(get_trip_len(c, trip))


def get_subp(n, route_id, trip):
    seq = np.zeros(n + 1, dtype=int)
    count = 0
    for rid in route_id:
        for cust in trip[rid]:
            if cust:
                seq[count] = cust
                count += 1
            else:
                break
    seq = fill_zero(count, seq[:count])
    return seq


def get_subpar(seq, q, c, d, cx, cy):
    sub_n = len(seq) - 1
    sub_q = q[seq]
    sub_c = c[seq][:, seq]
    sub_d = d[seq]
    sub_cx = cx[seq]
    sub_cy = cy[seq]
    return sub_n, sub_q, sub_c, sub_d, sub_cx, sub_cy


def get_random_initial_sol(size, sub_n, sub_q, sub_d, sub_c, w, max_dist):
    initial_sol = np.empty((size, sub_n + 1), dtype=int)
    initial_fit = np.empty(size)
    for i in range(size):
        ind = get_new_ind(sub_n)
        initial_sol[i] = ind
        label, val = split(sub_n, ind, sub_q, sub_d, sub_c, w, max_dist)
        initial_fit[i] = val
    return initial_sol, initial_fit
