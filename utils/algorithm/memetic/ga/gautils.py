import numpy as np
from numba import njit, int32
from utils.aorr.tripattr import get_neighbors, get_demand, get_route_pos
from utils.algorithm.memetic.localsearch.lsoperator.relocate import m1_cost_inter, do_m1_inter, do_m1_intra
from utils.algorithm.memetic.localsearch.lsoperator.dblrelocate import m2_cost_inter, do_m2_inter, do_m2_intra
from utils.algorithm.memetic.localsearch.lsoperator.oropt import m3_cost_inter, do_m3_inter, do_m3_intra
from utils.algorithm.memetic.localsearch.lsoperator.cox import m4_cost_inter, do_m4_inter, do_m4_intra
from utils.algorithm.memetic.localsearch.lsoperator.asmox import m5_cost_inter, do_m5_inter, do_m5_intra
from utils.algorithm.memetic.localsearch.lsoperator.dblox import m6_cost_inter, do_m6_inter


@njit(fastmath=True)
def fill_chromosome(p1: np.ndarray, p2: np.ndarray, c1: np.ndarray, i: int, j: int, n: int) -> None:
    """
    iteratively fill elements in each chromosome to achieve LOX
    """
    count = 0
    # p1_present = p1[i:j]
    p1_present = np.zeros(len(p1) + 1, dtype=int32)
    p1_present[p1[i:j]] = 1
    for t in p2[j:]:
        if j + count < n:
            if not p1_present[t]:
                c1[j + count] = t
                count += 1
        else:
            break
    count_f = 0
    for t in p2[:j]:
        if not p1_present[t]:
            if j + count < n:
                c1[j + count] = t
                count += 1
            else:
                c1[count_f] = t
                count_f += 1


@njit(fastmath=True)
def get_new_ind(n):
    tmp = np.random.permutation(n) + 1
    s = np.empty(n + 1, dtype=int32)
    s[0] = 0
    s[1:] = tmp
    return s


@njit(fastmath=True)
def descend(n, lookup, lookup_prev, lookup_next, q, trip_dmd, trip_num, c, w, neighbor, tol=1e-4):
    """
    test the correctness of m1 operations
    """
    num_neighbor = len(neighbor)
    idx = np.random.randint(0, num_neighbor)
    neighbor_cgd = np.empty_like(neighbor)
    neighbor_cgd[:(num_neighbor - idx)] = neighbor[idx:]
    neighbor_cgd[(num_neighbor - idx):] = neighbor[:idx]
    for u, v in neighbor_cgd:
        r1, r2, pos1, pos2 = get_route_pos(lookup, u, v)
        u_prev, x, x_post, v_prev, y, y_post = get_neighbors(lookup_prev, lookup_next, u, v)
        u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
        if r1 != r2:
            if trip_dmd[r2] + u_dmd <= w:
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > tol:
                    do_m1_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, trip_num, lookup_prev, lookup_next, u_prev, u, x,
                                v, y)
                    return gain
            if x and trip_dmd[r2] + u_dmd + x_dmd <= w:
                gain = m2_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain > tol:
                    do_m2_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                u_prev, u, x, x_post, v, y)
                    return gain
                gain = m3_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain > tol:
                    do_m3_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                u_prev, u, x, x_post, v, y)
                    return gain
            if trip_dmd[r1] - u_dmd + v_dmd <= w and trip_dmd[r2] - v_dmd + u_dmd <= w:  # 1:
                gain = m4_cost_inter(c, u_prev, u, x, v_prev, v, y)
                if gain > tol:
                    do_m4_inter(r1, r2, pos1, pos2, u_prev, u, x, v_prev, v, y, lookup, lookup_prev, lookup_next,
                                trip_dmd, u_dmd, v_dmd)
                    return gain
            if x and trip_dmd[r1] - u_dmd - x_dmd + v_dmd <= w and trip_dmd[r2] - v_dmd + u_dmd + x_dmd <= w:
                gain = m5_cost_inter(c, u_prev, u, x, x_post, v_prev, v, y)
                if gain > tol:
                    do_m5_inter(r1, r2, pos1, pos2, lookup, trip_dmd, u_dmd, x_dmd, v_dmd, trip_num, lookup_prev,
                                lookup_next, u_prev, u, x, x_post, v_prev, v, y)
                    return gain
            if x and y:
                gain = m6_cost_inter(c, u_prev, u, x, x_post, v_prev, v, y, y_post)
                if gain > tol:
                    do_m6_inter(r1, r2, pos1, pos2, u_prev, u, x, x_post, v_prev, v, y, y_post, lookup, lookup_prev,
                                lookup_next, trip_dmd, u_dmd, v_dmd, x_dmd, y_dmd)
        else:
            if u != y and v != x:
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > tol:
                    do_m1_intra(pos1, pos2, u_prev, u, x, v, y, lookup, lookup_next, lookup_prev)
                    return gain
            if x and x != v and u != y:  # abs(pos1 - pos2) > 1:
                gain = m2_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain > tol:
                    do_m2_intra(pos1, pos2, u_prev, u, x, x_post, v, y, lookup, lookup_next, lookup_prev)
                    return gain
                gain = m3_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain > tol:
                    do_m3_intra(pos1, pos2, u_prev, u, x, x_post, v, y, lookup, lookup_next, lookup_prev)
                    return gain
            if abs(pos1 - pos2) > 2:  # todo m4 intra when |pos1 - pos2| = 1
                gain4 = m4_cost_inter(c, u_prev, u, x, v_prev, v, y)
                if gain4 > 0:
                    do_m4_intra(pos1, pos2, u_prev, u, x, v_prev, v, y, lookup, lookup_prev, lookup_next)
                    return gain4
            if x and abs(pos1 - pos2) > 2:
                gain = m5_cost_inter(c, u_prev, u, x, x_post, v_prev, v, y)
                if gain > tol:
                    do_m5_intra(pos1, pos2, u_prev, u, x, x_post, v_prev, v, y, lookup, lookup_next, lookup_prev)
                    return gain


    # relocate into empty trip
    for r2 in range(len(trip_num)):
        if not trip_num[r2]:
            v = 0
            for u in range(1, n + 1):
                u_prev, x, x_post, v_prev, y, y_post = get_neighbors(lookup_prev, lookup_next, u, v)
                u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
                r1, pos1 = lookup[u]
                pos2 = -1
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > tol:
                    do_m1_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, trip_num, lookup_prev, lookup_next, u_prev, u, x,
                                v, y)
                    return gain
                gain = m2_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain > tol:
                    do_m2_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                u_prev, u, x, x_post, v, y)
                    return gain
                gain = m3_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain > tol:
                    do_m3_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                u_prev, u, x, x_post, v, y)
                    return gain
            break
    return 0.


# todo: return 0 and 1 instead of True and False, timeit
@njit(fastmath=True)
def check_spaced(space_hash: np.ndarray, val: float, delta: float) -> bool:
    """
    check if new chromosome is well-spaced in the population in O(1) time
    """
    idx = int32(val / delta)
    if space_hash[idx]:
        return False
    else:
        space_hash[idx] = 1.
        return True


@njit
def generation_management(size, sol_count, sol_pool, fit_pool, delta, n):
    idx_sort = np.argsort(fit_pool[:sol_count])

    target_pool = np.empty((size, n + 1), dtype=int32)
    target_fit = np.empty(size)

    target_spaced = np.zeros(500000)
    # permutation = np.random.permutation(sol_count)
    # fill first 3 elements
    target_count = 0
    for i in range(sol_count):
        idx = idx_sort[np.random.randint(0, 80)]
        if not target_spaced[int(fit_pool[idx] / delta)]:
            target_spaced[int(fit_pool[idx] / delta)] = 1
            target_pool[target_count] = sol_pool[idx]
            target_fit[target_count] = fit_pool[idx]
            if target_count == 2:
                break
            else:
                target_count += 1

    for i in range(sol_count):
        mid = sol_count // 2
        idx = idx_sort[np.random.randint(max(30, mid - 100), min(mid + 100, sol_count))]
        if not target_spaced[int(fit_pool[idx] / delta)]:
            target_spaced[int(fit_pool[idx] / delta)] = 1
            target_pool[target_count] = sol_pool[idx]
            target_fit[target_count] = fit_pool[idx]
            if target_count == size - 5:
                break
            else:
                target_count += 1

    for i in range(sol_count):
        idx = idx_sort[np.random.randint(sol_count - 100, sol_count - 1)]
        if not target_spaced[int(fit_pool[idx] / delta)]:
            target_spaced[int(fit_pool[idx] / delta)] = 1
            target_pool[target_count] = sol_pool[idx]
            target_fit[target_count] = fit_pool[idx]
            if target_count == size - 1:
                break
            else:
                target_count += 1
    return target_pool, target_fit
