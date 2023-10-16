import numpy as np
from numba import njit, int32
from utils.aorr.tripattr import get_trip_num, get_neighbors, get_demand, get_trip_dmd, get_route_pos, get_trip_len
from utils.algorithm.memetic.localsearch.lsoperator.single_relocate import m1_cost_inter, do_m1_inter, do_m1_intra


@njit(fastmath=True, cache=True)
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


@njit(fastmath=True, cache=True)
def get_new_ind(n):
    tmp = np.random.permutation(n) + 1
    s = np.empty(n + 1, dtype=int32)
    s[0] = 0
    s[1:] = tmp
    return s


@njit(fastmath=True, cache=True)
def descend(n, test_lookup, test_lookup_prev, test_lookup_next, q, trip_dmd, test_trip_num, c, w, neighbor,
            tol=1e-4):
    """
    test the correctness of m1 operations
    """
    for u, v in neighbor:
        r1, r2, pos1, pos2 = get_route_pos(test_lookup, u, v)
        u_prev, x, x_post, v_prev, y, y_post = get_neighbors(test_lookup_prev, test_lookup_next, u, v)
        u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
        if r1 != r2:
            if trip_dmd[r2] + u_dmd <= w:
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > tol:
                    do_m1_inter(r1, r2, pos2, test_lookup, trip_dmd, u_dmd, test_trip_num, test_lookup_prev,
                                test_lookup_next, u_prev, u, x, v, y)
                    return gain
        else:
            if u != y and v != x:
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > tol:
                    do_m1_intra(pos1, pos2, u_prev, u, x, v, y, test_lookup, test_lookup_next, test_lookup_prev)
                    return gain
    # relocate into empty trip
    for r2 in range(len(test_trip_num)):
        if not test_trip_num[r2]:
            v = 0
            for u in range(1, n + 1):
                u_prev, x, x_post, v_prev, y, y_post = get_neighbors(test_lookup_prev, test_lookup_next, u, v)
                r1, pos1 = test_lookup[u]
                pos2 = -1
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > tol:
                    u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
                    do_m1_inter(r1, r2, pos2, test_lookup, trip_dmd, u_dmd, test_trip_num, test_lookup_prev,
                                test_lookup_next, u_prev, u, x, v, y)

                    return gain
            break
    return 0.
