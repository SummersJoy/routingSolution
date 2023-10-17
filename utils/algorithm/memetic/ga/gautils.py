import numpy as np
from numba import njit, int32
from utils.aorr.tripattr import get_trip_num, get_neighbors, get_demand, get_trip_dmd, get_route_pos, get_trip_len
from utils.algorithm.memetic.localsearch.lsoperator.single_relocate import m1_cost_inter, do_m1_inter, do_m1_intra


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
def descend(n, test_lookup, test_lookup_prev, test_lookup_next, q, trip_dmd, test_trip_num, c, w, neighbor, tol=1e-4):
    """
    test the correctness of m1 operations
    """
    num_neighbor = len(neighbor)
    idx = np.random.randint(0, num_neighbor)
    neighbor_cgd = np.empty_like(neighbor)
    neighbor_cgd[:(num_neighbor - idx)] = neighbor[idx:]
    neighbor_cgd[(num_neighbor - idx):] = neighbor[:idx]
    for u, v in neighbor_cgd:
        r1, r2, pos1, pos2 = get_route_pos(test_lookup, u, v)
        u_prev, x, x_post, v_prev, y, y_post = get_neighbors(test_lookup_prev, test_lookup_next, u, v)
        u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
        if r1 != r2:
            if trip_dmd[r2] + u_dmd <= w:
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > tol:
                    do_m1_inter(r1, r2, pos2, test_lookup, trip_dmd, u_dmd, test_trip_num, test_lookup_prev,
                                test_lookup_next, u_prev, u, x, v, y)
                    # new_trip = lookup2trip(test_lookup, max_route_len, len(trip))
                    # new_lookup_prev, new_lookup_next = trip_lookup_precedence(new_trip, test_trip_num, n)
                    #
                    # trip_benchmark[r1] = np.append(np.delete(trip_benchmark[r1], pos1), 0)
                    # trip_benchmark[r2] = np.insert(trip_benchmark[r2], pos2 + 1, u)[:-1]
                    # if np.sum(trip_benchmark) != trip_total:
                    #     raise ValueError("Missing customer in trip benchmark")
                    # assert abs(get_trip_len(c, trip_benchmark) - get_trip_len(c, new_trip)) <= 1e-4
                    # # assert np.allclose(new_trip[:n_row], trip_benchmark[:n_row])
                    # assert np.allclose(new_lookup_prev[1:], test_lookup_prev[1:])
                    # assert np.allclose(new_lookup_next[1:], test_lookup_next[1:])
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
