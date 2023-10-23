import numpy as np
from numba import njit
from utils.io.rxml import read_xml
from utils.aorr.triprepr import trip_lookup, lookup2trip, trip_lookup_precedence
from utils.aorr.tripattr import get_trip_num, get_neighbors, get_demand, get_trip_dmd, get_route_pos, get_trip_len
from utils.algorithm.memetic.localsearch.lsoperator.relocate import m1_cost_inter, do_m1_inter, do_m1_intra


def do_ls_inter_m1_test(n, test_lookup, test_lookup_prev, test_lookup_next, q, trip_dmd, test_trip_num, max_route_len,
                        trip, c, trip_benchmark, n_row, w, neighbor, trip_total, idx, tol=1e-4):
    """
    test the correctness of m1 operations
    """
    num_neighbor = len(neighbor)
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
                    np.savetxt("lookup_ori", test_lookup)
                    np.savetxt("lookup_prev_ori", test_lookup_prev)
                    np.savetxt("lookup_next_ori", test_lookup_next)
                    np.savetxt("trip_ori", trip_benchmark)
                    np.savetxt("u", np.array([u]))
                    np.savetxt("v", np.array([v]))
                    np.savetxt("r2", np.array([r2]))
                    np.savetxt("gain", np.array([gain]))
                    do_m1_inter(r1, r2, pos2, test_lookup, trip_dmd, u_dmd, test_trip_num, test_lookup_prev,
                                test_lookup_next, u_prev, u, x, v, y)
                    new_trip = lookup2trip(test_lookup, max_route_len, len(trip))
                    new_lookup_prev, new_lookup_next = trip_lookup_precedence(new_trip, test_trip_num, n)

                    trip_benchmark[r1] = np.append(np.delete(trip_benchmark[r1], pos1), 0)
                    # tmp_route = trip_benchmark[r2]
                    # tmp_route[pos2 + 1:] = tmp_route[pos2:-1]
                    # tmp_route[pos2 + 1] = u
                    trip_benchmark[r2] = np.insert(trip_benchmark[r2], pos2 + 1, u)[:-1]
                    np.savetxt("lookup_cgd", test_lookup)
                    np.savetxt("lookup_prev_cgd", test_lookup_prev)
                    np.savetxt("lookup_next_cgd", test_lookup_next)
                    np.savetxt("trip_cgd", trip_benchmark)
                    np.savetxt("new_trip", new_trip)
                    np.savetxt("lookup_prev_bench", new_lookup_prev)
                    np.savetxt("lookup_next_bench", new_lookup_next)
                    if np.sum(trip_benchmark) != trip_total:
                        raise ValueError("Missing customer in trip benchmark")
                    assert abs(get_trip_len(c, trip_benchmark) - get_trip_len(c, new_trip)) <= 1e-4
                    # assert np.allclose(new_trip[:n_row], trip_benchmark[:n_row])
                    assert np.allclose(new_lookup_prev[1:], test_lookup_prev[1:])
                    assert np.allclose(new_lookup_next[1:], test_lookup_next[1:])
                    return gain
        else:
            if u != y and v != x:
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > tol:
                    np.savetxt("lookup_ori", test_lookup)
                    np.savetxt("lookup_prev_ori", test_lookup_prev)
                    np.savetxt("lookup_next_ori", test_lookup_next)
                    np.savetxt("trip_ori", trip_benchmark)
                    np.savetxt("u", np.array([u]))
                    np.savetxt("v", np.array([v]))
                    np.savetxt("r2", np.array([r2]))
                    np.savetxt("gain", np.array([gain]))
                    do_m1_intra(pos1, pos2, u_prev, u, x, v, y, test_lookup, test_lookup_next, test_lookup_prev)
                    new_trip = lookup2trip(test_lookup, max_route_len, len(trip))
                    new_lookup_prev, new_lookup_next = trip_lookup_precedence(new_trip, test_trip_num, n)

                    u_removed_route = np.delete(trip_benchmark[r1], pos1)
                    if pos1 < pos2:
                        # tmp_route = np.empty(len(trip_benchmark[r2]))
                        # tmp_route[:pos2] = u_removed_route[:pos2]
                        # tmp_route[pos2] = u
                        # tmp_route[pos2 + 1:] = u_removed_route[pos2:]
                        trip_benchmark[r2] = np.insert(u_removed_route, pos2, u)
                    else:
                        # tmp_route = np.empty(len(trip_benchmark[r2]))
                        # pos = pos2 + 1
                        # tmp_route[:pos] = u_removed_route[:pos]
                        # tmp_route[pos] = u
                        # tmp_route[pos + 1:] = u_removed_route[pos:]
                        trip_benchmark[r2] = np.insert(u_removed_route, pos2 + 1, u)
                    np.savetxt("lookup_cgd", test_lookup)
                    np.savetxt("lookup_prev_cgd", test_lookup_prev)
                    np.savetxt("lookup_next_cgd", test_lookup_next)
                    np.savetxt("trip_cgd", trip_benchmark)
                    np.savetxt("new_trip", new_trip)
                    np.savetxt("lookup_prev_bench", new_lookup_prev)
                    np.savetxt("lookup_next_bench", new_lookup_next)
                    if np.sum(trip_benchmark) != trip_total:
                        raise ValueError("Missing customer in trip benchmark")
                    assert abs(get_trip_len(c, trip_benchmark) - get_trip_len(c, new_trip)) <= 1e-4
                    # assert np.allclose(new_trip[:n_row], trip_benchmark[:n_row])
                    assert np.allclose(new_lookup_prev[1:], test_lookup_prev[1:])
                    assert np.allclose(new_lookup_next[1:], test_lookup_next[1:])
                    return gain
    # relocate into empty trip
    for r2 in range(len(test_trip_num)):
        if not test_trip_num[r2]:
            v = 0
            print("Performing empty route relocate")
            for u in range(1, n + 1):
                u_prev, x, x_post, v_prev, y, y_post = get_neighbors(test_lookup_prev, test_lookup_next, u, v)
                r1, pos1 = test_lookup[u]
                pos2 = -1
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > tol:
                    u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
                    np.savetxt("lookup_ori", test_lookup)
                    np.savetxt("lookup_prev_ori", test_lookup_prev)
                    np.savetxt("lookup_next_ori", test_lookup_next)
                    np.savetxt("trip_ori", trip_benchmark)
                    np.savetxt("u", np.array([u]))
                    np.savetxt("r2", np.array([r2]))
                    np.savetxt("gain", np.array([gain]))
                    do_m1_inter(r1, r2, pos2, test_lookup, trip_dmd, u_dmd, test_trip_num, test_lookup_prev,
                                test_lookup_next, u_prev, u, x, v, y)
                    new_trip = lookup2trip(test_lookup, max_route_len, len(trip))
                    new_lookup_prev, new_lookup_next = trip_lookup_precedence(new_trip, test_trip_num, n)

                    trip_benchmark[r1] = np.append(np.delete(trip_benchmark[r1], pos1), 0)
                    # tmp_route = trip_benchmark[r2]
                    # tmp_route[pos2 + 1:] = tmp_route[pos2:-1]
                    # tmp_route[pos2 + 1] = u
                    trip_benchmark[r2] = np.insert(trip_benchmark[r2], pos2 + 1, u)[:-1]
                    np.savetxt("lookup_cgd", test_lookup)
                    np.savetxt("lookup_prev_cgd", test_lookup_prev)
                    np.savetxt("lookup_next_cgd", test_lookup_next)
                    np.savetxt("trip_cgd", trip_benchmark)
                    np.savetxt("lookup_prev_bench", new_lookup_prev)
                    np.savetxt("lookup_next_bench", new_lookup_next)
                    if np.sum(trip_benchmark) != trip_total:
                        raise ValueError("Missing customer in trip benchmark")
                    assert abs(get_trip_len(c, trip_benchmark) - get_trip_len(c, new_trip)) <= 1e-4
                    # assert np.allclose(new_trip[:n_row], trip_benchmark[:n_row])
                    assert np.allclose(new_lookup_prev[1:], test_lookup_prev[1:])
                    assert np.allclose(new_lookup_next[1:], test_lookup_next[1:])
                    return gain
            break
    return 0.


def do_ls_empty_m1():
    pass


def get_test_dist_mat(n: int):
    dist_mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            dist_mat[i, j] = max(100 * i + j, 100 * j + i)
    return dist_mat


def get_test_neighbor(n):
    res = np.empty((n * (n - 1), 2), dtype=int)
    count = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                res[count] = i, j
                count += 1
    return res
