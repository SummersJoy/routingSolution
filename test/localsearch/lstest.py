import numpy as np
import time
from utils.io.rxml import read_xml
from utils.aorr.triprepr import trip_lookup, lookup2trip, trip_lookup_precedence, label2route
from utils.aorr.tripattr import get_trip_num, get_neighbors, get_demand, get_trip_dmd, get_route_pos, get_trip_len
from test.localsearch.lstestutils import get_test_dist_mat
from utils.algorithm.memetic.localsearch.lsoperator.relocate import m1_cost_inter, do_m1_inter, do_m1_intra
from localsearch.lstestutils import do_ls_inter_m1_test, get_test_neighbor
from utils.algorithm.memetic.ga.gaoperator import split, decoding

# filename = "D:/ga/ga/data/dvrp/christofides/CMT01.xml"
# cx, cy, q, w, depot = read_xml(filename)
n_row = 10
n_col = 10
n = n_row * n_col
c = get_test_dist_mat(n)
q = np.random.randint(1, 20, n + 1)
d = np.zeros(n + 1)
max_dist = np.inf
w = 120.
max_route_len = n
trip = np.zeros((n_row + 1, max_route_len), dtype=int)
for i in range(n_row):
    for j in range(n_col):
        trip[i, j] = i * n_col + j + 1
trip_total = np.sum(np.arange(0, n + 1))
trip_num = get_trip_num(trip)
trip_dmd = get_trip_dmd(trip, q, trip_num)
lookup = trip_lookup(trip, n)
lookup_prev, lookup_next = trip_lookup_precedence(trip, trip_num, n)

great_count = 0
best_obj = np.inf
best_trip = None


# case 1: all pairs of inter route relocate

def test_case1_inter_route_m1():
    """
    relocate to empty route
    conditions:
    1. distinct node pairs (u != v)
    2. distinct route (r1 != r2)
    3. capacity violation (optional)
    """

    for u in range(1, n + 1):
        for v in range(1, n + 1):
            if u != v:
                r1, r2, pos1, pos2 = get_route_pos(lookup, u, v)
                if r1 != r2:
                    test_lookup = trip_lookup(trip, n)
                    test_lookup_prev, test_lookup_next = trip_lookup_precedence(trip, trip_num.copy(), n)
                    test_trip_num = get_trip_num(trip)
                    u_prev, x, x_post, v_prev, y, y_post = get_neighbors(lookup_prev, lookup_next, u, v)
                    gain = m1_cost_inter(c, u_prev, u, x, v, y)
                    u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
                    do_m1_inter(r1, r2, pos2, test_lookup, trip_dmd, u_dmd, test_trip_num, test_lookup_prev,
                                test_lookup_next, u_prev, u, x, v, y)
                    new_trip = lookup2trip(test_lookup, max_route_len, len(trip))
                    new_lookup_prev, new_lookup_next = trip_lookup_precedence(new_trip, test_trip_num, n)

                    trip_benchmark = trip.copy()
                    trip_benchmark[r1] = np.append(np.delete(trip_benchmark[r1], pos1), 0)
                    trip_benchmark[r2] = np.insert(trip_benchmark[r2], pos2 + 1, u)[:-1]
                    assert abs(get_trip_len(c, trip) - gain - get_trip_len(c, new_trip)) <= 1e-4
                    assert np.allclose(new_trip[:n_row], trip_benchmark[:n_row])
                    assert np.allclose(new_lookup_prev[1:], test_lookup_prev[1:])
                    assert np.allclose(new_lookup_next[1:], test_lookup_next[1:])


def test_case2_empty_route_m1():
    """
    relocate to empty route
    :return:
    """
    v = 0
    r2 = n_row
    pos2 = -1
    for u in range(1, n + 1):
        test_lookup = trip_lookup(trip, n)
        test_lookup_prev, test_lookup_next = trip_lookup_precedence(trip, trip_num.copy(), n)
        test_trip_num = get_trip_num(trip)
        r1, pos1 = lookup[u]
        u_prev, x, x_post, v_prev, y, y_post = get_neighbors(test_lookup_prev, test_lookup_next, u, v)
        u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
        gain = m1_cost_inter(c, u_prev, u, x, v, y)
        do_m1_inter(r1, r2, pos2, test_lookup, trip_dmd, u_dmd, test_trip_num, test_lookup_prev, test_lookup_next,
                    u_prev, u, x, v, y)
        new_trip = lookup2trip(test_lookup, max_route_len, len(trip))
        new_lookup_prev, new_lookup_next = trip_lookup_precedence(new_trip, test_trip_num, n)
        trip_benchmark = trip.copy()
        trip_benchmark[r1] = np.append(np.delete(trip_benchmark[r1], pos1), 0)
        trip_benchmark[n_row, 0] = u
        assert abs(get_trip_len(c, trip) - gain - get_trip_len(c, new_trip)) <= 1e-4
        assert np.allclose(new_trip[:n_row], trip_benchmark[:n_row])
        assert np.allclose(new_lookup_prev[1:], test_lookup_prev[1:])
        assert np.allclose(new_lookup_next[1:], test_lookup_next[1:])


def test_case3_inter_route_all_m1():
    test_lookup = trip_lookup(trip, n)
    test_lookup_prev, test_lookup_next = trip_lookup_precedence(trip, trip_num.copy(), n)
    test_trip_num = get_trip_num(trip)
    trip_benchmark = trip.copy()
    for u in range(1, n + 1):
        for v in range(1, n + 1):
            if u != v:
                r1, r2, pos1, pos2 = get_route_pos(test_lookup, u, v)
                if r1 != r2:
                    u_prev, x, x_post, v_prev, y, y_post = get_neighbors(test_lookup_prev, test_lookup_next, u, v)
                    gain = m1_cost_inter(c, u_prev, u, x, v, y)
                    u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
                    do_m1_inter(r1, r2, pos2, test_lookup, trip_dmd, u_dmd, test_trip_num, test_lookup_prev,
                                test_lookup_next, u_prev, u, x, v, y)
                    new_trip = lookup2trip(test_lookup, max_route_len, len(trip))
                    new_lookup_prev, new_lookup_next = trip_lookup_precedence(new_trip, test_trip_num, n)

                    trip_benchmark[r1] = np.append(np.delete(trip_benchmark[r1], pos1), 0)
                    trip_benchmark[r2] = np.insert(trip_benchmark[r2], pos2 + 1, u)[:-1]
                    if np.sum(trip_benchmark) != trip_total:
                        raise ValueError("Missing customer in trip benchmark")
                    assert abs(get_trip_len(c, trip_benchmark) - get_trip_len(c, new_trip)) <= 1e-4
                    assert np.allclose(new_trip[:n_row], trip_benchmark[:n_row])
                    assert np.allclose(new_lookup_prev[1:], test_lookup_prev[1:])
                    assert np.allclose(new_lookup_next[1:], test_lookup_next[1:])


def test_case4_inter_route_descend():
    """
    relocate to empty route
    :return:
    """
    sol = np.random.permutation(n) + 1
    sol = np.insert(sol, 0, 0)
    label, obj_val = split(n, sol, q, d, c, w, max_dist)
    trip = label2route(n, label, sol, max_route_len - 1)
    test_lookup = trip_lookup(trip, n)
    test_trip_num = get_trip_num(trip)
    test_lookup_prev, test_lookup_next = trip_lookup_precedence(trip, test_trip_num.copy(), n)
    test_trip_dmd = get_trip_dmd(trip, q, test_trip_num)
    trip_benchmark = trip.copy()
    neighbor = get_test_neighbor(n)

    fitness = get_trip_len(c, trip)
    idx = np.random.randint(0, len(neighbor))
    gain = do_ls_inter_m1_test(n, test_lookup, test_lookup_prev, test_lookup_next, q, test_trip_dmd, test_trip_num,
                               max_route_len, trip, c, trip_benchmark, n_row, w, neighbor, trip_total, idx)
    fitness -= gain
    count = 1
    while gain > 0:
        idx = np.random.randint(0, len(neighbor))
        gain = do_ls_inter_m1_test(n, test_lookup, test_lookup_prev, test_lookup_next, q, test_trip_dmd,
                                   test_trip_num, max_route_len, trip, c, trip_benchmark, n_row, w, neighbor,
                                   trip_total, idx)
        fitness -= gain
        count += 1

    trip_benchmark = lookup2trip(test_lookup, max_route_len, len(trip))
    assert abs(get_trip_len(c, trip_benchmark) - fitness) <= 1e-4
    chromosome = decoding(trip_benchmark, n)
    new_label, new_obj = split(n, chromosome, q, d, c, w, max_dist)
    new_split_trip = label2route(n, new_label, chromosome, max_route_len - 1)

    print(f"#. descends: {count}, obj_val: {fitness}")
    if get_trip_len(c, new_split_trip) < fitness:
        global great_count
        great_count += 1
        print(f"great! Improvement: {fitness - get_trip_len(c, new_split_trip)}")
    if get_trip_len(c, new_split_trip) == fitness:
        print("even")
    if get_trip_len(c, new_split_trip) > fitness:
        print("shit")
        raise ValueError(f"{get_trip_len(c, new_split_trip)} is greater than {fitness}")
    global best_obj
    global best_trip
    if new_obj < best_obj:
        best_obj = new_obj
        best_trip = new_split_trip


start = time.perf_counter()
test_case4_inter_route_descend()
end = time.perf_counter()
print(f"compile time: {end - start}")


test_lookup = np.loadtxt("lookup_ori").astype(int)
test_lookup_prev = np.loadtxt("lookup_prev_ori").astype(int)
test_lookup_next = np.loadtxt("lookup_next_ori").astype(int)
trip_benchmark = np.loadtxt("trip_ori").astype(int)
u = int(np.loadtxt("u"))
v = int(np.loadtxt("v"))
r2 = int(np.loadtxt("r2"))
test_lookup_cgd = np.loadtxt("lookup_cgd").astype(int)
test_lookup_prev_cgd = np.loadtxt("lookup_prev_cgd").astype(int)
test_lookup_next_cgd = np.loadtxt("lookup_next_cgd").astype(int)
trip_benchmark_cgd = np.loadtxt("trip_cgd").astype(int)
gain = np.loadtxt("gain")
new_lookup_prev = np.loadtxt("lookup_prev_bench").astype(int)
new_lookup_next = np.loadtxt("lookup_next_bench").astype(int)
new_trip = np.loadtxt("new_trip").astype(int)
