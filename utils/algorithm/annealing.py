import numpy as np
from numba import njit, int32
from utils.aorr.tripattr import get_neighbors, get_demand, get_route_pos
from utils.aorr.triprepr import lookup2trip
from utils.algorithm.memetic.localsearch.lsoperator.relocate import m1_cost_inter, do_m1_inter, do_m1_intra
from utils.algorithm.memetic.localsearch.lsoperator.dblrelocate import m2_cost_inter, do_m2_inter, do_m2_intra
from utils.algorithm.memetic.localsearch.lsoperator.oropt import m3_cost_inter, do_m3_inter, do_m3_intra
from utils.algorithm.memetic.localsearch.lsoperator.cox import m4_cost_inter, do_m4_inter, do_m4_intra
from utils.algorithm.memetic.localsearch.lsoperator.asmox import m5_cost_inter, do_m5_inter, do_m5_intra
from utils.algorithm.memetic.localsearch.lsoperator.dblox import m6_cost_inter, do_m6_inter


@njit
def mutation_annealing(n, c, fitness, trip_dmd, q, w, lookup, neighbor, trip_num, lookup_prev, lookup_next, idx,
                       neighbor_size, temp, factor):
    stall = 0
    for i in range(1000):
        gain, label = descend_annealing(n, lookup, lookup_prev, lookup_next, q, trip_dmd, trip_num, c, w, neighbor, idx,
                                        neighbor_size, temp)
        fitness -= gain
        temp *= factor
        if abs(gain) <= 1e-4:
            stall += 1
        if stall == 20:
            break
    return fitness


@njit(fastmath=True)
def descend_annealing(n, lookup, lookup_prev, lookup_next, q, trip_dmd, trip_num, c, w, neighbor, idx, neighbor_size,
                      temp, tol=1e-4):
    for i in range(idx, neighbor_size):
        gain, label = do_descend_annealing(lookup, lookup_prev, lookup_next, q, trip_dmd, trip_num, c, w, neighbor, i,
                                           temp, tol)
        if label:
            return gain, label
    for i in range(idx):
        gain, label = do_descend_annealing(lookup, lookup_prev, lookup_next, q, trip_dmd, trip_num, c, w, neighbor, i,
                                           temp, tol)
        if label:
            return gain, label
    gain, label = empty_route_descend_annealing(n, q, trip_num, trip_dmd, lookup, lookup_prev, lookup_next, c, temp,
                                                tol)
    if label:
        return gain, label
    return 0., False


@njit(fastmath=True)
def do_descend_annealing(lookup, lookup_prev, lookup_next, q, trip_dmd, trip_num, c, w, neighbor, i, temp, tol):
    u = neighbor[i, 0]
    v = neighbor[i, 1]
    r1, r2, pos1, pos2 = get_route_pos(lookup, u, v)
    u_prev, x, x_post, v_prev, y, y_post = get_neighbors(lookup_prev, lookup_next, u, v)
    u_dmd, x_dmd, v_dmd, y_dmd = get_demand(q, u, x, v, y)
    if r1 != r2:
        gain, label = inter_route_descend_annealing(u_prev, u, x, x_post, v_prev, v, y, y_post, u_dmd, x_dmd, v_dmd,
                                                    y_dmd, w, r1, r2, pos1, pos2, trip_num, trip_dmd, lookup,
                                                    lookup_prev, lookup_next, c, temp, tol)
        if label:
            return gain, True
    else:
        gain, label = intra_route_descend_annealing(u_prev, u, x, x_post, v_prev, v, y, pos1, pos2, lookup, lookup_prev,
                                                    lookup_next, c, temp, tol)
        if label:
            return gain, True
    return 0., False


@njit(fastmath=True)
def inter_route_descend_annealing(u_prev, u, x, x_post, v_prev, v, y, y_post, u_dmd, x_dmd, v_dmd, y_dmd, w, r1, r2,
                                  pos1, pos2, trip_num, trip_dmd, lookup, lookup_prev, lookup_next, c, temp, tol):
    if trip_dmd[r2] + u_dmd <= w:
        gain = m1_cost_inter(c, u_prev, u, x, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            do_m1_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, trip_num, lookup_prev, lookup_next, u_prev, u, x,
                        v, y)
            return gain, True
    if x and trip_dmd[r2] + u_dmd + x_dmd <= w:
        gain = m2_cost_inter(c, u_prev, u, x, x_post, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            # trip = lookup2trip(lookup, 50, 20)
            # np.savetxt("lookup_ori", lookup)
            # np.savetxt("lookup_prev_ori", lookup_prev)
            # np.savetxt("lookup_next_ori", lookup_next)
            # np.savetxt("u", np.array([u]))
            # np.savetxt("v", np.array([v]))
            # np.savetxt("r2", np.array([r2]))
            # np.savetxt("gain", np.array([gain]))
            # np.savetxt("trip", trip)
            # assert np.sum(trip) == 1275
            do_m2_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                        u_prev, u, x, x_post, v, y)
            # trip = lookup2trip(lookup, 50, 20)
            # np.savetxt("lookup_cgd", lookup)
            # np.savetxt("lookup_prev_cgd", lookup_prev)
            # np.savetxt("lookup_next_cgd", lookup_next)
            # np.savetxt("trip_cgd", trip)
            # assert np.sum(trip) == 1275
            return gain, True
        gain = m3_cost_inter(c, u_prev, u, x, x_post, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            do_m3_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                        u_prev, u, x, x_post, v, y)
            return gain, True
    if trip_dmd[r1] - u_dmd + v_dmd <= w and trip_dmd[r2] - v_dmd + u_dmd <= w:
        gain = m4_cost_inter(c, u_prev, u, x, v_prev, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            do_m4_inter(r1, r2, pos1, pos2, u_prev, u, x, v_prev, v, y, lookup, lookup_prev, lookup_next,
                        trip_dmd, u_dmd, v_dmd)
            return gain, True
    if x and trip_dmd[r1] - u_dmd - x_dmd + v_dmd <= w and trip_dmd[r2] - v_dmd + u_dmd + x_dmd <= w:
        gain = m5_cost_inter(c, u_prev, u, x, x_post, v_prev, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            do_m5_inter(r1, r2, pos1, pos2, lookup, trip_dmd, u_dmd, x_dmd, v_dmd, trip_num, lookup_prev,
                        lookup_next, u_prev, u, x, x_post, v_prev, v, y)
            return gain, True
    if x and y and trip_dmd[r1] - u_dmd - x_dmd + v_dmd + y_dmd <= w and trip_dmd[r2] + u_dmd + x_dmd - v_dmd - y_dmd <= w:
        gain = m6_cost_inter(c, u_prev, u, x, x_post, v_prev, v, y, y_post)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            do_m6_inter(r1, r2, pos1, pos2, u_prev, u, x, x_post, v_prev, v, y, y_post, lookup, lookup_prev,
                        lookup_next, trip_dmd, u_dmd, v_dmd, x_dmd, y_dmd)
            return gain, True
    return 0., False


@njit(fastmath=True)
def intra_route_descend_annealing(u_prev, u, x, x_post, v_prev, v, y, pos1, pos2, lookup, lookup_prev, lookup_next, c,
                                  temp, tol):
    if u != y and v != x:
        gain = m1_cost_inter(c, u_prev, u, x, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            do_m1_intra(pos1, pos2, u_prev, u, x, v, y, lookup, lookup_next, lookup_prev)
            return gain, True
    if x and x != v and u != y:  # abs(pos1 - pos2) > 1:
        gain = m2_cost_inter(c, u_prev, u, x, x_post, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            # trip = lookup2trip(lookup, 50, 20)
            # np.savetxt("lookup_ori", lookup)
            # np.savetxt("lookup_prev_ori", lookup_prev)
            # np.savetxt("lookup_next_ori", lookup_next)
            # np.savetxt("u", np.array([u]))
            # np.savetxt("v", np.array([v]))
            # np.savetxt("gain", np.array([gain]))
            # np.savetxt("trip", trip)
            # assert np.sum(trip) == 1275
            do_m2_intra(pos1, pos2, u_prev, u, x, x_post, v, y, lookup, lookup_next, lookup_prev)
            # trip = lookup2trip(lookup, 50, 20)
            # np.savetxt("lookup_cgd", lookup)
            # np.savetxt("lookup_prev_cgd", lookup_prev)
            # np.savetxt("lookup_next_cgd", lookup_next)
            # np.savetxt("trip_cgd", trip)
            # assert np.sum(trip) == 1275
            return gain, True
        gain = m3_cost_inter(c, u_prev, u, x, x_post, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            do_m3_intra(pos1, pos2, u_prev, u, x, x_post, v, y, lookup, lookup_next, lookup_prev)
            return gain, True
    if abs(pos1 - pos2) > 2:  # todo m4 intra when |pos1 - pos2| = 1
        gain = m4_cost_inter(c, u_prev, u, x, v_prev, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            do_m4_intra(pos1, pos2, u_prev, u, x, v_prev, v, y, lookup, lookup_prev, lookup_next)
            return gain, True
    if x and abs(pos1 - pos2) > 2:
        gain = m5_cost_inter(c, u_prev, u, x, x_post, v_prev, v, y)
        if gain > tol or np.random.random() < np.exp(gain / temp):
            do_m5_intra(pos1, pos2, u_prev, u, x, x_post, v_prev, v, y, lookup, lookup_next, lookup_prev)
            return gain, True
    return 0., False


@njit(fastmath=True)
def empty_route_descend_annealing(n, q, trip_num, trip_dmd, lookup, lookup_prev, lookup_next, c, temp, tol):
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
                if gain > tol or np.random.random() < np.exp(gain / temp):
                    do_m1_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, trip_num, lookup_prev, lookup_next, u_prev, u, x,
                                v, y)
                    return gain, True
                if x:
                    gain = m2_cost_inter(c, u_prev, u, x, x_post, v, y)
                    if gain > tol or np.random.random() < np.exp(gain / temp):
                        # trip = lookup2trip(lookup, 50, 20)
                        # np.savetxt("lookup_ori", lookup)
                        # np.savetxt("lookup_prev_ori", lookup_prev)
                        # np.savetxt("lookup_next_ori", lookup_next)
                        # np.savetxt("u", np.array([u]))
                        # np.savetxt("v", np.array([v]))
                        # np.savetxt("r2", np.array([r2]))
                        # np.savetxt("gain", np.array([gain]))
                        # np.savetxt("trip", trip)
                        # assert np.sum(trip) == 1275
                        do_m2_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                    u_prev, u, x, x_post, v, y)
                        # trip = lookup2trip(lookup, 50, 20)
                        # np.savetxt("lookup_cgd", lookup)
                        # np.savetxt("lookup_prev_cgd", lookup_prev)
                        # np.savetxt("lookup_next_cgd", lookup_next)
                        # np.savetxt("trip_cgd", trip)
                        # assert np.sum(trip) == 1275
                        return gain, True
                    gain = m3_cost_inter(c, u_prev, u, x, x_post, v, y)
                    if gain > tol or np.random.random() < np.exp(gain / temp):
                        do_m3_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                    u_prev, u, x, x_post, v, y)
                        return gain, True
            break
    return 0., False


# #  Duplicated i, j: 20, 36
# test_lookup = np.loadtxt("lookup_ori").astype(int)
# test_lookup_prev = np.loadtxt("lookup_prev_ori").astype(int)
# test_lookup_next = np.loadtxt("lookup_next_ori").astype(int)
# u = int(np.loadtxt("u"))
# v = int(np.loadtxt("v"))
# r2 = int(np.loadtxt("r2"))
# test_lookup_cgd = np.loadtxt("lookup_cgd").astype(int)
# test_lookup_prev_cgd = np.loadtxt("lookup_prev_cgd").astype(int)
# test_lookup_next_cgd = np.loadtxt("lookup_next_cgd").astype(int)
# gain = np.loadtxt("gain")
# trip = np.loadtxt("trip").astype(int)
# trip_cgd = np.loadtxt("trip_cgd").astype(int)


