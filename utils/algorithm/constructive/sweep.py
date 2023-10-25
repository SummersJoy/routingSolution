import numpy as np
from numba import njit, int32
from utils.algorithm.memetic.localsearch.neighbor import get_angle


@njit
def sweep2lookup(seq, n, w, q):
    # seq is angle sorted customer
    lookup = np.empty((n + 1, 2), dtype=int32)
    route_idx = 0
    pos = 0
    cap = 0
    for i in range(1, n + 1):
        cap += q[seq[i]]
        if cap <= w:
            lookup[seq[i], 0] = route_idx
            lookup[seq[i], 1] = pos
            pos += 1
        else:
            cap = 0
            route_idx += 1
            pos = 0
            lookup[seq[i], 0] = route_idx
            lookup[seq[i], 1] = pos
            pos += 1
    return lookup


@njit
def sweep_heuristics(cx, cy, n, idx, w, q):
    angle = get_angle(cx, cy)
    angle_sorted = np.argsort(angle)[1:]
    seq = np.concatenate((np.zeros(1, dtype=int32), angle_sorted[idx:], angle_sorted[:idx][::-1]))
    lookup = sweep2lookup(seq, n, w, q)
    return lookup


# lookup = sweep_heuristics(cx, cy, n, idx, w, q)
# trip = lookup2trip(lookup, max_route_len, 11)
# val = get_trip_len(c, trip)
