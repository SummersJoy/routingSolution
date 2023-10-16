"""
trip representation
"""
import numpy as np
from numba import njit, int32


@njit(cache=True)
def trip_lookup(trip: np.ndarray, n: int) -> np.ndarray:
    """
    Matrix representation of trips
    eg: res[2, 0] = 4, res[2, 1] = 1
    res[:, 0] -> trip_id, res[:, 1] -> position_id
    customer 2 is in trip 4, location 1
    n: number of customers
    """
    res = np.empty((n + 1, 2), dtype=int32)
    res[0:] = -1
    for route_id, route in enumerate(trip):
        for cust_id, cust in enumerate(route):
            if cust == 0:
                break
            res[cust, 0] = route_id
            res[cust, 1] = cust_id
    return res


@njit(fastmath=True, cache=True)
def trip_lookup_precedence(trip: np.ndarray, trip_num: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    compute lookup_prev and lookup_next
    """
    lookup_next = np.zeros((n + 1), dtype=int32)
    lookup_prev = np.zeros((n + 1), dtype=int32)
    for i in range(len(trip)):
        for j in range(trip_num[i] - 1):
            target = trip[i, j]
            t_next = trip[i, j + 1]
            lookup_next[target] = t_next
            lookup_prev[t_next] = target
    return lookup_prev, lookup_next


@njit(fastmath=True, cache=True)
def lookup2trip(lookup: np.ndarray, max_route_len: int, m: int) -> np.ndarray:
    """
    retrieve trip variable from lookup table
    """
    res = np.zeros((m, max_route_len), dtype=int32)
    max_rid = 0
    for i in range(1, len(lookup)):
        rid, pos = lookup[i]
        if rid > max_rid:
            max_rid = rid
        res[rid, pos] = i
    n_row = max(max_rid + 1, m)
    return res[:n_row]


@njit(fastmath=True, cache=True)
def label2route(n: int, p: np.ndarray, s: np.ndarray, max_rl: int) -> np.ndarray:
    """
    convert label to trip
    """
    trip = np.zeros((n, max_rl + 1), dtype=int32)
    t = 0  # -1
    j = n
    while True:
        t += 1
        i = p[j]
        count = 0
        for k in range(i + 1, j + 1):
            trip[t, count] = s[k]
            count += 1
        j = i
        if i == 0:
            break
    return trip[:(t + 1), :]
