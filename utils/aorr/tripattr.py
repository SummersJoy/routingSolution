"""
trip attributes
"""
import numpy as np
from numba import njit, int32


@njit(fastmath=True)
def get_route_len(c: np.ndarray, route: np.ndarray) -> float:
    res = 0.
    res += c[0, route[0]]
    i = 0
    for i in range(len(route) - 1):
        if route[i] == 0:
            break
        else:
            res += c[route[i], route[i + 1]]
    res += c[route[i + 1], 0]
    return res


@njit(fastmath=True)
def get_trip_len(c: np.ndarray, trip: np.ndarray) -> float:
    """
    compute the total travel distance of a given trip
    """
    res = 0.
    for route in trip:
        res += get_route_len(c, route)
    return res


@njit(fastmath=True)
def get_trip_dmd(trip: np.ndarray, q: np.ndarray, trip_num: np.ndarray) -> np.ndarray:
    """
    compute the total demand on each trip
    """
    n = len(trip)
    res = np.empty(n)
    for i in range(n):
        demand = 0.
        for j in range(trip_num[i]):
            demand += q[trip[i, j]]
        res[i] = demand
    return res


@njit(fastmath=True)
def get_trip_num(trip: np.ndarray) -> np.ndarray:
    """
    compute the number of customers on each trip
    """
    n = len(trip)
    res = np.empty(n, dtype=int32)
    for i in range(n):
        count = 0
        for j in trip[i]:
            if j == 0:
                break
            else:
                count += 1
        res[i] = count
    return res


@njit(fastmath=True)
def get_max_route_len(q: np.ndarray, w: float) -> int:
    """
    Compute the max number of possible customer in each trip
    """
    n = len(q) - 1
    demand = sorted(q[1:])
    current = 0
    for i in range(n):
        current += demand[i]
        if current > w:
            return i + 1
    return n


@njit()
def get_neighbors(lookup_prev, lookup_next, u, v):
    u_prev = lookup_prev[u]
    x = lookup_next[u]
    x_post = lookup_next[x] if x else 0

    v_prev = lookup_prev[v] if v else 0
    y = lookup_next[v] if v else 0
    y_post = lookup_next[y] if y else 0
    return u_prev, x, x_post, v_prev, y, y_post


@njit()
def get_demand(q, u, x, v, y):
    u_dmd = q[u]
    x_dmd = q[x]
    v_dmd = q[v]
    y_dmd = q[y]
    return u_dmd, x_dmd, v_dmd, y_dmd


@njit()
def get_route_pos(lookup, i, j):
    r1 = lookup[i, 0]
    pos1 = lookup[i, 1]
    r2 = lookup[j, 0]
    pos2 = lookup[j, 1]
    return r1, r2, pos1, pos2


@njit
def fill_zero(n: int, arr: np.array) -> np.array:
    tmp = np.zeros(n + 1, dtype=int32)
    tmp[1:] = arr
    return tmp
