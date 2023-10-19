from numba import njit, int32


@njit(fastmath=True)
def m4_cost_inter(c, u_prev, u, x, v_prev, v, y):
    route1_break = c[u_prev, u] + c[u, x]
    route1_repair = c[u_prev, v] + c[v, x]
    route1_gain = route1_break - route1_repair

    route2_break = c[v_prev, v] + c[v, y]
    route2_repair = c[v_prev, u] + c[u, y]
    route2_gain = route2_break - route2_repair
    gain = route1_gain + route2_gain
    return gain


@njit
def do_m4_inter(r1, r2, pos1, pos2, u_prev, u, x, v_prev, v, y, lookup, lookup_prev, lookup_next, trip_dmd, u_dmd,
                v_dmd):
    # update lookup table
    lookup[u, 0] = r2
    lookup[u, 1] = pos2
    lookup[v, 0] = r1
    lookup[v, 1] = pos1
    # update lookup precedence
    lookup_next[u_prev] = v
    lookup_prev[v] = u_prev
    lookup_next[v] = x
    lookup_prev[x] = v

    lookup_next[v_prev] = u
    lookup_prev[u] = v_prev
    lookup_next[u] = y
    lookup_prev[y] = u
    # update route demand
    trip_dmd[r1] += v_dmd - u_dmd
    trip_dmd[r2] += u_dmd - v_dmd

    # trip_num remains the same


@njit
def do_m4_intra(pos1, pos2, u_prev, u, x, v_prev, v, y, lookup, lookup_prev, lookup_next):
    # update lookup table
    lookup[u, 1] = pos2
    lookup[v, 1] = pos1
    # update lookup precedence
    if pos1 + 1 == pos2:
        lookup_next[u_prev] = v
        lookup_prev[v] = u_prev
        lookup_next[v] = u
        lookup_prev[u] = v
        lookup_next[u] = y
        lookup_prev[y] = u
    elif pos1 + 2 == pos2:
        lookup_next[u_prev] = v
        lookup_prev[v] = u_prev
        lookup_next[v] = x
        lookup_prev[x] = v
        lookup_next[x] = u
        lookup_prev[u] = x
        lookup_next[u] = y
        lookup_prev[y] = u
    elif pos2 + 1 == pos1:
        lookup_next[v_prev] = u
        lookup_prev[u] = v_prev
        lookup_next[u] = v
        lookup_prev[v] = u
        lookup_next[v] = x
        lookup_prev[x] = v

    elif pos2 + 2 == pos1:
        lookup_next[v_prev] = u
        lookup_prev[u] = v_prev
        lookup_next[u] = y
        lookup_prev[y] = u
        lookup_next[y] = v
        lookup_prev[v] = y
        lookup_next[v] = x
        lookup_prev[x] = v
    else:
        lookup_next[u_prev] = v
        lookup_prev[v] = u_prev
        lookup_next[v] = x
        lookup_prev[x] = v
        lookup_next[v_prev] = u
        lookup_prev[u] = v_prev
        lookup_next[u] = y
        lookup_prev[y] = u
