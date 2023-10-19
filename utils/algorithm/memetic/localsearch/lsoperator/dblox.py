from numba import njit


@njit(fastmath=True)
def m6_cost_inter(c, u_prev, u, x, x_post, v_prev, v, y, y_post):
    route1_break = c[u_prev, u] + c[u, x] + c[x, x_post]
    route1_repair = c[u_prev, v] + c[v, y] + c[y, x_post]
    route1_gain = route1_break - route1_repair

    route2_break = c[v_prev, v] + c[v, y] + c[y, y_post]
    route2_repair = c[v_prev, u] + c[u, x] + c[x, y_post]
    route2_gain = route2_break - route2_repair
    gain = route1_gain + route2_gain
    return gain


@njit
def do_m6_inter(r1, r2, pos1, pos2, u_prev, u, x, x_post, v_prev, v, y, y_post, lookup, lookup_prev, lookup_next,
                trip_dmd, u_dmd, v_dmd, x_dmd, y_dmd):
    # update lookup table
    lookup[u, 0] = r2
    lookup[u, 1] = pos2
    lookup[x, 0] = r2
    lookup[x, 1] = pos2 + 1
    lookup[v, 0] = r1
    lookup[v, 1] = pos1
    lookup[y, 0] = r1
    lookup[y, 1] = pos1 + 1

    # update lookup precedence
    lookup_next[u_prev] = v
    lookup_prev[v] = u_prev
    lookup_next[y] = x_post
    lookup_prev[x_post] = y

    lookup_next[v_prev] = u
    lookup_prev[u] = v_prev
    lookup_next[x] = y_post
    lookup_prev[y_post] = x
    # update route demand
    trip_dmd[r1] += v_dmd + y_dmd - u_dmd - x_dmd
    trip_dmd[r2] += u_dmd + x_dmd - v_dmd - y_dmd

