import numpy as np
from utils.io.rxml import read_xml
from utils.manipulate import reformat_depot, get_dist_mat
from utils.aorr.tripattr import get_max_route_len, fill_zero
from utils.algorithm.memetic.ga.gaoperator import optimize, get_new_ind

# parameters
pm = 0.1
size = 30
max_dist = 10000
alpha = 9000
beta = 3000
delta = 0.5
rho = 16  # number of restarts
max_agl = 22.5  # angle threshold
filename = "C:/Users/shiyao/Downloads/christofides/CMT01.xml"
# filename = "/mnt/d/ga/ga/data/dvrp/golden/Golden_02.xml"
cx, cy, q, w, depot = read_xml(filename)
cx = reformat_depot(cx)
cy = reformat_depot(cy)
c = np.round(get_dist_mat(cx, cy), 3)
n = len(cx) - 1
d = np.zeros(n)
q = fill_zero(n, q)
d = fill_zero(n, d)
max_route_len = get_max_route_len(q, w)

# main loop
for _ in range(100):
    heuristic_sol = np.empty((3, n + 1), dtype=int)
    for i in range(3):
        ind = get_new_ind(n)
        heuristic_sol[i] = ind
    h_sol = heuristic_sol
    pool, ind_fit = optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl,
                             heuristic_sol)
    print(ind_fit[0])
