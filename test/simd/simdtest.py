import numpy as np
from utils.algorithm.memetic.ga.gaoperator import split


def get_test_dist_mat(n: int):
    dist_mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            dist_mat[i, j] = max(100 * i + j, 100 * j + i)
    return dist_mat


def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split('\n'):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print('No instructions found')


n = 50
c = get_test_dist_mat(n)
q = np.random.randint(1, 20, n + 1)
d = np.zeros(n + 1)
max_dist = np.inf
w = 120.
sol = np.random.permutation(50) + 1
sol = np.insert(sol, 0, 0)
label, obj_val = split(n, sol, q, d, c, w, max_dist)

find_instr(split, keyword="subp", sig=0)
find_instr(split, keyword="mulp", sig=0)
