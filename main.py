from numba import njit
import numpy as np

arr = np.arange(10)

arr[4:9] = arr[5:10]


@njit
def right_shift(a: np.array, start: int, end: int) -> np.ndarray:
    a[start:end] = a[(start + 1): (end + 1)]
    return a


right_shift(np.arange(10), 4, 9)
