import numpy as np
import vrplib


def read_vrp(root: str, instance: str, fmt: str) -> tuple[dict, dict]:
    """
    read instance and solution from vrplib
    """
    path = f"{root}/{instance}"
    if fmt == "vrp":
        instance_path = f"{path}.vrp"
        solution_path = f"{path}.sol"
        instance = vrplib.read_instance(instance_path)
        solution = vrplib.read_solution(solution_path)
    elif fmt == "solomon":
        instance_path = f"{path}.txt"
        solution_path = f"{path}.sol"
        instance = vrplib.read_instance(instance_path)
        solution = vrplib.read_solution(solution_path)
    else:
        raise IOError("Wrong input file format, possible choices: 'vrp' for vrplib format and '"
                      "solomon' for solomon format")
    return instance, solution


def parse_vrplib(instance: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, int, np.ndarray]:
    """
    parse vrplib instance to x, y coordinates, demand q, capacity w and distance matrix c
    """
    cx = instance["node_coord"][:, 0]
    cy = instance["node_coord"][:, 1]
    q = instance["demand"]
    w = instance["capacity"]
    c = instance["edge_weight"]
    n = len(cx) - 1
    d = np.zeros(n + 1)
    return cx, cy, q, w, c, n, d
