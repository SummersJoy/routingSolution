import json


def config_feature(feature):
    feature_dict = json.load(open(feature))
    annealing = feature_dict["annealing"]
    do_sa = annealing["do_sa"]
    temp = annealing["temp"]
    factor = annealing["factor"]

    lift = feature_dict["lift"]
    perturbation_size = lift["perturbation_size"]
    div_lim = lift["lift_iter"]
    return do_sa, temp, factor, perturbation_size, div_lim


def config_ga(args):
    p = json.load(open(args))
    pm = p["pm"]
    size = p["size"]
    max_dist = p["max_dist"]
    alpha = p["alpha"]
    beta = p["beta"]
    delta = p["delta"]
    rho = p["rho"]
    max_agl = p["max_agl"]
    return pm, size, max_dist, alpha, beta, delta, rho, max_agl
