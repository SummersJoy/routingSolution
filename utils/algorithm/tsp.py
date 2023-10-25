import os
import scipy.spatial.distance as sd
import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt
from mip import Model, xsum, CBC, GRB, BINARY, INTEGER, minimize


# helper functions
def dir_manipulation(data_type, n_city, n_instance):
    data_dir = "../data"
    file_name = f"{n_city}cities_instance{n_instance}.csv"
    return os.path.join(data_dir, data_type, file_name)


def data_generator(n_city, n_instance):
    coordinates = np.random.rand(2, n_city)
    dist_mat = sd.cdist(coordinates.transpose(), coordinates.transpose())
    np.fill_diagonal(dist_mat, np.inf)
    data_file = dir_manipulation("coordinates", n_city, n_instance)
    dist_file = dir_manipulation("dist_mat", n_city, n_instance)
    np.savetxt(data_file, coordinates)
    np.savetxt(dist_file, dist_mat)


def data_reader(n_city, n_instance):
    data_file = dir_manipulation("coordinates", n_city, n_instance)
    dist_file = dir_manipulation("dist_mat", n_city, n_instance)
    coordinates = np.loadtxt(data_file)
    dist_mat = np.loadtxt(dist_file)
    return coordinates, dist_mat


def generate_base_plot(points, n, n_instance):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.scatter(points[0], points[1])
    ax.title.set_text(f"TSP with {n} cities instance {n_instance}")
    for i, txt in enumerate(range(n)):
        ax.annotate(txt, (points[0][i] + 0.01, points[1][i] + 0.01))
    plt.show()
    return fig, ax


def tsp_lazy(dist_mat):
    tour_animation = []
    # initialize a model calling solver CBC (default is gurobi)
    model = Model("TSP")

    n = len(dist_mat)

    # binary variables indicating if arc (i,j) is used on the route or not
    x = [[model.add_var(var_type=BINARY) for _ in range(n)] for _ in range(n)]

    # objective function: minimize tour length
    model.objective = minimize(xsum(dist_mat[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j))

    # constraint type 1: from each city, only 1 of remaining cities can be visited
    for j in range(n):
        model.add_constr(xsum(x[i][j] for i in range(n) if i != j) == 1)

    # constraint type 2: for each city, only 1 inbound arc from other cities is allowed
    for i in range(n):
        model.add_constr(xsum(x[i][j] for j in range(n) if j != i) == 1)

    model.optimize()

    arcs = dict()
    for i in range(n):
        for j in range(n):
            if x[i][j].x >= 0.9:
                arcs[i] = j

    all_tour = subtour_detection(arcs, n)
    tour_animation.append(all_tour)

    while len(all_tour) != 1:
        for tour in all_tour:
            tour_length = len(tour) - 1
            model.add_constr(xsum(x[tour[i]][tour[i + 1]] for i in range(tour_length)) <= tour_length - 1)
            if len(tour) > 3:
                model.add_constr(xsum(x[tour[i + 1]][tour[i]] for i in range(tour_length)) <= tour_length - 1)
        model.verbose = 1
        model.write("tsp.lp")
        print("iteration")
        model.optimize()
        arcs = dict()
        for i in range(n):
            for j in range(n):
                if x[i][j].x >= 0.9:
                    arcs[i] = j

        all_tour = subtour_detection(arcs, n)
        tour_animation.append(all_tour)
    print(model.objective_value)
    return all_tour[0], tour_animation, model.objective_value


def subtour_detection(arcs, n_cities):
    unvisited = list(range(n_cities))
    all_tour = []
    while unvisited:
        current = unvisited[0]
        tour = []
        for i in range(len(arcs)):
            tour.append(current)
            unvisited.remove(current)
            current = arcs[current]
            if current in tour:
                tour.append(current)
                break
        all_tour.append(tour)
        print(tour)
    return all_tour


def solve_instance(n_city, instances, optimal_sol):
    for n_instance in instances:
        # data_generator(n_city, n_instance)
        coordinates, dist_mat = data_reader(n_city, n_instance)
        fig, ax = generate_base_plot(coordinates, n_city, n_instance)
        fig.show()
        optimal_tour, animation, optimal_len = tsp_lazy(dist_mat)  # solve instance
        optimal_sol[(n_city, n_instance)] = optimal_len
        optimal_len = np.round(optimal_len, 4)
        ax.plot(coordinates[0][optimal_tour], coordinates[1][optimal_tour], 'r-')
        ax.title.set_text(f"Optimal tour of TSP with {n_city} cities instance {n_instance} with length: {optimal_len}")
        fig.show()


def route_length(n_city, n_instance, route):
    coordinates, dist_mat = data_reader(n_city, n_instance)
    total = 0
    for i in range(n_city - 1):
        total += dist_mat[route[i], route[i + 1]]
    total += dist_mat[route[-1], route[0]]
    return total, coordinates, dist_mat


def evaluate_user_solution(n_city, n_instance, route, optimal_sol):
    user_route_length, coordinates, dist_mat = route_length(n_city, n_instance, route)
    optimal_len = optimal_sol[(n_city, n_instance)]
    user_route_gap = np.round(100 * (user_route_length - optimal_len) / optimal_len, 4)
    fig, ax = generate_base_plot(coordinates, n_city, n_instance)
    route.append(route[0])
    ax.plot(coordinates[0][route], coordinates[1][route], 'r-')
    ax.title.set_text(f"User tour of TSP with {n_city} cities instance {n_instance}, gap: {user_route_gap}%")
    fig.show()


class Tsp:
    def __init__(self, n_city, n_instance):
        self.coordinates, self.dist_mat = data_reader(n_city, n_instance)
        self.n_cities = self.dist_mat.shape[0]

    def exhaustive_search(self):
        best_route = None
        shortest_len = np.inf
        # get all permutations
        feasible_routes = permutations(list(range(self.n_cities)))
        for route in feasible_routes:
            tour_len = self.route_length(route)
            if tour_len < shortest_len:
                best_route = route
                shortest_len = tour_len
        return best_route, shortest_len

    def exact_solve(self):
        pass

    def greedy(self):
        pass

    def route_length(self, route):
        total = 0
        for i in range(self.n_cities - 1):
            total += self.dist_mat[route[i], route[i + 1]]
        total += self.dist_mat[route[-1], route[0]]
        return total

# n_city = 5
# n_instance = 1
# data_generator(n_city, n_instance)
# tsp = Tsp(n_city=5, n_instance=1)
# tsp.exhaustive_search()