import matplotlib.pyplot as plt
import numpy as np


def base_plot(cx, cy, fitness):
    n = len(cx)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.scatter(cx, cy)
    ax.title.set_text(f"CVRP with {n - 1} customers and distance {np.round(fitness, 3)}")
    for i, txt in enumerate(range(n)):
        ax.annotate(txt, (cx[i], cy[i] + 0.01))
    return fig, ax


def plot_trip(cx, cy, t, ax):
    lst = []
    for i in t:
        if i:
            lst.append(i)
        else:
            break
    ax.plot(cx[lst], cy[lst])


def plot_sol(cx: np.ndarray, cy: np.ndarray, trip: np.ndarray, fitness: float, instance_id: str) -> None:
    fig, ax = base_plot(cx, cy, fitness)
    for t in trip:
        plot_trip(cx, cy, t, ax)
    plt.savefig(f"./data/fig/{instance_id}_{str(fitness).replace('.','_')}.png")
    plt.show()