import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from tools.plots import plot_dendrogram
from sklearn.cluster import AgglomerativeClustering


def solve(d):
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(d['X1'], d['X2'])
    for index in range(len(d)):
        ax.annotate(str(index), (d["X1"][index], d["X2"][index]))
    plt.show()

    ac_single = AgglomerativeClustering(linkage='single', n_clusters=None, distance_threshold=0)
    ac_single.fit(d)

    ac_complete = AgglomerativeClustering(linkage='complete', n_clusters=None, distance_threshold=0)
    ac_complete.fit(d)

    plot_dendrogram(ac_single, truncate_mode='level', p=4)
    plt.show()
    plot_dendrogram(ac_complete, truncate_mode='level', p=4)
    plt.show()

    for index in range(len(ac_single.children_)):
        print("{}: {}".format(index + 10, ac_single.children_[index]))
    print(ac_single.distances_)
    for index in range(len(ac_complete.children_)):
        print("{}: {}".format(index + 10, ac_complete.children_[index]))
    print(ac_complete.distances_)

    # Most distant 3 merges are (4+5), (12+16), (15+17)
    # (4+5) = (4) + (5)
    # (12+16) = (0, 1, 2, 3) + (4, 5)
    # (15+17) = (6, 7, 8, 9) + (0, 1, 2, 3, 4, 5)
    # Clusters are: [0, 1, 2, 3], [4], [5], [6, 7, 8, 9]
    clusters_single = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    # Most distant 3 merges are (10+11), (14+15), (16+17)
    # (10+11) = (0, 1) + (2, 3)
    # (14+15) = (4, 5) + (6, 7, 8, 9)
    # (16+17) = (0, 1, 2, 3) + (4, 5, 6, 7, 8, 9)
    # Clusters are: [0, 1], [2, 3], [4, 5], [6, 7, 8, 9]
    clusters_complete = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3]

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(d['X1'], d['X2'], c=clusters_single)
    for index in range(len(d)):
        ax.annotate(str(index), (d["X1"][index], d["X2"][index]))
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(d['X1'], d['X2'], c=clusters_complete)
    for index in range(len(d)):
        ax.annotate(str(index), (d["X1"][index], d["X2"][index]))
    plt.show()

    # Single linkage seems to be more direct (rough),
    # splitting anywhere if the distance is big enough

    # Complete linkage seems to be more balanced, sacrificing tighter
    # clusters for inclusion of more distant (or isolated) values
    # in existing clusters


if __name__ == '__main__':
    data = pd.DataFrame({
        "X1": [-4, -3, -2, -1, 1, 1, 2, 3, 3, 4],
        "X2": [-2, -2, -2, -2, -1, 1, 3, 2, 4, 3]
    })
    solve(data)
