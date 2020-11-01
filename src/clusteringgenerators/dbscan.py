from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter


def apply_unsupervised_learning(dataset, eps, min_samples, algorithm):
    if min_samples < dataset.shape[1] + 1 or min_samples < 3:
        print("[WARNING] 'min_samples' parameter should be greater than D(features) + 1 and at least 3")

    print("Parameters (eps=" + str(eps) + ", min_samples=" + str(min_samples) + ")")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm = algorithm)
    model = dbscan.fit(dataset)
    return model


def plot_k_neighbor_distance(dataset, k):
    np_dataset = dataset.to_numpy()
    k_nnd = np.zeros((np_dataset.shape[0]))
    for index, i in enumerate(np_dataset):
        distances = euclidean_distances([i], np.delete(np_dataset, index, axis=0))
        k_nnd[index] = np.sort(distances[0])[k-1]

    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    plt.title("kth Nearest Neighbor Distance (k=" + str(k) + ")")
    plt.xlabel("Points Sorted according to Distance of " + str(k) + "th Nearest Neighbor")
    plt.ylabel(str(k) + "th Nearest Neighbor Distance")
    h_outliers_count = k_nnd[k_nnd > np.percentile(k_nnd, 95)].shape[0]
    plt.plot(list(range(0, dataset.shape[0] - h_outliers_count)),
             np.sort(k_nnd)[:-h_outliers_count], marker=".")
    plt.show()


def run_dbscan(dataset, eps, min_samples, algorithm):
    print(dataset.head())
    model = apply_unsupervised_learning(dataset, eps, min_samples)
    labels = model.labels_
    clusters = Counter(labels)
    # 'the id -1 contains the outliers
    print("Clusters id and the points inside:", clusters)
    print('Num of clusters = {}'.format(len(clusters) - 1))

