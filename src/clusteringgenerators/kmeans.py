from sklearn.metrics.pairwise import euclidean_distances
from utils import plotter
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from collections import Counter


def apply_unsupervised_learning(dataset, k, max_iterations=30, use_default_seed=True, plot_distances=False):

    if use_default_seed:
        np.random.seed(0)
    np_dataset = dataset.to_numpy()
    centroids = [np_dataset[i] for i in np.random.randint(np_dataset.shape[0], size=k)]
    sample_cluster = -1 * np.ones(np_dataset.shape[0], dtype=int)
    no_change = False
    iteration = 0
    iteration_distances = []
    print("Clustering...")
    while no_change is False and iteration < max_iterations:
        # print("Iteration: " + str(iteration))
        no_change = True
        iteration_distance = 0.
        for index, sample in enumerate(np_dataset):
            distances_to_clusters = euclidean_distances([sample], centroids)[0]
            iteration_distance += np.min(distances_to_clusters)
            closest_cluster = np.argmin(distances_to_clusters)
            if closest_cluster != sample_cluster[index]:
                sample_cluster[index] = closest_cluster
                no_change = False

        iteration_distances.append(iteration_distance)
        centroids = [np.mean(np_dataset[np.where(sample_cluster == centroid_index)[0]], axis=0)
                     for centroid_index in range(0, k)]
        iteration += 1

    if plot_distances:
        plotter.plot_error(iteration_distances)

    return sample_cluster, iteration_distance, centroids


def get_best_k(dataset, max_iterations=30, max_k=20, print_k=True, print_silhouette=True,
               print_calinski_harabasz=True, print_davies_bouldin=True):
    k_errors = []
    s_scores = []
    ch_score = []
    db_score = []
    for index in range(2, max_k+1):
        labels, k_error, _ = apply_unsupervised_learning(dataset, index, max_iterations, plot_distances=False)
        k_errors.append(k_error)
        if print_silhouette:
            s_scores.append(silhouette_score(dataset, labels))
        if print_calinski_harabasz:
            ch_score.append(calinski_harabasz_score(dataset, labels))
        if print_davies_bouldin:
            db_score.append(davies_bouldin_score(dataset, labels))

    if print_k:
        plotter.plot_k_error(k_errors)
    if print_silhouette:
        plotter.plot_k_silhouette_score(s_scores)
    if print_calinski_harabasz:
        plotter.plot_k_calinski_harabasz_score(ch_score)
    if print_davies_bouldin:
        plotter.plot_k_davies_bouldin_score(db_score)


def run_kmeans(dataset, k, max_iterations=30):
    print(dataset.head())
    labels = apply_unsupervised_learning(dataset, k, max_iterations)[0]
    print(labels)
    clusters = Counter(labels)
    print("Clusters id and the points inside:", clusters)
    print('Num of clusters = {}'.format(len(clusters)))
