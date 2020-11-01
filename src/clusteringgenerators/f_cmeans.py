from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from utils import plotter
import numpy as np


def apply_unsupervised_learning(dataset, c, max_iterations=100, m=2):
    np.random.seed(0)
    np_dataset = dataset.to_numpy()
    centroids = [np_dataset[i]*1.001 for i in np.random.randint(np_dataset.shape[0], size=c)]

    eps = 0.01
    centroid_change = eps + 1
    iteration = 0
    exp = (2 / (m - 0.99999))

    while iteration < max_iterations and centroid_change > eps:
        print("Iteration:", iteration)

        distance_matrix = manhattan_distances(centroids, np_dataset)

        u = [[1 / (np.sum([(distance_matrix[index_c, index_s] / distance_matrix[index_oc, index_s]) ** exp
                           for index_oc in range(len(centroids))]))
              for index_s in range(np_dataset.shape[0])]
             for index_c in range(len(centroids))]

        last_centroids = np.copy(centroids)

        centroids = [np.sum([(u[cluster][index_s] ** m) * sample for index_s, sample in enumerate(np_dataset)], axis=0)
                     / np.sum([(u[cluster][index_s] ** m) for index_s in range(np_dataset.shape[0])])
                     for cluster in range(c)]

        error = np.sum([np.sum([(u[index_c][index_s] ** m) * euclidean_distances([sample], [cluster])[0][0]
                                for index_c, cluster in enumerate(centroids)])
                        for index_s, sample in enumerate(np_dataset)])

        centroid_change = np.sum([manhattan_distances([last_centroids[cluster]], [centroids[cluster]])[0][0]
                                  for cluster in range(c)])

        print("Error:", error, "| Centroid change:", centroid_change)

        iteration += 1

    sample_cluster = np.argmax(u, axis=0)
    average_vector = np.mean(np_dataset, axis=0)
    performance_index = np.sum([np.sum([u[index_c][index_s] * ((euclidean_distances([sample], [cluster])[0][0]) -
                                        (euclidean_distances([cluster], [average_vector])[0][0]))
                                        for index_s, sample in enumerate(np_dataset)])
                                for index_c, cluster in enumerate(centroids)])

    print("Sum of columns", np.sum(u, axis=0))
    # print("Universe Matrix Shape", len(u), len(u[0]), u[0])
    print("Error:", error)
    print("PI:", performance_index)

    return [sample_cluster, error, performance_index]


def get_best_c(dataset, max_iterations=10, max_c=20, print_c=True, print_perf_index=True,
               print_silhouette=True, print_calinski_harabasz=True, print_davies_bouldin=True):
    c_errors = []
    p_indexes = []

    s_scores = []
    ch_score = []
    db_score = []
    for index in range(2, max_c + 1):
        labels, c_error, performance_index = apply_unsupervised_learning(dataset, index, max_iterations, m=2)
        c_errors.append(c_error)
        p_indexes.append(performance_index)

        if print_silhouette:
            s_scores.append(silhouette_score(dataset, labels))
        if print_calinski_harabasz:
            ch_score.append(calinski_harabasz_score(dataset, labels))
        if print_davies_bouldin:
            db_score.append(davies_bouldin_score(dataset, labels))

    if print_c:
        plotter.plot_c_error(c_errors)
    if print_perf_index:
        plotter.plot_performance_index(p_indexes)
    if print_silhouette:
        plotter.plot_k_silhouette_score(s_scores)
    if print_calinski_harabasz:
        plotter.plot_k_calinski_harabasz_score(ch_score)
    if print_davies_bouldin:
        plotter.plot_k_davies_bouldin_score(db_score)
