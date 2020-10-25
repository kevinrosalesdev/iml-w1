from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import numpy as np


def apply_unsupervised_learning(dataset, c, max_iterations=100, m=2):
    np.random.seed(0)
    np_dataset = dataset.to_numpy()
    centroids = [np_dataset[i] for i in np.random.randint(np_dataset.shape[0], size=c)]

    eps = 0.01
    centroid_change = eps + 1
    iteration = 0

    while iteration < max_iterations and centroid_change > eps:

        print("Iteration:", iteration)

        u = [[1 / np.sum([(manhattan_distances([sample], [cluster])[0][0] /
                           0.00001 + manhattan_distances([sample], [other_cluster])[0][0]) ** (2 / m - 0.99999)
                          for other_cluster in np.delete(centroids, index_c, axis=0)])
              for sample in np_dataset]
             for index_c, cluster in enumerate(centroids)]

        last_centroids = np.copy(centroids)

        centroids = [np.sum([(u[cluster][index_s] ** m) * sample for index_s, sample in enumerate(np_dataset)], axis=0)
                     / np.sum([(u[cluster][index_s] ** m) for index_s in range(np_dataset.shape[0])])
                     for cluster in range(c)]

        error = np.sum([np.sum([(u[index_c][index_s] ** m) * euclidean_distances([sample], [cluster])[0][0]
                                for index_c, cluster in enumerate(centroids)])
                        for index_s, sample in enumerate(np_dataset)])

        centroid_change = np.max([manhattan_distances([last_centroids[cluster]], [centroids[cluster]])[0][0]
                                  for cluster in range(c)])

        iteration += 1

    sample_cluster = np.argmax(u, axis=0)
    # print("Universe Matrix", u)
    print("Universe Matrix Shape", len(u), len(u[0]))

    return [sample_cluster, u, error]
