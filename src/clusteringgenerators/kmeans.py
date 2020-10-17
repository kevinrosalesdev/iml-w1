from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
import numpy as np


def ul_kmeans(dataset, k, max_iterations=30, plot_distances=True):
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
        plot_error(iteration_distances)

    return sample_cluster, iteration_distance


def plot_error(iteration_distances):
    plt.plot(list(range(0, len(iteration_distances))), iteration_distances)
    plt.ylabel('Sum of distances from each sample to its nearest cluster')
    plt.xlabel('Iteration')
    plt.title('Sum of distances per iteration')
    plt.grid()
    plt.show()


def plot_k_error(k_error):
    plt.plot(list(range(0, len(k_error))), k_error)
    plt.ylabel('Sum of distances from each sample to its nearest cluster')
    plt.xlabel('K')
    plt.title('Sum of distances for each \'K\' value')
    plt.grid()
    plt.show()


