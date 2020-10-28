from clusteringgenerators import kmeans
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances



class BisectingKMeans:
    def __init__(self, n_clusters: int = 2, n_iterations: int = 1, selector_type: str = 'std'):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.sub_dataset_map = {}
        accepted_selector_type = ['dimension', 'std']
        if selector_type not in accepted_selector_type:
            raise ValueError(f"{selector_type}:: not valid selector type")
        else:
            if selector_type == 'dimension':
                self.get_branch_to_divide = self.get_biggest_cluster
            elif selector_type == 'std':
                self.get_branch_to_divide = self.get_biggest_std

    def apply_unsupervised_learning(self, X):
        if self.n_clusters < 2:
            print(f"Clustering is useless for {self.n_clusters} cluster")
            return
        iteration_distances = []
        branch_to_divide = X.copy()
        self.sub_dataset_map[0] = (branch_to_divide, np.zeros(len(branch_to_divide.columns)))
        i = 0
        print("Applying Bisecting Kmeans clustering...")
        while len(self.sub_dataset_map) < self.n_clusters:
            branch_to_divide = self.get_branch_to_divide()
            best_labels, min_iteration_distance, centroids = self.get_best_bisection(branch_to_divide)
            # dividing the dataset and putting the 2 sub-dataset in our dictionary
            left_branch = branch_to_divide[best_labels == 0]
            right_branch = branch_to_divide[best_labels == 1]
            # print(i in self.sub_dataset_map.keys())
            self.sub_dataset_map[i] = (left_branch, centroids[0])
            # print("left_branch =", len(left_branch.index), "key = ", i)
            i += 1
            # print(i in self.sub_dataset_map.keys())
            self.sub_dataset_map[i] = (right_branch, centroids[1])
            # print("right_branch =", len(right_branch.index), "key = ", i)
            i += 1
            total_sse = 0
            map_keys = list(self.sub_dataset_map.keys())
            for map_key in map_keys:
                subs = self.sub_dataset_map[map_key][0]
                centroid = self.sub_dataset_map[map_key][1]
                total_sse += self.compute_sse(subs, centroid)
            iteration_distances.append(total_sse)
        # print(f"END WHILE --> Number of clusters found = {len(self.sub_dataset_map)}")
        predictions = pd.Series(np.zeros(len(X), dtype=int), index=X.index)
        keys = list(self.sub_dataset_map.keys())
        for i in range(len(self.sub_dataset_map.keys())):
            key = keys[i]
            subset = self.sub_dataset_map[key][0]
            predictions[predictions.index.isin(subset.index)] = i
            # print("cluster", i, "-->", len(subset))
        return predictions.values, iteration_distances

    def get_biggest_cluster(self):
        largest_value = 0
        largest_key = 0
        for key, item in self.sub_dataset_map.items():
            sub_df = item[0]
            num_items = len(sub_df.index)
            if num_items >= largest_value:
                largest_value = num_items
                largest_key = key
        branch_to_divide = self.sub_dataset_map[largest_key]
        self.sub_dataset_map.pop(largest_key)
        # print("Biggest cluster --> key =", largest_key, "n_items =", largest_value)
        return branch_to_divide[0]

    def get_biggest_std(self):
        largest_value = 0
        largest_key = 0
        for key, item in self.sub_dataset_map.items():
            sub_df = item[0]
            std = sub_df.std().mean()
            if std >= largest_value:
                largest_value = std
                largest_key = key
        branch_to_divide, _ = self.sub_dataset_map[largest_key]
        self.sub_dataset_map.pop(largest_key)
        # print("Biggest cluster --> key =", largest_key, "std =", largest_value)
        return branch_to_divide

    def get_best_bisection(self, branch_to_divide):
        iteration = 0
        best_labels = []
        min_iteration_distance = float('inf')
        while iteration < self.n_iterations:
            labels, iteration_distance, centroids = kmeans.apply_unsupervised_learning(branch_to_divide, 2, use_default_seed=True,
                                                                            plot_distances=False)
            # print(f"iteration={iteration}, iteration_distance={iteration_distance}")
            if iteration_distance < min_iteration_distance:
                min_iteration_distance = iteration_distance
                best_labels = labels
            iteration += 1
        return best_labels, min_iteration_distance, centroids

    def compute_sse(self, dataset, centroid):
        np_dataset = dataset.to_numpy()
        centroid_2d = centroid.reshape(1, -1)
        distances_to_clusters = euclidean_distances(np_dataset, centroid_2d)
        return np.sum(distances_to_clusters)
