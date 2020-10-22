from clusteringgenerators import kmeans
import numpy as np
import pandas as pd


class BisectingKMeans:
    def __init__(self, n_clusters: int, selector_type: str):
        self.n_clusters = n_clusters
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
        branch_to_divide = X.copy()
        self.sub_dataset_map[0] = branch_to_divide
        i = 0
        print("Applying Bisecting Kmeans clustering...")
        while len(self.sub_dataset_map) < self.n_clusters:
            branch_to_divide = self.get_branch_to_divide()

            labels = kmeans.ul_kmeans(branch_to_divide, 2, plot_distances=False)[0]
            # dividing the dataset and putting the 2 sub-dataset in our dictionary
            left_branch = branch_to_divide[labels == 0]
            right_branch = branch_to_divide[labels == 1]
            # print(i in self.sub_dataset_map.keys())
            self.sub_dataset_map[i] = left_branch
            # print("left_branch =", len(left_branch.index), "key = ", i)
            i += 1
            # print(i in self.sub_dataset_map.keys())
            self.sub_dataset_map[i] = right_branch
            # print("right_branch =", len(right_branch.index), "key = ", i)
            i += 1

        # print(f"END WHILE --> Number of clusters found = {len(self.sub_dataset_map)}")
        predictions = pd.Series(np.zeros(len(X), dtype=int), index=X.index)
        keys = list(self.sub_dataset_map.keys())
        for i in range(len(self.sub_dataset_map.keys())):
            key = keys[i]
            subset = self.sub_dataset_map[key]
            predictions[predictions.index.isin(subset.index)] = i
            # print("cluster", i, "-->", len(subset))
        return predictions.values

    def get_biggest_cluster(self):
        largest_value = 0
        largest_key = 0
        for key, sub_df in self.sub_dataset_map.items():
            num_items = len(sub_df.index)
            if num_items >= largest_value:
                largest_value = num_items
                largest_key = key
        branch_to_divide = self.sub_dataset_map[largest_key]
        self.sub_dataset_map.pop(largest_key)
        # print("Biggest cluster --> key =", largest_key, "n_items =", largest_value)
        return branch_to_divide

    def get_biggest_std(self):
        largest_value = 0
        largest_key = 0
        for key, sub_df in self.sub_dataset_map.items():
            std = sub_df.std().mean()
            if std >= largest_value:
                largest_value = std
                largest_key = key
        branch_to_divide = self.sub_dataset_map[largest_key]
        self.sub_dataset_map.pop(largest_key)
        # print("Biggest cluster --> key =", largest_key, "std =", largest_value)
        return branch_to_divide
