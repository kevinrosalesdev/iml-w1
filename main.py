from arffdatasetreader import dataset_reader as dr
from clusteringgenerators import dbscan, kmeans, kmedians, f_cmeans, bisecting_kmeans
from validators import metrics
import time
import math


def test_dbscan(datasets):

    print("Numerical Dataset ('Pen-based') clustering with DBScan with 'auto'")
    min_samples = int(datasets[0].shape[1] + 1 + 0.001 * datasets[0].shape[0])
    dbscan.plot_k_neighbor_distance(datasets[0], k=min_samples)
    dbscan.run_dbscan(datasets[0], eps=0.4425, min_samples=min_samples, algorithm='auto')

    print("Categorical Dataset ('Kropt') clustering with DBScan with 'auto'")
    min_samples = int(datasets[1].shape[1] + 1 + 0.001 * datasets[1].shape[0])
    dbscan.plot_k_neighbor_distance(datasets[1], k=min_samples)
    dbscan.run_dbscan(datasets[1], eps=0.875, min_samples=min_samples, algorithm='auto')

    print("Mixed Dataset ('hypothyroid') clustering with DBScan with 'auto'")
    min_samples = int(datasets[2].shape[1] + 1 + 0.001 * datasets[2].shape[0])
    dbscan.plot_k_neighbor_distance(datasets[2], k=min_samples)
    dbscan.run_dbscan(datasets[2], eps=4.5, min_samples=min_samples, algorithm='auto')
    
    # print("Numerical Dataset ('Pen-based') clustering with DBScan with 'ball_tree'")
    # min_samples = int(datasets[0].shape[1] + 1 + 0.001 * datasets[0].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[0], k=min_samples)
    # dbscan.run_dbscan(datasets[0], eps=0.40, min_samples=min_samples, algorithm='ball_tree')
    #
    # print("Categorical Dataset ('Kropt') clustering with DBScan with 'ball_tree'")
    # min_samples = int(datasets[1].shape[1] + 1 + 0.001 * datasets[1].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[1], k=min_samples)
    # dbscan.run_dbscan(datasets[1], eps=0.85, min_samples=min_samples, algorithm='ball_tree')
    #
    # print("Mixed Dataset ('hypothyroid') clustering with DBScan with 'ball_tree'")
    # min_samples = int(datasets[2].shape[1] + 1 + 0.001 * datasets[2].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[2], k=min_samples)
    # dbscan.run_dbscan(datasets[2], eps=3.30, min_samples=min_samples, algorithm='ball_tree')
    #
    # print("Numerical Dataset ('Pen-based') clustering with DBScan with 'kd_tree'")
    # min_samples = int(datasets[0].shape[1] + 1 + 0.001 * datasets[0].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[0], k=min_samples)
    # dbscan.run_dbscan(datasets[0], eps=0.4, min_samples=min_samples, algorithm='kd_tree')
    #
    # print("Categorical Dataset ('Kropt') clustering with DBScan with 'kd_tree'")
    # min_samples = int(datasets[1].shape[1] + 1 + 0.001 * datasets[1].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[1], k=min_samples)
    # dbscan.run_dbscan(datasets[1], eps=0.85, min_samples=min_samples, algorithm='kd_tree')
    #
    # print("Mixed Dataset ('hypothyroid') clustering with DBScan with 'kd_tree'")
    # min_samples = int(datasets[2].shape[1] + 1 + 0.001 * datasets[2].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[2], k=min_samples)
    # dbscan.run_dbscan(datasets[2], eps=3.3, min_samples=min_samples, algorithm='kd_tree')
    

def test_kmeans(datasets):
    print("Numerical Dataset ('Pen-based') clustering with K-Means")
    kmeans.get_best_k(datasets[0], max_iterations=30)
    kmeans.run_kmeans(datasets[0], k=10, max_iterations=30)

    print("Categorical Dataset ('Kropt') clustering with K-Means")
    kmeans.get_best_k(datasets[1], max_iterations=30)
    kmeans.run_kmeans(datasets[1], k=18, max_iterations=30)

    print("Mixed Dataset ('hypothyroid') clustering with K-Means")
    kmeans.get_best_k(datasets[2], max_iterations=30)
    kmeans.run_kmeans(datasets[2], k=2, max_iterations=30)


def test_kmedians(datasets):
    print("Numerical Dataset ('Pen-based') clustering with K-Medians")
    kmedians.get_best_k(datasets[0], max_iterations=30)
    kmedians.run_kmedians(datasets[0], k=10, max_iterations=30)

    print("Numerical Dataset ('Kropt') clustering with K-Medians")
    kmedians.get_best_k(datasets[1], max_iterations=30)
    kmedians.run_kmedians(datasets[1], k=18, max_iterations=30)

    print("Mixed Dataset ('hypothyroid') clustering with K-Medians")
    kmedians.get_best_k(datasets[2], max_iterations=30)
    kmedians.run_kmedians(datasets[2], k=2, max_iterations=30)


def test_bisecting_kmeans(datasets):
    for i in range(0, len(datasets)):
        tic = time.time()
        bis_kmeans_dim = bisecting_kmeans.BisectingKMeans(10, 1, 'dimension')
        bis_kmeans_dim.apply_unsupervised_learning(datasets[i])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

    for i in range(0, len(datasets)):
        tic = time.time()
        bis_kmeans_std = bisecting_kmeans.BisectingKMeans(10, 1, 'std')
        bis_kmeans_std.apply_unsupervised_learning(datasets[i])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")


def test_f_cmeans(datasets):
    print("Numerical Dataset ('Pen-based') clustering with Fuzzy C-Means")
    f_cmeans.get_best_c(datasets[0], max_iterations=10, max_c=20)
    f_cmeans.run_f_cmeans(datasets[0], c=10, max_iterations=5, m=2)

    print("Numerical Dataset ('Kropt') clustering with Fuzzy C-Means")
    f_cmeans.get_best_c(datasets[0], max_iterations=10, max_c=20)
    f_cmeans.run_f_cmeans(datasets[1], c=18, max_iterations=30)

    print("Mixed Dataset ('hypothyroid') clustering with Fuzzy C-Means")
    f_cmeans.get_best_c(datasets[0], max_iterations=10, max_c=20)
    f_cmeans.run_f_cmeans(datasets[2], c=2, max_iterations=30)


if __name__ == '__main__':
    datasets_preprocessed = dr.get_datasets()
    targets_labels = dr.get_datasets_target()

    # TEST OUR ALGORITHMS WITH THESE FUNCTIONS
    # test_dbscan(datasets_preprocessed)
    # test_kmeans(datasets_preprocessed)
    # test_kmedians(datasets_preprocessed)
    # test_bisecting_kmeans(datasets_preprocessed)
    # test_f_cmeans(datasets_preprocessed)

    # ADD ONE variable of algorithms_params  TO THE FUNCTIONS BELOW TO RUN THE BEST K PLOTS WITH THE CLUSTERING METRICS
    # OR THE CONFUSION MATRIX AND PCA PLOTS
    algorithms_params = ['dbscan', 'kmeans', 'kmedians', 'b-kmeans', 'f-cmeans']

    # metrics.get_metrics(datasets_preprocessed, algorithm='b-kmeans', selector_type='std')
    # metrics.get_cf_and_pca(datasets_preprocessed, targets_labels, algorithm=algorithms_params[1])

    # TO RUN THEM ALL
    # for alg in algorithms_params:
    #   metrics.get_metrics(datasets_preprocessed, algorithm='b-kmeans', selector_type='std')
