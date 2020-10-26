from arffdatasetreader import dataset_reader as dr
from clusteringgenerators import dbscan, kmeans, kmedians, f_cmeans
from clusteringgenerators.bisecting_kmeans import BisectingKMeans
from collections import Counter
import time
import math
from utils import error_plotter



def run_dbscan(dataset, eps, min_samples):
    print(dataset.head())
    model = dbscan.apply_unsupervised_learning(dataset, eps, min_samples)
    labels = model.labels_
    clusters = Counter(labels)
    # 'the id -1 contains the outliers
    print("Clusters id and the points inside:", clusters)
    print('Num of clusters = {}'.format(len(clusters)-1))


def run_kmeans(dataset, k, max_iterations=30):
    print(dataset.head())
    labels = kmeans.apply_unsupervised_learning(dataset, k, max_iterations)[0]
    print(labels)
    clusters = Counter(labels)
    print("Clusters id and the points inside:", clusters)
    print('Num of clusters = {}'.format(len(clusters)))


def run_kmedians(dataset, k, max_iterations=30):
    print(dataset.head())
    labels = kmedians.apply_unsupervised_learning(dataset, k, max_iterations)[0]
    print(labels)
    clusters = Counter(labels)
    print("Clusters id and the points inside in K-medians:", clusters)
    print('Num of clusters in K-medians = {}'.format(len(clusters)))


def run_f_cmeans(dataset, c, max_iterations=100, m=2):
    print(dataset.head())
    labels = f_cmeans.apply_unsupervised_learning(dataset, c, max_iterations, m)[0]
    print(labels)
    clusters = Counter(labels)
    print("Clusters id and the points inside in K-medians:", clusters)
    print('Num of clusters in K-medians = {}'.format(len(clusters)))
    

def test_dbscan(datasets):
    print("Numerical Dataset ('Pen-based') clustering with DBScan")
    min_samples = int(datasets[0].shape[1] + 1 + 0.001 * datasets[0].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[0], k=min_samples)
    run_dbscan(datasets[0], eps=14.03, min_samples=min_samples)

    print("Categorical Dataset ('Kropt') clustering with DBScan")
    min_samples = int(datasets[1].shape[1] + 1 + 0.001 * datasets[1].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[1], k=min_samples)
    run_dbscan(datasets[1], eps=18.1, min_samples=min_samples)

    print("Mixed Dataset ('hypothyroid') clustering with DBScan")
    min_samples = int(datasets[2].shape[1] + 1 + 0.001 * datasets[2].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[2], k=min_samples)
    run_dbscan(datasets[2], eps=75.49, min_samples=min_samples)


def test_kmeans(datasets):
    print("Numerical Dataset ('Pen-based') clustering with K-Means")
    kmeans.get_best_k(datasets[0], max_iterations=10)
    run_kmeans(datasets[0], k=10, max_iterations=30)

    print("Categorical Dataset ('Kropt') clustering with K-Means")
    # kmeans.get_best_k(datasets[1], max_iterations=10)
    # run_kmeans(datasets[1], k=18, max_iterations=30)

    print("Mixed Dataset ('hypothyroid') clustering with K-Means")
    # kmeans.get_best_k(datasets[2], max_iterations=10)
    # run_kmeans(datasets[2], k=2, max_iterations=30)


def test_bisecting_kmeans(datasets):

    for i in range(0, len(datasets)):
        tic = time.time()
        bis_kmeans_dim = BisectingKMeans(10, 1, 'dimension')
        bis_kmeans_dim.apply_unsupervised_learning(datasets[i])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

    for i in range(0, len(datasets)):
        tic = time.time()
        bis_kmeans_std = BisectingKMeans(10, 1, 'std')
        bis_kmeans_std.apply_unsupervised_learning(datasets[i])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")


def get_best_k_bisecting_kmeans(dataset, n_iterations):

    k_error = []
    for index in range(1, 21):
        print("k =", index)
        bis_kmeans_dim = BisectingKMeans(n_clusters=index, n_iterations=n_iterations, selector_type='dimension')
        labels, k_error = bis_kmeans_dim.apply_unsupervised_learning(dataset)
    error_plotter.plot_k_error(k_error)


def test_kmedians(datasets):
    print("Numerical Dataset ('Pen-based') clustering with K-Medians")
    run_kmedians(datasets[0], k=10, max_iterations=30)
    
    print("Numerical Dataset ('Kropt') clustering with K-Medians")
    run_kmedians(datasets[1], k=18, max_iterations=30)
    
    print("Mixed Dataset ('hypothyroid') clustering with K-Medians")
    run_kmedians(datasets[2], k=2, max_iterations=30)


def test_f_cmeans(datasets):
    print("Numerical Dataset ('Pen-based') clustering with K-Medians")
    run_f_cmeans(datasets[0], c=10, max_iterations=5, m=2)

    print("Numerical Dataset ('Kropt') clustering with K-Medians")
    # run_f_cmeans(datasets[1], c=18, max_iterations=30)

    print("Mixed Dataset ('hypothyroid') clustering with K-Medians")
    # run_f_cmeans(datasets[2], c=2, max_iterations=30)


if __name__ == '__main__':
    datasets = dr.get_datasets()
    targets = dr.get_datasets_target()

    # test_dbscan(datasets)

    # test_kmeans(datasets)
    #test_bisecting_kmeans(datasets)
    get_best_k_bisecting_kmeans(dataset=datasets[2], n_iterations=1)
    # stress_test_bisecting_kmeans(datasets)
    # test_kmedians(datasets)

    #test_f_cmeans(datasets)