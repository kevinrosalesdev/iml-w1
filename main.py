from arffdatasetreader import dataset_reader as dr
from clusteringgenerators import dbscan, kmeans, kmedians, f_cmeans
from clusteringgenerators.bisecting_kmeans import BisectingKMeans
from collections import Counter
import time
import math
import os
from utils import error_plotter
from sklearn.metrics import silhouette_score # (-1:1) higher score relates to a model with better defined clusters
from sklearn.metrics import calinski_harabasz_score # The score is higher when clusters are dense and well separated
from sklearn.metrics import davies_bouldin_score # minor value = 0 the closest is the value, the best is the separation


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


def get_best_k_bisecting_kmeans(dataset, n_iterations=1, selector_type='std', max_k=21, print_k=True,
                                print_silhouette=True):

    print("selector_type =", selector_type)
    k_error = []
    s_scores = []
    for index in range(2, max_k+1):
        print("k =", index)
        bis_kmeans_dim = BisectingKMeans(n_clusters=index, n_iterations=n_iterations, selector_type=selector_type)
        labels, k_error = bis_kmeans_dim.apply_unsupervised_learning(dataset)
        if print_silhouette:
            s_scores.append(silhouette_score(dataset, labels))
    if print_k:
        error_plotter.plot_k_error(k_error)
    if print_silhouette:
        error_plotter.plot_k_silhouette_score(s_scores=s_scores)


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


def best_ks(datasets):
    number_k = [20, 25, 20]
    print_k = [False, True, False]
    print_silhouette = [True, True, True]

    for index in range(0, len(print_k)):
        tic = time.time()
        get_best_k_bisecting_kmeans(dataset=datasets[index], n_iterations=1, selector_type='std',
                                    max_k=number_k[index], print_k=print_k[index],
                                    print_silhouette=print_silhouette[index])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

    for index in range(0, len(print_k)):
        tic = time.time()
        get_best_k_bisecting_kmeans(dataset=datasets[index], n_iterations=1, selector_type='dimension',
                                    max_k=number_k[index], print_k=print_k[index],
                                    print_silhouette=print_silhouette[index])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

    for index in range(0, len(print_k)):
        tic = time.time()
        print("Kmeans ------", index)
        kmeans.get_best_k(datasets[index], max_iterations=30, max_k=number_k[index], print_k=print_k[index],
                          print_silhouette=print_silhouette[index])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")


if __name__ == '__main__':
    datasets = dr.get_datasets()
    #targets = dr.get_datasets_target()
    best_ks(datasets)
    os.system('say "Esecuzione terminata, capra!"')
    # stress_test_bisecting_kmeans(datasets)
    # test_kmedians(datasets)
    #test_f_cmeans(datasets)