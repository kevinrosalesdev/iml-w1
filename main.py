from arffdatasetreader import dataset_reader as dr
from clusteringgenerators import dbscan, kmeans, kmedians, f_cmeans
from clusteringgenerators.bisecting_kmeans import BisectingKMeans
from collections import Counter
import time
import math
import os
from utils import plotter
# Internal Metrics
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
    kmeans.get_best_k(datasets[1], max_iterations=10)
    run_kmeans(datasets[1], k=18, max_iterations=30)

    print("Mixed Dataset ('hypothyroid') clustering with K-Means")
    kmeans.get_best_k(datasets[2], max_iterations=10)
    run_kmeans(datasets[2], k=2, max_iterations=30)


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


def get_best_k_bisecting_kmeans(dataset, n_iterations=1, selector_type='std', max_k=20, print_k=True,
                                print_silhouette=True,
                                print_calinski_harabasz=True,
                                print_davies_bouldin=True):

    print("selector_type =", selector_type)
    k_error = []
    s_scores = []
    ch_score = []
    db_score = []
    for index in range(2, max_k+1):
        print("k =", index)
        bis_kmeans_dim = BisectingKMeans(n_clusters=index, n_iterations=n_iterations, selector_type=selector_type)
        labels, k_error = bis_kmeans_dim.apply_unsupervised_learning(dataset)
        if print_silhouette:
            s_scores.append(silhouette_score(dataset, labels))
        if print_calinski_harabasz:
            ch_score.append(calinski_harabasz_score(dataset, labels))
        if print_davies_bouldin:
            db_score.append(davies_bouldin_score(dataset, labels))
    if print_k:
        plotter.plot_k_error(k_error)
    if print_silhouette:
        plotter.plot_k_silhouette_score(s_scores=s_scores)
    if print_calinski_harabasz:
        plotter.plot_k_calinski_harabasz_score(ch_score=ch_score)
    if print_davies_bouldin:
        plotter.plot_k_davies_bouldin_score(db_score=db_score)


def test_kmedians(datasets):
    print("Numerical Dataset ('Pen-based') clustering with K-Medians")
    kmedians.get_best_k(datasets[0], max_iterations=10)
    run_kmedians(datasets[0], k=10, max_iterations=30)
    
    print("Numerical Dataset ('Kropt') clustering with K-Medians")
    kmedians.get_best_k(datasets[1], max_iterations=10)
    run_kmedians(datasets[1], k=18, max_iterations=30)
    
    print("Mixed Dataset ('hypothyroid') clustering with K-Medians")
    kmedians.get_best_k(datasets[2], max_iterations=10)
    run_kmedians(datasets[2], k=2, max_iterations=30)


def test_f_cmeans(datasets):
    print("Numerical Dataset ('Pen-based') clustering with K-Medians")
    run_f_cmeans(datasets[0], c=10, max_iterations=5, m=2)

    print("Numerical Dataset ('Kropt') clustering with K-Medians")
    # run_f_cmeans(datasets[1], c=18, max_iterations=30)

    print("Mixed Dataset ('hypothyroid') clustering with K-Medians")
    # run_f_cmeans(datasets[2], c=2, max_iterations=30)


def best_k_bis_kmeans_plots(datasets):
    number_k = [20, 25, 20]
    print_k = [True, True, True]
    print_silhouette = [True, True, True]
    print_calinski_harabasz = [True, True, True]
    print_davies_bouldin = [True, True, True]

    for index in range(0, len(print_k)):
        tic = time.time()
        get_best_k_bisecting_kmeans(dataset=datasets[index], n_iterations=1, selector_type='std',
                                    max_k=number_k[index], print_k=print_k[index],
                                    print_silhouette=print_silhouette[index],
                                    print_calinski_harabasz=print_calinski_harabasz[index],
                                    print_davies_bouldin=print_davies_bouldin[index])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

        """ 
    for index in range(0, len(print_k)):
        tic = time.time()
        get_best_k_bisecting_kmeans(dataset=datasets[index], n_iterations=1, selector_type='dimension',
                                    max_k=number_k[index], print_k=print_k[index],
                                    print_silhouette=print_silhouette[index],
                                    print_calinski_harabasz=print_calinski_harabasz[index],
                                    print_davies_bouldin=print_davies_bouldin[index])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
        """

    for index in range(0, len(print_k)):
        tic = time.time()
        print("Kmeans ------", index)
        kmeans.get_best_k(datasets[index], max_iterations=30, max_k=number_k[index], print_k=print_k[index],
                          print_silhouette=print_silhouette[index],
                          print_calinski_harabasz=print_calinski_harabasz[index],
                          print_davies_bouldin=print_davies_bouldin[index])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

    for index in range(0, len(print_k)):
        tic = time.time()
        print("Kmedians ------", index)
        kmedians.get_best_k(datasets[index], max_iterations=30, max_k=number_k[index], print_k=print_k[index],
                          print_silhouette=print_silhouette[index],
                          print_calinski_harabasz=print_calinski_harabasz[index],
                          print_davies_bouldin=print_davies_bouldin[index])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")



def validate_best_k_bis_kmeans(dataset, targets):
    #test lines
    #best_k_bisecting = [3]
    #best_k_kmeans = [3]
    #real_k = [2]
    dataset_names = ["Pen-based (num)", "Kropt (cat)", "Hypothyroid (mxd)"]

    best_k_bisecting = [10, 18, 15]
    best_k_kmeans = [7, 23, 6]
    real_k = [10, 18, 4]

    for index in range(0, len(best_k_bisecting)):
        target_labels = targets[index]

        tic = time.time()
        bis_kmeans = BisectingKMeans(n_clusters=best_k_bisecting[index], n_iterations=3, selector_type='std')
        predicted_labels, k_error = bis_kmeans.apply_unsupervised_learning(dataset[index])
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

        plotter.plot_confusion_matrix(target_labels, predicted_labels,
                                      plot_title=f"{dataset_names[index]} - Bisecting K-Means K={best_k_bisecting[index]}", is_real_k=False)
        plotter.plot_pca_2D(dataset[index], predicted_labels,
                            plot_title=f"{dataset_names[index]} - Bisecting K-Means K={best_k_bisecting[index]}")

        if real_k[index] != best_k_bisecting[index]:
            tic = time.time()
            bis_kmeans_real = BisectingKMeans(n_clusters=real_k[index], n_iterations=3, selector_type='std')
            pred_labels, k_error = bis_kmeans_real.apply_unsupervised_learning(dataset[index])
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

            plotter.plot_confusion_matrix(target_labels, pred_labels,
                                          plot_title=f"{dataset_names[index]} - Bisecting K-Means K={real_k[index]}", is_real_k=True)
            plotter.plot_pca_2D(dataset[index], pred_labels,
                                plot_title=f"{dataset_names[index]} - Bisecting K-Means K={real_k[index]}")


    for index in range(0, len(best_k_kmeans)):
        target_labels = targets[index]

        tic = time.time()
        print("K-Means ------", index)
        predicted_labels, iteration_distance, centroids = kmeans.apply_unsupervised_learning(dataset[index],
                                                        best_k_kmeans[index], max_iterations=30,
                                                        use_default_seed=True, plot_distances=False)
        toc = time.time()
        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
        plotter.plot_confusion_matrix(target_labels, predicted_labels,
                                      plot_title=f"{dataset_names[index]} - K-Means K={best_k_kmeans[index]}", is_real_k=False)
        plotter.plot_pca_2D(dataset[index], predicted_labels,
                            plot_title=f"{dataset_names[index]} - K-Means K={best_k_kmeans[index]}")

        if real_k[index] != best_k_kmeans[index]:
            tic = time.time()
            pred_labels, iteration_distance, centroids = kmeans.apply_unsupervised_learning(dataset[index],
                                                        real_k[index], max_iterations=30,
                                                        use_default_seed=True, plot_distances=False)
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
            plotter.plot_confusion_matrix(target_labels, pred_labels,
                                          plot_title=f"{dataset_names[index]} - K-Means K={real_k[index]}", is_real_k=True)
            plotter.plot_pca_2D(dataset[index], pred_labels,
                                plot_title=f"{dataset_names[index]} - K-Means K={real_k[index]}")

    print("Plotting PCA DATASET WITH REAL LABELS")
    for index in range(0, len(best_k_kmeans)):
        target_labels = targets[index]
        plotter.plot_pca_2D(dataset[index], target_labels, plot_title=f"{dataset_names[index]} Real classes K={real_k[index]}")


if __name__ == '__main__':
    datasets_preprocessed = dr.get_datasets()
    targets_labels = dr.get_datasets_target()
    # best_k_bis_kmeans_plots(datasets_preprocessed)
    validate_best_k_bis_kmeans(datasets_preprocessed, targets_labels)
    # stress_test_bisecting_kmeans(datasets_preprocessed)
    # test_kmedians(datasets_preprocessed)
    # test_f_cmeans(datasets_preprocessed)
    # os.system('say "Esecuzione terminata, capra!"')

