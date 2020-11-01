import time, math
from utils import plotter
from clusteringgenerators import dbscan, bisecting_kmeans, kmeans, kmedians, f_cmeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


def get_cf_and_pca(dataset, targets, algorithm='b-kmeans', plot_pca_real_labels=False):
    dataset_names = ["Pen-based (num)", "Kropt (cat)", "Hypothyroid (mxd)"]
    best_k = []
    if algorithm == 'b-kmeans':
        best_k.extend([10, 18, 15])
    elif algorithm == 'kmeans':
        best_k.extend([9, 19, 6])
    elif algorithm == 'kmedians':
        best_k.extend([10, 18, 8])
    elif algorithm == 'f-cmeans':
        best_k.extend([10, 18, 10])

    real_k = [10, 18, 4]
    best_eps = [0.4425, 0.875, 4.5]
    best_min_samples = [27, 35, 35]

    for index in range(0, len(real_k)):
        target_labels = targets[index]

        tic = time.time()
        if algorithm == 'b-kmeans':
            bis_kmeans = bisecting_kmeans.BisectingKMeans(n_clusters=best_k[index], n_iterations=3,
                                                          selector_type='std')
            pred_labels, k_error = bis_kmeans.apply_unsupervised_learning(dataset[index])
        elif algorithm == 'kmeans':
            pred_labels, iteration_distance, _ = kmeans.apply_unsupervised_learning(dataset[index],
                                                                                    best_k[index],
                                                                                    max_iterations=30,
                                                                                    use_default_seed=True,
                                                                                    plot_distances=False)
        elif algorithm == 'kmedians':
            pred_labels, iteration_distance = kmedians.apply_unsupervised_learning(dataset[index],
                                                                                   best_k[index],
                                                                                   max_iterations=30,
                                                                                   plot_distances=False)
        elif algorithm == 'f-cmeans':
            pred_labels, error, _ = f_cmeans.apply_unsupervised_learning(dataset[index], c=best_k[index],
                                                                         max_iterations=3)

        elif algorithm == 'dbscan':
            model = dbscan.apply_unsupervised_learning(dataset[index], eps=best_eps[index],
                                                       min_samples=best_min_samples[index])
            pred_labels = model.labels_

        toc = time.time()

        print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
        print("Silhouette score", silhouette_score(dataset[index], pred_labels))
        print("Calinski-Harabasz score", calinski_harabasz_score(dataset[index], pred_labels))
        print("Davies Bouldin score", davies_bouldin_score(dataset[index], pred_labels))

        if algorithm != 'dbscan':
            plotter.plot_confusion_matrix(target_labels, pred_labels,
                                          plot_title=f"{dataset_names[index]} - {algorithm} K={best_k[index]}",
                                          is_real_k=False)
            plotter.plot_pca_2D(dataset[index], pred_labels,
                                plot_title=f"{dataset_names[index]} - {algorithm} K={best_k[index]}")
        elif algorithm == 'dbscan':
            plotter.plot_pca_2D(dataset[index], pred_labels,
                                plot_title=f"{dataset_names[index]} - {algorithm} eps={best_eps[index]} "
                                           f"min_s={best_min_samples[index]}")

        if algorithm != 'dbscan' and real_k[index] != best_k[index]:
            tic = time.time()
            if algorithm == 'b-kmeans':
                bis_kmeans_real = bisecting_kmeans.BisectingKMeans(n_clusters=real_k[index], n_iterations=3,
                                                                   selector_type='std')
                pred_labels, k_error = bis_kmeans_real.apply_unsupervised_learning(dataset[index])
            elif algorithm == 'kmeans':
                pred_labels, iteration_distance, _ = kmeans.apply_unsupervised_learning(dataset[index],
                                                                                        real_k[index],
                                                                                        max_iterations=30,
                                                                                        use_default_seed=True,
                                                                                        plot_distances=False)
            elif algorithm == 'kmedians':
                pred_labels, iteration_distance = kmedians.apply_unsupervised_learning(dataset[index],
                                                                                       real_k[index],
                                                                                       max_iterations=30,
                                                                                       plot_distances=False)
            elif algorithm == 'f-cmeans':
                pred_labels, error, _ = f_cmeans.apply_unsupervised_learning(dataset[index], c=real_k[index],
                                                                             max_iterations=3)

            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

            plotter.plot_confusion_matrix(target_labels, pred_labels,
                                          plot_title=f"{dataset_names[index]} - {algorithm} K={real_k[index]}",
                                          is_real_k=True)
            plotter.plot_pca_2D(dataset[index], pred_labels,
                                plot_title=f"{dataset_names[index]} - {algorithm} K={real_k[index]}")

    if plot_pca_real_labels:
        print("Plotting PCA DATASET WITH REAL LABELS")
        for index in range(0, len(real_k)):
            target_labels = targets[index]
            plotter.plot_pca_2D(dataset[index], target_labels,
                                plot_title=f"{dataset_names[index]} Real classes K={real_k[index]}")


def get_metrics(datasets, algorithm='b-kmeans', selector_type='std'):
    number_k = [20, 25, 20]
    print_k = [True, True, True]
    print_silhouette = [True, True, True]
    print_calinski_harabasz = [True, True, True]
    print_davies_bouldin = [True, True, True]

    if algorithm == 'b-kmeans':
        if selector_type == 'std':
            for index in range(0, len(print_k)):
                tic = time.time()
                bisecting_kmeans.get_best_k(dataset=datasets[index], n_iterations=1, selector_type='std',
                                            max_k=number_k[index], print_k=print_k[index],
                                            print_silhouette=print_silhouette[index],
                                            print_calinski_harabasz=print_calinski_harabasz[index],
                                            print_davies_bouldin=print_davies_bouldin[index])
                toc = time.time()
                print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
        elif selector_type == 'dimension':
            for index in range(0, len(print_k)):
                tic = time.time()
                bisecting_kmeans.get_best_k(dataset=datasets[index], n_iterations=1, selector_type='dimension',
                                            max_k=number_k[index], print_k=print_k[index],
                                            print_silhouette=print_silhouette[index],
                                            print_calinski_harabasz=print_calinski_harabasz[index],
                                            print_davies_bouldin=print_davies_bouldin[index])
                toc = time.time()
                print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

    elif algorithm == 'kmeans':
        for index in range(0, len(print_k)):
            tic = time.time()
            kmeans.get_best_k(datasets[index], max_iterations=30, max_k=number_k[index], print_k=print_k[index],
                              print_silhouette=print_silhouette[index],
                              print_calinski_harabasz=print_calinski_harabasz[index],
                              print_davies_bouldin=print_davies_bouldin[index])
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

    elif algorithm == 'kmedians':
        for index in range(0, len(print_k)):
            tic = time.time()
            kmedians.get_best_k(datasets[index], max_iterations=30, max_k=number_k[index], print_k=print_k[index],
                                print_silhouette=print_silhouette[index],
                                print_calinski_harabasz=print_calinski_harabasz[index],
                                print_davies_bouldin=print_davies_bouldin[index])
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

    elif algorithm == 'f-cmeans':
        for index in range(0, len(print_k)):
            tic = time.time()
            f_cmeans.get_best_c(datasets[index], max_iterations=10, max_c=number_k[index], print_c=print_k[index],
                                print_silhouette=print_silhouette[index],
                                print_calinski_harabasz=print_calinski_harabasz[index],
                                print_davies_bouldin=print_davies_bouldin[index])
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
