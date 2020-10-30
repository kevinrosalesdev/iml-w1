import time, math, main
from utils import plotter
from clusteringgenerators import bisecting_kmeans, kmeans, kmedians


def get_cf_and_pca(dataset, targets, algorithm='b-kmeans'):
    dataset_names = ["Pen-based (num)", "Kropt (cat)", "Hypothyroid (mxd)"]

    best_k_bisecting = [10, 18, 15]
    best_k_kmeans = [7, 23, 6]
    real_k = [10, 18, 4]

    if algorithm == "b-kmeans":
        for index in range(1, len(best_k_bisecting) - 1):
            target_labels = targets[index]

            tic = time.time()
            bis_kmeans = bisecting_kmeans.BisectingKMeans(n_clusters=best_k_bisecting[index], n_iterations=3,
                                                          selector_type='std')
            predicted_labels, k_error = bis_kmeans.apply_unsupervised_learning(dataset[index])
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

            plotter.plot_confusion_matrix(target_labels, predicted_labels,
                                          plot_title=f"{dataset_names[index]} - Bisecting K-Means K={best_k_bisecting[index]}",
                                          is_real_k=False)
            plotter.plot_pca_2D(dataset[index], predicted_labels,
                                plot_title=f"{dataset_names[index]} - Bisecting K-Means K={best_k_bisecting[index]}")

            if real_k[index] != best_k_bisecting[index]:
                tic = time.time()
                bis_kmeans_real = bisecting_kmeans.BisectingKMeans(n_clusters=real_k[index], n_iterations=3,
                                                                   selector_type='std')
                pred_labels, k_error = bis_kmeans_real.apply_unsupervised_learning(dataset[index])
                toc = time.time()
                print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")

                plotter.plot_confusion_matrix(target_labels, pred_labels,
                                              plot_title=f"{dataset_names[index]} - Bisecting K-Means K={real_k[index]}",
                                              is_real_k=True)
                plotter.plot_pca_2D(dataset[index], pred_labels,
                                    plot_title=f"{dataset_names[index]} - Bisecting K-Means K={real_k[index]}")
    elif algorithm == 'kmeans':
        for index in range(0, len(best_k_kmeans)):
            target_labels = targets[index]

            tic = time.time()
            print("K-Means ------", index)
            predicted_labels, iteration_distance, centroids = kmeans.apply_unsupervised_learning(dataset[index],
                                                                                                 best_k_kmeans[index],
                                                                                                 max_iterations=30,
                                                                                                 use_default_seed=True,
                                                                                                 plot_distances=False)
            toc = time.time()
            print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
            plotter.plot_confusion_matrix(target_labels, predicted_labels,
                                          plot_title=f"{dataset_names[index]} - K-Means K={best_k_kmeans[index]}",
                                          is_real_k=False)
            plotter.plot_pca_2D(dataset[index], predicted_labels,
                                plot_title=f"{dataset_names[index]} - K-Means K={best_k_kmeans[index]}")

            if real_k[index] != best_k_kmeans[index]:
                tic = time.time()
                pred_labels, iteration_distance, centroids = kmeans.apply_unsupervised_learning(dataset[index],
                                                                                                real_k[index],
                                                                                                max_iterations=30,
                                                                                                use_default_seed=True,
                                                                                                plot_distances=False)
                toc = time.time()
                print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
                plotter.plot_confusion_matrix(target_labels, pred_labels,
                                              plot_title=f"{dataset_names[index]} - K-Means K={real_k[index]}",
                                              is_real_k=True)
                plotter.plot_pca_2D(dataset[index], pred_labels,
                                    plot_title=f"{dataset_names[index]} - K-Means K={real_k[index]}")

    print("Plotting PCA DATASET WITH REAL LABELS")
    for index in range(0, len(best_k_kmeans)):
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
                main.get_best_k_bisecting_kmeans(dataset=datasets[index], n_iterations=1, selector_type='std',
                                                 max_k=number_k[index], print_k=print_k[index],
                                                 print_silhouette=print_silhouette[index],
                                                 print_calinski_harabasz=print_calinski_harabasz[index],
                                                 print_davies_bouldin=print_davies_bouldin[index])
                toc = time.time()
                print(f"execution time: {math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
        elif selector_type == 'dimension':
            for index in range(0, len(print_k)):
                tic = time.time()
                main.get_best_k_bisecting_kmeans(dataset=datasets[index], n_iterations=1, selector_type='dimension',
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
