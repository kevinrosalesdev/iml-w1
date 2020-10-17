from arffdatasetreader import dataset_reader as dr
from clusteringgenerators import dbscan, kmeans, b_kmeans, k_x
from collections import Counter


def get_data():
    num_ds = dr.read_processed_data('numerical', False)
    cat_ds = dr.read_processed_data('categorical', False)
    mix_ds = dr.read_processed_data('mixed', False)
    return [num_ds, cat_ds, mix_ds]


def run_dbscan(dataset, eps, min_samples):
    print(dataset.head())
    model = dbscan.ul_dbscan(dataset, eps, min_samples)
    labels = model.labels_
    clusters = Counter(labels)
    # 'the id -1 contains the outliers
    print("Clusters id and the points inside:", clusters)
    print('Num of clusters = {}'.format(len(clusters)-1))


def test_dbscan(datasets):
    print("Numerical Dataset ('Pen-based') clustering with DBScan")
    min_samples = int(datasets[0].shape[1] + 1 + 0.001 * datasets[0].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[0], k=min_samples)
    run_dbscan(datasets[0], eps=14.03, min_samples=min_samples)

    print("Categorical Dataset ('Kropt') clustering with DBScan")
    min_samples = int(datasets[1].shape[1] + 1 + 0.001 * datasets[1].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[1], k=min_samples)
    run_dbscan(datasets[1], eps=18.1, min_samples=min_samples)

    print("Mixed Dataset ('Adult') clustering with DBScan")
    min_samples = int(datasets[2].shape[1] + 1 + 0.001 * datasets[2].shape[0])
    # dbscan.plot_k_neighbor_distance(datasets[2], k=min_samples)
    run_dbscan(datasets[2], eps=75.49, min_samples=min_samples)


def test_kmeans(datasets):
    kmeans.ul_kmeans(datasets[0])
    kmeans.ul_kmeans(datasets[1])
    kmeans.ul_kmeans(datasets[2])


def test_b_kmeans(datasets):
    b_kmeans.ul_b_kmeans(datasets[0])
    b_kmeans.ul_b_kmeans(datasets[1])
    b_kmeans.ul_b_kmeans(datasets[2])


def test_k_x(datasets):
    k_x.ul_k_x(datasets[0])
    k_x.ul_k_x(datasets[1])
    k_x.ul_k_x(datasets[2])


if __name__ == '__main__':
    datasets = get_data()

    test_dbscan(datasets)

    test_kmeans(datasets)
    test_b_kmeans(datasets)
    test_k_x(datasets)
