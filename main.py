from arffdatasetreader import dataset_reader as dr
from clusteringgenerators import dbscan
from collections import Counter


def get_data():
    num_ds = dr.read_processed_data('numerical', False)
    cat_ds = dr.read_processed_data('categorical', False)
    mix_ds = dr.read_processed_data('mixed', False)
    return num_ds, cat_ds, mix_ds


def test_dbscan(dataset, eps, min_samples):
    print(dataset.head())
    model = dbscan.ul_dbscan(dataset, eps, min_samples)
    labels = model.labels_
    clusters = Counter(labels)
    # 'the id -1 contains the outliers
    print("Clusters id and the points inside:", clusters)
    print('Num of clusters = {}'.format(len(clusters)-1))


if __name__ == '__main__':
    num_ds, cat_ds, mix_ds = get_data()

    # DBScan testing section

    print("Numerical Dataset ('Pen-based') clustering with DBScan")
    min_samples = int(num_ds.shape[1] + 1 + 0.001 * num_ds.shape[0])
    # dbscan.plot_k_neighbor_distance(num_ds, k=min_samples)
    test_dbscan(num_ds, eps=14.03, min_samples=min_samples)

    print("Categorical Dataset ('Kropt') clustering with DBScan")
    min_samples = int(cat_ds.shape[1] + 1 + 0.001 * cat_ds.shape[0])
    # dbscan.plot_k_neighbor_distance(cat_ds, k=min_samples)
    test_dbscan(cat_ds, eps=18.1, min_samples=min_samples)

    print("Mixed Dataset ('Adult') clustering with DBScan")
    min_samples = int(mix_ds.shape[1] + 1 + 0.001 * mix_ds.shape[0])
    # dbscan.plot_k_neighbor_distance(mix_ds, k=min_samples)
    test_dbscan(mix_ds, eps=75.49, min_samples=min_samples)
