from arffdatasetreader import dataset_reader as dr
from clusteringgenerators import dbscan
from collections import Counter


def get_data():
    num_ds = dr.read_processed_data('numerical', False)
    cat_ds = dr.read_processed_data('categorical', False)
    mix_ds = dr.read_processed_data('mixed', False)
    return num_ds, cat_ds, mix_ds


def test_dbscan(dataset, eps=3, min_samples=4):
    print(dataset.head())
    model = dbscan.ul_dbscan(dataset, eps, min_samples)
    labels = model.labels_
    clusters = Counter(labels)
    # 'the id -1 contains the outliers
    print("Clusters id and the points inside", clusters)
    print('Num of clusters = {}'.format(len(clusters)-1))


if __name__ == '__main__':
    num_ds, cat_ds, mix_ds = get_data()

    # DBScan testing section
    test_dbscan(num_ds)
