from sklearn.cluster import DBSCAN
from collections import Counter


def test_dbscan(dataset):
    dbscan = DBSCAN(eps=3, min_samples=4, metric='euclidean')

    model = dbscan.fit(dataset)

    labels = model.labels_
    clusters = Counter(labels)
    # 'the id -1 contains the outliers
    print("Clusters id and the points inside", clusters)
    print('Num of clusters = {}'.format(len(clusters)-1))
    return model
