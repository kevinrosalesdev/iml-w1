from sklearn.cluster import DBSCAN


def ul_dbscan(dataset, eps, min_samples):
    dbscan = DBSCAN(eps=3, min_samples=4, metric='euclidean')
    model = dbscan.fit(dataset)
    return model
