from arffdatasetreader import dataset_reader as dr
from clusteringgenerators import dbscan

num_ds = dr.read_processed_data('numerical', False)
cat_ds = dr.read_processed_data('categorical', False)
mix_ds = dr.read_processed_data('mixed', False)
print(num_ds.head())
print(cat_ds.head())
print(mix_ds.head())

print(dbscan.test_dbscan())
