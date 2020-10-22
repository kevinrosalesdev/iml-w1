from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def read_processed_data(dataset_type: str, force_creation: bool) -> pd.DataFrame:

    accepted_ds_types = ['numerical', 'categorical', 'mixed']
    if dataset_type not in accepted_ds_types:
        raise ValueError(f"{dataset_type}:: not valid dataset type")

    processed = "_processed"
    csv = ".csv"
    print("Reading dataset: " + dataset_type)
    if dataset_type == 'numerical':
        num_ds_path = "datasets/pen-based"
        if not force_creation:
            try:
                return pd.read_csv(num_ds_path + processed + csv)
            except FileNotFoundError as e:
                print("Processed dataset file not found")
        return process_num_data(num_ds_path)

    if dataset_type == 'categorical':
        cat_ds_path = "datasets/kropt"
        if not force_creation:
            try:
                return pd.read_csv(cat_ds_path + processed + csv)
            except FileNotFoundError as e:
                print("Processed dataset file not found")
        return process_cat_data(cat_ds_path)

    if dataset_type == 'mixed':
        mix_ds_path = "datasets/adult"
        if not force_creation:
            try:
                return pd.read_csv(mix_ds_path + processed + csv)
            except FileNotFoundError as e:
                print("Processed dataset file not found")
        return process_mix_data(mix_ds_path)


def process_num_data(path):
    print("Processing Numerical dataset")

    pen_based_dataset, pen_based_meta = arff.loadarff(path + ".arff")

    numerical_df = pd.DataFrame(pen_based_dataset)
    numerical_df_without_class = numerical_df.drop('a17', axis=1)
    numerical_df_without_class.to_csv(path + '_processed.csv', index=False)
    print("Numerical dataset precessed and created")
    return pd.read_csv(path + '_processed.csv')


def process_cat_data(path):
    print("Processing Categorical dataset")

    kropt_dataset, kropt_meta = arff.loadarff(path + ".arff")

    categ_df = pd.DataFrame(kropt_dataset)
    # Decoding the dataset, these strings are in the form u'string_value'
    for column in categ_df:
        if categ_df[column].dtype == object:
            categ_df[column] = categ_df[column].str.decode('utf8')

    categ_df_without_class = categ_df.drop('game', axis=1)

    # Label Encoding
    for col in categ_df_without_class.columns:
        le = LabelEncoder()
        categ_df_without_class[col] = le.fit_transform(categ_df_without_class[col])

    # Normalizing
    sc = StandardScaler()
    categ_values_normalized = sc.fit_transform(categ_df_without_class)
    categ_df_normalized = pd.DataFrame(categ_values_normalized, columns=categ_df_without_class.columns)
    categ_df_normalized.to_csv(path + '_processed.csv', index=False)
    print("Categorical dataset precessed and created")
    return pd.read_csv(path + '_processed.csv')


def process_mix_data(path):
    print("Processing Mixed dataset")

    adult_dataset, adult_meta = arff.loadarff(path + ".arff")

    mixed_df = pd.DataFrame(adult_dataset)
    for column in mixed_df:
        if mixed_df[column].dtype == object:
            mixed_df[column] = mixed_df[column].str.decode('utf8')

    # Converting Unknown char from "?" to NaN and eliminate the corresponding rows
    mixed_df = mixed_df.replace('?', np.nan)
    mixed_df = mixed_df.dropna()
    mixed_df_without_class = mixed_df.drop('class', axis=1)

    # Label encoding Sex column
    le = LabelEncoder()
    mixed_df_without_class['sex'] = le.fit_transform(mixed_df_without_class['sex'])

    # One hot encoding
    mixed_df_encoded = pd.get_dummies(mixed_df_without_class)

    # Normalizing
    sc = StandardScaler()
    mixed_values_normalized = sc.fit_transform(mixed_df_encoded)
    mixed_df_normalized = pd.DataFrame(mixed_values_normalized, columns=mixed_df_encoded.columns)
    mixed_df_normalized.to_csv(path + '_processed.csv', index=False)
    print("Mixed dataset precessed and created")
    return pd.read_csv(path + '_processed.csv')


def get_datasets(force_creation: bool = False):
    num_ds = read_processed_data('numerical', force_creation)
    cat_ds = read_processed_data('categorical', force_creation)
    mix_ds = read_processed_data('mixed', force_creation)
    return [num_ds, cat_ds, mix_ds]
