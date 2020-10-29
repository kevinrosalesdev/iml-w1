import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.decomposition import PCA
# External Metric
from sklearn.metrics import confusion_matrix


def plot_error(iteration_distances):
    plt.plot(list(range(0, len(iteration_distances))), iteration_distances)
    plt.ylabel('Sum of distances from each sample to its nearest cluster')
    plt.xlabel('Iteration')
    plt.title('Sum of distances per iteration')
    plt.grid()
    plt.show()


def plot_k_error(k_error):
    plt.plot(list(range(2, len(k_error) + 2)), k_error, 'o-')
    plt.ylabel('Sum of distances from each sample to its nearest cluster')
    plt.xlabel('K')
    plt.title('Sum of distances for each \'K\' value')
    plt.xticks(list(range(2, len(k_error) + 2)))
    plt.grid()
    plt.show()


def plot_k_silhouette_score(s_scores):
    plt.plot(list(range(2, len(s_scores) + 2)), s_scores, 'o-')
    plt.ylabel('Silhouette score')
    plt.xlabel('K')
    plt.title('Silhouette score for each \'K\' value')
    plt.xticks(list(range(2, len(s_scores) + 2)))
    plt.grid()
    plt.show()


def plot_k_calinski_harabasz_score(ch_score):
    plt.plot(list(range(2, len(ch_score) + 2)), ch_score, 'o-')
    plt.ylabel('Calinski-Harabasz score')
    plt.xlabel('K')
    plt.title('Calinski-Harabasz score score for each \'K\' value')
    plt.xticks(list(range(2, len(ch_score) + 2)))
    plt.grid()
    plt.show()


def plot_k_davies_bouldin_score(db_score):
    plt.plot(list(range(2, len(db_score) + 2)), db_score, 'o-')
    plt.ylabel('Davies-Bouldin score')
    plt.xlabel('K')
    plt.title('Davies-Bouldin score for each \'K\' value')
    plt.xticks(list(range(2, len(db_score) + 2)))
    plt.grid()
    plt.show()


def plot_confusion_matrix(target, predicted):
    conf_matrix = confusion_matrix(target, predicted)
    unique_target = list(set(target))
    unique_predicted = list(set(predicted))
    if len(unique_predicted) < len(unique_target):
        conf_matrix_df = modify_labels_length_drop_zeros(conf_matrix, unique_predicted, unique_target, True)
    elif len(unique_predicted) > len(unique_target):
        conf_matrix_df = modify_labels_length_drop_zeros(conf_matrix, unique_target, unique_predicted, False)
    else:
        conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_target, columns=unique_predicted)

    sn.heatmap(conf_matrix_df, annot=True, fmt='g', cmap='Blues', linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def modify_labels_length_drop_zeros(conf_matrix, list_to_change, list_target, has_less_columns):
    unique_predicted_bigger = list_target.copy()
    for index in range(0, len(list_to_change)):
        unique_predicted_bigger[index] = list_to_change[index]
    conf_matrix_df = pd.DataFrame(conf_matrix, index=list_target, columns=unique_predicted_bigger)
    if has_less_columns:
        conf_matrix_df = conf_matrix_df.loc[:, (conf_matrix_df != 0).any(axis=0)]
    else:
        conf_matrix_df = conf_matrix_df.loc[(conf_matrix_df != 0).any(axis=1)]
    return conf_matrix_df


def plot_pca_2D(dataset, labels):
    pca = PCA(n_components=2)
    df_2D = pd.DataFrame(pca.fit_transform(dataset), columns=['PCA1', 'PCA2'])
    df_2D['Cluster'] = labels
    sn.lmplot(x="PCA1", y="PCA2", data=df_2D, fit_reg=False, hue='Cluster', legend=True, scatter_kws={"s": 80})
    plt.show()
