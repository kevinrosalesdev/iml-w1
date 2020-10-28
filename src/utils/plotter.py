from matplotlib import pyplot as plt


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
