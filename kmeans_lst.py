from sklearn.cluster import KMeans
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Input - data , number of clusters.
# Output - list of indexes - for each sample, which cluster it belongs.
def Kmeans_lst(data,number_of_tonal_clusters):
    logging.debug('Start kmeans')
    KMEANS=KMeans(n_clusters=number_of_tonal_clusters, random_state=0)
    run_kmeans = KMEANS.fit(data)
    clusters_index = list(run_kmeans.labels_)
    logging.debug('Finish kmeans')
    return clusters_index


