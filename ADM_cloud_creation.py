import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from kmeans_lst import Kmeans_lst
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

############ Create the clouds for ADM - based on one artist data only ###############
def create_clouds(tonal_data, cloud_mfcc, number_of_tonal_clusters):
    logging.debug('Start ADM_cloud_creation')
    #run kmeans on mfcc - one artist data
    Kmeans_mfcc = Kmeans_lst(cloud_mfcc,number_of_tonal_clusters)


    #Add kmeans result to tonal_data and cloud_mfcc
    tonal_data['kmeans_result'] = number_of_tonal_clusters + 1
    tonal_data.loc[cloud_mfcc.index, 'kmeans_result'] = Kmeans_mfcc

    cloud_mfcc['kmeans_result'] = Kmeans_mfcc
    #Create the column to store the info on the balanced clusters
    tonal_data['balanced_clusters'] = number_of_tonal_clusters + 1

    #### Balance the groups ####

    #group by original means to create centers
    central_means = cloud_mfcc.groupby('kmeans_result').mean()
    cloud_mfcc = cloud_mfcc.drop('kmeans_result',axis = 1)

    #Get 20 Nearest neighbors of each center
    cloud_nn = NearestNeighbors(n_neighbors=min(20,cloud_mfcc.shape[0])).fit(cloud_mfcc)
    distances, indices = cloud_nn.kneighbors(central_means)
    Central_points_knn = indices[(-1*number_of_tonal_clusters):,:]

    #Initiate the central points and cov matrixes lists for adm
    central_points = []
    clusters_cov_inverse = []

    # Calculate the mean and cov matrix for each tonal cluster -  BALANCED groups
    cluster_num = 0
    for j in Central_points_knn:
        current_tonal_cluster = pd.DataFrame(cloud_mfcc).iloc[j,:]

        #Update the column stores the info on the balanced clusters
        tonal_data['balanced_clusters'].loc[current_tonal_cluster.index] = cluster_num
        cluster_num+=1

        #Update the lists of mean and cov for the current cluster
        central_points.append(current_tonal_cluster.mean())
        clusters_cov_inverse.append(np.linalg.pinv(np.cov(np.transpose(current_tonal_cluster.values))))

    #Finish log
    logging.debug('Finish ADM_cloud_creation')
    return(tonal_data, cloud_mfcc, central_points, clusters_cov_inverse)

