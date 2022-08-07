import numpy as np
from scipy.sparse.linalg import svds, eigs
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Input - data for diffusion, central points, covarince matrix for each cluster, Number of clusters, Number of samples, number of DM to keep
# Output - Diffusion vectors
def adm(data, central_points, clusters_cov_inverse, M, m, epsilon=3, embeddim=3):
    logging.debug('Start ADM')
    # The distance between each sample and the center of each one of the clouds/clusters
    distance_matrix = np.zeros((M, m))

    # dif1 = np.subtract(data,central_points[1])
    # a = (dif1.dot(clusters_cov_inverse[1]))
    # print(a.shape)
    #
    # for j in range(M):
    #     print(dif1.iloc[j, :])
    #     print(dif1.iloc[j, :].T)
    #     distance_matrix[j, 1] = (a.dot(dif1.iloc[j, :].T))
    # print(distance_matrix.iloc[:,1])

    for i in range(M):  # for each song
        for j in range(m):  # for each cluster - calculate the distance between song i and the center of cluster j
            difference = np.subtract(data.iloc[i, :], central_points[j])
            print('difference shape: ' + str(difference.shape))
            print('clusters_cov_inverse[j] shape: ' + str(clusters_cov_inverse[j].shape))
            print('difference.T shape: ' + str(np.transpose(difference).shape))
            distance_matrix[i, j] = (difference.dot(clusters_cov_inverse[j])).dot(np.transpose(difference))

    # Heuristic for epsilon
    row_min = distance_matrix.min(axis=1)  # finding min of each row
    min_eps = row_min.max()
    print("epsilon used in ADM: " + str(0.05 * min_eps))
    # Convert the distance matrix to exp values and normalize by row sum(markovic)
    A = np.exp(-distance_matrix / (0.05 * min_eps))

    # Create the kernel - based on the central points and the cov matrix
    W_big = A.dot(np.transpose(A))
    W_big = np.divide(W_big, np.sum(W_big, axis=1))

    # Computing SVD decomposition of P
    eig_vec, eig_val, vt = svds(W_big, k=embeddim, which='LM')

    # Create mask for sorting
    sort_mask = np.flip(np.argsort(eig_val))

    # Sorting:
    eig_val_sorted = eig_val[sort_mask]
    eig_vec_sorted = eig_vec[:, sort_mask]

    # Return first few dimensions, as specified in "embeddim" var
    logging.debug('Finish ADM')
    return (eig_vec_sorted[:, 1:] * eig_val_sorted[1:])

