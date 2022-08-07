# Input - data for diffusion, central points, covarince matrix for each cluster, Number of clusters, Number of samples, number of DM to keep
# Output - Diffusion vectors

import datetime
import numpy as np
from numpy import linalg as LA
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def scalable_adm(data, central_points, clusters_cov_inverse, M, m, eps_factor, embeddim=3):
    logging.debug('Start SCL_ADM')

    ct = datetime.datetime.now()

    # The distance between each sample and the center of each one of the clouds/clusters
    distance_matrix = np.zeros((M, m))

    for i in range(M):  # for each song
        for j in range(m):  # for each cluster - calculate the distance between song i and the center of cluster j
            difference = np.subtract(data.iloc[i, :], central_points[j])
            distance_matrix[i, j] = (difference.dot(clusters_cov_inverse[j])).dot(np.transpose(difference))

    # Heuristic for epsilon
    row_min = distance_matrix.min(axis=1)  # finding min of each row
    min_eps = row_min.max()
    print("epsilon used in ADM: " + str(eps_factor * min_eps))
    # Convert the distance matrix to exp values and normalize by row sum(markovic)
    A = np.exp(-distance_matrix / (eps_factor * min_eps))

    ####### Scaling part######

    W_ref = np.transpose(A).dot(A)  # 15*15

    d1 = np.sum(W_ref, axis=0).reshape(-1, 1)

    A_div_d1 = np.divide(A, (d1 * A.shape[0]).T)

    d = np.sum(A_div_d1, axis=1).reshape(-1, 1)

    A1 = np.divide(A_div_d1, d * A.shape[1])

    W1 = np.matmul(A1.T, A1)

    eigval, eigvec = LA.eigh(W1)  # Return the eigenvalues and eigenvectors of a real symmetric matrix.
    eigval_sorted, eigvec_sorted = np.flip(eigval), np.flip(eigvec, axis=1)

    eigvec_mini_cld = eigvec_sorted[:, 1:]  # get rid of the first eigen vector    #D*phi_j_sorted[:, 1:]
    eigval_mini_cld = eigval_sorted[1:]  # get rid of the first eigen value

    psi_mat = []
    for i in range(eigvec_mini_cld.shape[1]):
        psi = np.divide(np.matmul(A1, eigvec_mini_cld[:, i]),
                        np.sqrt((eigval_mini_cld[i])))      # Wref and Wext eigvec conversion
        psi_mat.append(psi)
    psi_mat = np.asarray(psi_mat).T
    ct2 = datetime.datetime.now()
    print(ct2)
    logging.debug('Finish SCL_ADM')
    return (psi_mat[:, :3])