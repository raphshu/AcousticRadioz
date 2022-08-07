import numpy as np
from scipy.sparse.linalg import svds, eigs
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# data = Kernel , dimension = Number of DM we want to keep
def diffusionMapping(data, t, **kwargs):
    logging.debug('Start DM')
    try:
        kwargs['dim'] or kwargs['delta']
    except KeyError:
        raise KeyError('specify either dim or delta as keyword argument!')

    # Computing a square matrix of Euclidean dist.
    eucl_mtx = squareform(pdist(data, metric='euclidean'))

    # Computing epsilon using max-min heuristic:
    np.fill_diagonal(eucl_mtx, np.max(eucl_mtx))
    row_min = eucl_mtx.min(axis=1)  # finding min. of each row
    eps = row_min.max()
    #It was 10 * eps
    print('epsilon used in DM: ' + str(10 * eps))
    # Create kernel
    W = np.exp((-(np.square(eucl_mtx))) / (10 * eps))

    # Normalize kernel by row
    P = W / W.sum(axis=1).reshape(-1, 1)  # Markov Matrix

    # computing SVD decomposition of P
    u, s, vt = svds(P, k=kwargs['dim'], which='LM')

    # sorting:
    sort_mask = np.flip(np.argsort(s))

    s_sorted = s[sort_mask]
    u_sorted = u[:, sort_mask]

    logging.debug('Finish DM')
    return u_sorted[:, 1:] * s_sorted[1:]

