from typing import *
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def elbow_tuner(data: np.ndarray, k_to_try: Iterable[int]) -> np.ndarray:
    '''
        Perform elbow tuning for k-means clustering
        
        Input:
            data: M observations of N features, M x N array 
            k_to_try: values of K to try
        
        Return:
            avg_ss: 1D array containing mean squared centroid-observation distances, for each K
    '''

    # Result variable here!
    avg_ss = np.zeros(shape=(len(k_to_try),))
    M = data.shape[0]

    for i, K in enumerate(k_to_try):
        # fit data to K clusters
        model = MiniBatchKMeans(n_clusters=K, n_init=5).fit(data)
        centriods = model.cluster_centers_
        cluster_labels = model.labels_

        """
        For each of the K clusters, do the following in order:
            1. find the Euclidean distances between cluster's centroid and EACH observation
            2. find the sum of 1), divided by M == data.shape[0]
            3. add 2) to avg_ss[i]
        After the for loop below terminates, avg_ss[i] contains the Euclidean
        distance between each observation in the *data* and its centroid, averaged.
        """
        for k in range(K):
            label_mask = (cluster_labels == k)
            # 1) Euclidean distance of EACH observation with centroid; shape (label_mask.sum(), 1)
            ss_k = np.sum(np.power(data[label_mask] - centriods[k], 2.), axis=1)
            # 2) and 3)
            avg_ss[i] += np.sum(ss_k) / M

    return avg_ss
