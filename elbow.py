import numpy as np
from sklearn.cluster import MiniBatchKMeans

def elbow_tuner(data, k_to_try):
    '''
        Perform elbow tuning for k-means clustering
        
        Input:
            data: MxN matrix --- M observations of N features 
            k_to_try: (1D) iterable containing values of K to try
        
        Return:
            intra_cluster_ss: 1D array containing mean squared centroid-observation distances, for each K
    '''
    
    # Result variable here!
    # To compute each value: given a K, compute the (Euclidean) separation
    # between each observation and its assigned cluster, then average over all observations
    intra_cluster_ss = np.zeros(shape=(len(k_to_try),))
    
    # For each K to try...
    for ix, K in enumerate(k_to_try):
        # Fit data to K clusters
        clusterer = MiniBatchKMeans(n_clusters=K, n_init=5)
        clusterer.fit(data)
        
        # Extract cluster centroids and the assignment each observation received
        cluster_centers = clusterer.cluster_centers_
        cluster_assignments = clusterer.labels_
        M = data.shape[0] # Total number of observations
        
        # For each K, define variable to record the mean squared centroid-observation distance
        ss_total = 0
        
        # For each cluster, computes the squared distance between its members and centroid
        for k in range(K):
            center = cluster_centers[k]
            cluster_assignment_mask = (cluster_assignments == k)
            
            # Obtain squared centroid-observation distances
            ss_k = np.sum( np.power(data[cluster_assignment_mask,:] - center, 2.0), axis=0)
            
            # Update the global mean of squared centroid-observation distances
            # Divide by M to avoid overflow (just in case vectors aren't normalized to begin with)
            ss_k = np.sum(ss_k)/M            
            ss_total += ss_k
        
        intra_cluster_ss[ix] = ss_total
    
    return intra_cluster_ss