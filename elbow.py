def elbow_tuner(data, k_to_try):
    '''
        Perform elbow tuning for k-means clustering
        
        Input:
            data: MxN matrix --- M observations of N features 
            k_to_try: (1D) iterable containing values of K to try
        
        Return:
            intra_cluster_ss: 1D array containing averaged intra-cluster sum-of-squares, for each K
    '''
    
    # Result variable: Averaged intra-cluster sum-of-squares, for each K
    # To compute each value: given a K, compute the (Euclidean) separation
    # between each observation and its assigned cluster, then average over all observations
    intra_cluster_ss = np.zeros(shape=(len(k_to_try),))
    
    for ix, K in enumerate(k_to_try):
        # Fit data to K clusters
        clusterer = MiniBatchKMeans(n_clusters=K)
        clusterer.fit(data)
        
        # Extract cluster centroids and the assignment each observation received
        cluster_centers = clusterer.cluster_centers_
        cluster_assignments = clusterer.labels_
        M = data.shape[0] # Total number of observations
        
        # Compute the intra-cluster sum-of-squares for each cluster
        # Divide by M in advance to avoid overflow (not likely, but just in case)
        ss_total = 0
        
        for k in range(K):
            center = cluster_centers[k]
            cluster_assignment_mask = (cluster_assignments == k)
            
            # Obtain sum-of-squares for each observation
            ss_k = np.sum( np.power(data[cluster_assignment_mask,:] - center, 2.0), axis=0)
            
            # Add up sum-of-squares for ALL observations
            ss_k = np.sum(ss_k)/M            
            ss_total += ss_k
        
        intra_cluster_ss[ix] = ss_total
    
    return intra_cluster_ss