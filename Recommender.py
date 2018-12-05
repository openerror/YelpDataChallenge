import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from time import time

class ItemItemRecommender(object):

    def __init__(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size

    def fit(self, ratings_mat):
        self.ratings_mat = ratings_mat
        self.n_users = ratings_mat.shape[0]
        self.n_items = ratings_mat.shape[1]
        self.item_sim_mat = cosine_similarity(self.ratings_mat.T)
        self._set_neighborhoods()

    def _set_neighborhoods(self):
        least_to_most_sim_indexes = np.argsort(self.item_sim_mat, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]

    def pred_one_user(self, user_id, report_run_time=False):
        start_time = time()
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[0]
        # Just initializing so we have somewhere to put rating preds
        out = np.zeros(self.n_items)
        for item_to_rate in range(self.n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)  # assume_unique speeds up intersection op            
            
            if len(relevant_items) == 0:
                out[item_to_rate] = 0.0
            else:
                out[item_to_rate] = np.dot(self.ratings_mat[user_id, relevant_items], \
                    self.item_sim_mat[item_to_rate, relevant_items]) / \
                    self.item_sim_mat[item_to_rate, relevant_items].sum()
                
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        cleaned_out = np.nan_to_num(out)
        return cleaned_out

    def pred_all_users(self, report_run_time=False):
        start_time = time()
        all_ratings = [
            self.pred_one_user(user_id) for user_id in range(self.n_users)]
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        return np.array(all_ratings)

    def top_n_recs(self, user_id, n):
        pred_ratings = self.pred_one_user(user_id)
        item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[0]
        unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                        if item not in items_rated_by_this_user]
        return unrated_items_by_pred_rating[-n:]

    
def displayHelper(user_index, preds, utility, business_info_df, N):
    '''
        Helper function for displaying results of restaurant recommendations
        
        user_index:
        preds:
        utility:
        df_recommender:
        
        Return Pandas DataFrame, containing information about the top N restaurants
    '''
    
    assert (user_index < utility.shape[0]), "ERROR"
    
    # Assumes that utility matrix is created by pandas.pivot_table
    user_ids = utility.index
    business_ids = utility.columns
    
    # Get user_id of the user, for whom we have made predictions
    current_user = user_ids[user_index]
    
    # Identify business that have NOT been reviewed, AND for which recommendations are made
    # In the utility matrix, businesses not reviewed have rating 0.0
    not_reviewed_index = np.nonzero( (preds > 0.0) & (utility.iloc[user_index,:] <= 1e-8) )
    not_reviewed_bid = business_ids[ not_reviewed_index ]
    
    # Isolate those predictions that
    # Get their business_ids
    relevant_preds = preds[not_reviewed_index].copy()
    relevant_bid = not_reviewed_bid[ np.argsort(relevant_preds)[-1:-(N+1):-1] ]
        
    return relevant_bid.values