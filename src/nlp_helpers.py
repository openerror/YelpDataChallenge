from typing import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Helper functions for selecting search engine results
def get_top_values(lst, n, labels):
    '''
    INPUT: LIST, INTEGER, LIST
    OUTPUT: LIST

    Given a list of values, find the indices with the highest n values.
    Return the labels for each of these indices.

    e.g.
    lst = [7, 3, 2, 4, 1]
    n = 2
    labels = ["cat", "dog", "mouse", "pig", "rabbit"]
    output: ["cat", "pig"]
    '''
    # np.argsort by default sorts values in ascending order
    return [labels[i] for i in np.argsort(lst)[::-1][:n]]  


def get_bottom_values(lst, n, labels):
    '''
    INPUT: LIST, INTEGER, LIST
    OUTPUT: LIST

    Given a list of values, find the indices with the lowest n values.
    Return the labels for each of these indices.

    e.g.
    lst = [7, 3, 2, 4, 1]
    n = 2
    labels = ["cat", "dog", "mouse", "pig", "rabbit"]
    output: ["mouse", "rabbit"]
    '''
    return [labels[i] for i in np.argsort(lst)[:n]]


def get_n_most_frequent_tokens(reviews: Iterable[str], 
                               n: int = 20, 
                               max_df: float = 0.98) -> Iterable[Tuple[str, int]]:
    """ Get the n-most frequent unigrams, given a document-frequency cap """
    
    count_vec = CountVectorizer(max_df=max_df)
    per_token_counts = np.asarray(count_vec.fit_transform(reviews).sum(axis=0))[0]
    ix_to_token = {ix:tk for tk, ix in count_vec.vocabulary_.items()}
        
    i = 0    
    for token_ix in reversed(np.argsort(per_token_counts)):
        if i < n:
            i += 1
            yield ix_to_token[token_ix], per_token_counts[token_ix]