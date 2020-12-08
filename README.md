# Sentiment Analysis & Restaurant Recommendation System
## Introduction

Stack: Python 3.7 - 3., Pandas, Matplotlib, [scikit-learn](http://scikit-learn.org/), [scikit-surprise](http://surpriselib.com/).
After serious searching I am glad to have come upon scikit-surprise --- it may be the only single-machine, Python recommendation system library that accounts for [user/item biases](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF) in (explicit) ratings.

Utilizing [real-world data](https://www.yelp.com/dataset/challenge) released by Yelp, a review platform for local
businesses [across the world](https://www.yelp.com/locations), this project has built for
*restaurants in Las Vegas*
1. a sentiment analysis model (positive *v.s.* negative reviews)
2. a recommendation system for unvisited/unrated restaurants
3. a NLP-enabled case study of two restaurants with perfect/disastrous ratings

To try out this repository, download all the code, and then place the Yelp data into the
directory `yelp_dataset`. Then run `Preprocessing.ipynb` first to extract
the necessary data from the enormous raw set.

### Summary of Results:
1. Using (**logistic regression** and **naive Bayes**) classifiers, achieved **(84%, 82%) accuracies** on
test set. Since the labels are not too imbalanced, accuracy is an appropriate metric. In case
there are questions about "poor performance", see discussion in next section.

2. Built a recommendation system based on explicit restaurant ratings; item-item collaborative
filtering and [Non-Negative Matrix Factorization (NMF)]((http://www.albertauyeung.com/post/python-matrix-factorization/)) attempted. For NMF, **RMSE of 1.2** achieved on masked ratings; in comparison, the winner of the famous
Netflix competition achieved a RMSE of 0.9.

3. Confirmed empirically some common-sense assumptions: there *is* a casual relationship between
hygiene, service quality, and customer retention

Please see below for details on project methodology and the purpose of each file.

## File Directory & Summary of Techniques
### Search Engine for Similar Reviews + Sentiment Analysis
####  `SentimentAnalysis_Search.ipynb`
Each review is transformed from raw text to "term frequency-inverse document frequency"
(**Tf-idf**) vectors. Note that some stop words are not filtered out, because they contain words that may encode
sentiments, e.g. don't can't ...

Using the Tf-idf vectors, a basic search engine was built; it computes the **cosine similarity** between
a query string and the review corpus, then returns from the corpus those reviews that
are the most similar. Cosine similarity is used, because it is scale-invariant with respect to
the lengths of both the query and the corpus documents. Using this search engine, one
can quickly find, e.g. reviews containing the same criticisms.

Then, **logistic regression** and **naive Bayes** classifiers were
built as 80%-accuracy baselines. I hoped to follow up with a random forest model, but
unfortunately my laptop couldn't complete **randomized hyperparameter search** + **3-fold cross validation**
after hours.

To put the classification accuracies into perspective, here are some [empirical
findings](https://blog.paralleldots.com/data-science/breakthrough-research-papers-and-models-for-sentiment-analysis/). When classifying sentiments in IMDB benchmark data, which is of similar size to our present data set, state-of-the-art deep learning models were required to achieve an accuracy of 90+%. All in all, 80+% accuracy is already quite alright;
it may not be worthwhile to implement more complicated models that are difficult to interpret.

Machine learning models are provided by [scikit-learn](http://scikit-learn.org/).

### Recommendation System
#### `Build-Recommender.ipynb`, `src/Recommender.py`
Two approaches --- **item-item collaborative** filtering and **Non-negative Matrix
Factorization** [(NMF)](http://www.albertauyeung.com/post/python-matrix-factorization/) are
explored. The top recommendations for a given user are compared with the restaurants he/she
reviewed, in order to ensure that the recommendations make sense. To evaluate more rigorously
the performance of each approach, I will have to create a hold-out test set (pending).

One key step to building a recommendation system is to form a utility matrix, which
records users' preferences for each item (restaurant). This project has demonstrated two
approaches at quantitatively encoding "preference": one using the star rating attached
to each review, and the other weighs the star rating according to the number of 'useful' votes
each review receives. For simplicity, we have only utilized the former, but it only takes a
simple variable name change to use the other! In a production system, use A/B testing
to decide between the two.

Item-item collaborative filtering is implemented using custom code, while NMF functionality
is provided by [scikit-surprise](http://surpriselib.com/). For the NMF recommendation system,
I have utilized scikit-surprise's built-in functionality for measuring performance:
by masking some non-zero entries in the utility matrix and let the trained model "predict" them,
we can calculate a RMSE which serves as a measure for the quality of recommendations.

### Discovering Reviews
#### `TopicDiscovery.ipynb`
Applied bag-of-word techniques to quickly extract and visualize diners' thoughts on two
famous restaurant franchises: McDonald's and Gordon Ramsay Hell's Kitchen. Interesting how
much one can accomplish without machine learning, but acquiring `word2vec` embeddings of
reviews and clustering the resulted vectors should be an interesting extension.

![gordon ramsay hell kitchen review trends](gordon.png)
