# Sentiment Analysis & Restaurant Recommendation System
## Introduction

Stack: Python 3.7, Pandas, Matplotlib, [scikit-learn](http://scikit-learn.org/), [scikit-surprise](http://surpriselib.com/)

Utilizing [**real-world data**]((https://www.yelp.com/dataset/challenge)) released by Yelp, a review platform for local
businesses [across the world](https://www.yelp.com/locations), this project has built for
*restaurants in Las Vegas*
1. a sentiment analysis model (positive *v.s.* negative reviews)
2. a recommendation system for unvisited/unrated restaurants
3. a clustering study of restaurants with perfect/disastrous ratings

To try out this repository, download all the code, and then place the Yelp data into the
directory `yelp_dataset`. Then run `Preprocessing.ipynb` first to extract
the necessary data from the enormous raw set.

### Summary of Results:
1. Using (**logistic regression, random forest**), achieved **accuracies of (84%, 80%)** on
test set. Since the labels are not too imbalanced, accuracy is an appropriate metric. In case
there are questions about "poor performance", see discussion in next section.

2. Built a recommendation system based on explicit restaurant ratings; item-item collaborative
filtering and [Non-Negative Matrix Factorization (NMF)]((http://www.albertauyeung.com/post/python-matrix-factorization/)) attempted. For NMF, RMSE of 1.2 achieved on masked ratings; in comparison, the winner of the famous
Netflix competition achieved a RMSE of 0.9.

3. Confirmed empirically some common-sense assumptions: there is a casual relationship between
hygiene, service quality, and customer retention

Please see below for details on project methodology and the purpose of each file.

## File Directory & Summary of Techniques
### Sentiment Analysis
`NLP.ipynb`
Each review is transformed from raw text to "term frequency-inverse document frequency"
(**Tf-idf**) vectors. A **logistic regression** classifier is built as the baseline, followed
by a **random forest** for hopefully improved performance; **grid-search cross validation** is performed
whenever time and computing resources permit. After building each model, feature importances
are explored and interpreted accordingly.

Note that some stop words are not filtered out, because they contain words that may encode
sentiments, e.g. don't can't ...

On the other hand, one may criticize my models for achieving just 80-85% accuracy, but here are some [empirical
findings](https://blog.paralleldots.com/data-science/breakthrough-research-papers-and-models-for-sentiment-analysis/) to put that into perspective. When classifying sentiments in IMDB
benchmark data, which is of similar size to our present data set, state-of-the-art deep learning models
were required to achieve an accuracy of 90+%. It may not always be worthwhile to implement complex models
that are difficult to interpret, just to chase after a 10% increase in accuracy.

Machine learning models are provided by [scikit-learn](http://scikit-learn.org/).

### Recommendation System
`Build-Recommender.ipynb`, `Recommender.py`
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
is provided by [scikit-surprise](http://surpriselib.com/).

### Clustering of Restaurants
`Clustering.ipynb`

Performed K-means clustering of restaurant reviews (encoded as Tf-idf vectors), explored recurring themes expressed by restaurant goers. Tuned clustering output using the elbow method.

Tried clustering all reviews simultaneously, and also only the ones with the lowest star rating (one). Attempted to identify what makes popular restaurants popular, and what constitutes restaurants that receive the worst ratings.

Clustering models are provided by [scikit-learn](http://scikit-learn.org/).
