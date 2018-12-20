# Sentiment Analysis & Restaurant Recommendation System
## Overview

This project utilizes **real-world data** from the 12th [Yelp Data Challenge](https://www.yelp.com/dataset/challenge); Yelp is a review platform for local
businesses [across the world](https://www.yelp.com/locations). The data set includes reviews
of businesses by customers, the photos taken, check-in instances, and detailed attributes 
of each business.

This project focuses on restaurants in Las Vegas, and provides
1. a sentiment analysis model (positive *v.s.* negative reviews)
2. a recommendation system for restaurants
3. a clustering study of restaurants with perfect/disastrous ratings 


To try out this repository, download all the code, and place the Yelp data into the
directory `yelp_dataset`. Then run `Preprocessing.ipynb` first to extract
the necessary data from the enormous raw set.

## File Directory & Summary of Techniques
### Sentiment Analysis
`NLP.ipynb`
Each review is transformed from raw text to "term frequency-inverse document frequency"
(**Tf-idf**) vectors. A **logistic regression** classifier is built as the baseline, followed
by a **random forest** for improved performance; **grid-search cross validation** is performed
whenever time and computing resources permit. After building each model, feature importances
are explored and interpreted accordingly.

Note that stop words are not filtered out, because they contain words that may encode
sentiments, e.g. don't can't ...

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
each review receives. For simplicity, we have only utilized the former; it only takes a
simple variable name change to use the other!

### Clustering of Restaurants
`Clustering.ipynb`

Performed K-means clustering of restaurant reviews (encoded as Tf-idf vectors), explored recurring themes expressed by restaurant goers. Tuned clustering output using the elbow method.

Tried clustering all reviews simultaneously, and also only the ones with the lowest star rating (one). Attempted to identify what makes popular restaurants popular, and what constitutes restaurants that receive the worst ratings.