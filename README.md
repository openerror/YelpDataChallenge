# Sentiment Analysis & Restaurant Recommendation System
## Overview

This project utilizes **real-world data** from the 12th [Yelp Data Challenge](https://www.yelp.com/dataset/challenge), where Yelp
releases several years worth of data from its operations. The data set includes reviews of businesses by customers, the photos taken, check-in instances, and detailed attributes of each business.

This project focuses on restaurants in Las Vegas, and provides
1. a sentiment analysis model (positive *v.s.* negative reviews)
2. a recommendation system for restaurants 
3. a clustering study of restaurants with perfect/disastrous ratings (**pending**)


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

Note that stop words are not filtered out, because the available lists contain words that
may encode sentiments.

### Recommendation System
`Recommender.ipynb`

For the recommendation system ...

### Clustering of Restaurants
`Clustering.ipynb`

## File Directory