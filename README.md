 # Yelp-Rating-Prediction
The Yelp dataset is a subset of our businesses, reviews, and user data for use in personal, educational, and academic purposes.
The dataset contains around 5,200,000 reviews,74,000 businesses,00,000 pictures,1 metropolitan areas, 100,000 tips by 1,300,000 users and 1.2 million business attributes like hours, parking, availability, and ambience.

![Yelp Logo](/image/yelp_logo.png)

Aggregated check-ins over time for each of the 174,000 businesses
The goal of this project was to predict reviews' star ratings on Yelp using the review text. We built the following models that perform text analysis on review data to predict the rating stars.

## Feature Selection
1. Basic without any Filtering
2. Stop Word Removal
3. Stemming using Snowball Stemmer

## Machine Learning Algorithms used
1. Logistic Regression
2. Support Vector Machine
3. Naive Bayes

## Data and Preprocessing
"Yelp Dataset Challenge” dataset has been selected to study in this research. The Yelp dataset has
been published to be studied on photo classification, graph mining and natural language processing &
sentiment analysis.A python script is implemented to parse the reviews JSON data file. During the parsing process, only star ratings
and text reviews are taken into consideration, all the other information is ignored. The raw data is stored
in three different dictionaries on the basis of review, sentiments and stars.
In the data pre-processing phase, the entire text is converted into lowercase to reduce redundancy in
subsequent feature selection. Several regular expressions are used, followed by the removal of punctuations

## Feature Selection 
In the scope of this research, a unique feature set is built based on the user text reviews. In addition to this, some variations to
our process are implemented: (1) With no pre-processing or changes (2) Removing English stop words
(i.e. extremely common words) from the feature set using the stop word removal feature available in
Natural Language Toolkit (NLTK) Corpus (3) Stemming (i.e. reducing a word to its stem/root form) to
remove repetitive features using the Snowball Stemmer algorithm which is a built-in feature in NLTK.
and white spaces from the review text.
Accordingly, for the first basic sentiment analysis a simple rule is considered, if the star rating is greater
3 than 3 value 1.0 is assigned which s inferred as a ”Positive” sentiment and otherwise it was assigned 0.0
for ”Negative” sentiment.

Three different machine learning algorithms are implemented and examined: Naive Bayes, SVM and
Logistic Regression.

## Results

### Naive Bayes
Multinomianal-Naive Bayes is evaluated on 100,000 instances. The results are represented with precision,
recall and f1-score metrics. First, polarity of the reviews are observed (Fig. 2). Then same methods
are implemented on 5 classes which represent 5 stars (Fig. 3). The results are observed relatively high
for 2 classes polarity evaluation. However, a significant decrease is observed in the results for 5 classes.
This inference can be based on the fact that lexicons with 4 and 5 stars are relatively close and lexicons
with rating of 1,2 and 3 are relatively close.

![Yelp Logo](/image/ta1.png)

![Yelp Logo](/image/ta2.png)

### Support Vector Machine
Support Vector Machines is a discriminative classifier formally defined by a separating hyperplane. The
algorithm outputs an optimal hyperplane which categorizes new incoming instances, given labeled training
data.
![Yelp Logo](/image/ta5.png)

![Yelp Logo](/image/ta6.png)

### Logistic Reg3ession
The outcome is measured with a dichotomous variable (in
this case, t4o to five possible outcomes). The goal was to find the best fitting model to describe the relationship
between the dichotomous characteristic of interest (reviews) and a set of independent variables.

![Yelp Logo](/image/ta3.png)

![Yelp Logo](/image/ta4.png)

