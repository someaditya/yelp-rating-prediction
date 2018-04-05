import json
import nltk

nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
from nltk import pos_tag, word_tokenize
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


def tokenizeReview(reviewList):
    tokenizedWords = {}
    for review in reviewList:
        tokenizedWords[review[0]] = word_tokenize(review[1])
    # print(tokenizedWords)
    return tokenizedWords


def buildLexicon(tokenizedWords):
    lexicon = set()
    i = 1
    for i in range(1, len(tokenizedWords) + 1):
        lexicon.update(tokenizedWords[i])
    return lexicon


def tf(word, tokenizedWords):
    return tokenizedWords.count(word)


def createTfIdfMatrix(tokenizedWords):
    lexicon = buildLexicon(tokenizedWords)
    tf_vector = {}
    for i in range(1, len(tokenizedWords) + 1):
        tf_vector[i] = [tf(word, tokenizedWords[i]) for word in lexicon]
    return lexicon, tf_vector


def createTags(dictSent):
    tags = dictSent.values()
    return tags


def svcclassification(trainVecs, trainTags):
    clf = OneVsRestClassifier(SVC(C=1, kernel='linear', gamma=1, verbose=False, probability=False))
    clf.fit(trainVecs, trainTags)
    print "Classifier Trained..."
    predicted = cross_validation.cross_val_predict(clf, trainVecs, trainTags, cv=5)
    print "Cross Fold Validation Done..."
    print "accuracy score: ", metrics.accuracy_score(trainTags, predicted)
    print "precision score: ", metrics.precision_score(trainTags, predicted, pos_label=None, average='weighted')
    print "recall score: ", metrics.recall_score(trainTags, predicted, pos_label=None, average='weighted')
    print "classification_report: \n ", metrics.classification_report(trainTags, predicted)
    print "confusion_matrix:\n ", metrics.confusion_matrix(trainTags, predicted)
    fig, ax = plot_confusion_matrix(conf_mat=metrics.confusion_matrix(trainTags, predicted))
    plt.show()


def nbclassification(trainVecs, trainTags):
    clf = MultinomialNB()
    clf.fit(trainVecs, trainTags)
    print "Classifier Trained..."
    predicted = cross_validation.cross_val_predict(clf, trainVecs, trainTags, cv=5)
    print "Cross Fold Validation Done..."
    print "accuracy score: ", metrics.accuracy_score(trainTags, predicted)
    print "precision score: ", metrics.precision_score(trainTags, predicted, pos_label=None, average='weighted')
    print "recall score: ", metrics.recall_score(trainTags, predicted, pos_label=None, average='weighted')
    print "classification_report: \n ", metrics.classification_report(trainTags, predicted)
    print "confusion_matrix:\n ", metrics.confusion_matrix(trainTags, predicted)
    fig, ax = plot_confusion_matrix(conf_mat=metrics.confusion_matrix(trainTags, predicted))
    plt.show()

    return


def lrclassification(trainVecs, trainTags):
    clf = LogisticRegression()
    clf.fit(trainVecs, trainTags)
    print "Classifier Trained..."
    predicted = cross_validation.cross_val_predict(clf, trainVecs, trainTags, cv=5)
    print "Cross Fold Validation Done..."
    print "accuracy score: ", metrics.accuracy_score(trainTags, predicted)
    print "precision score: ", metrics.precision_score(trainTags, predicted, pos_label=None, average='weighted')
    print "recall score: ", metrics.recall_score(trainTags, predicted, pos_label=None, average='weighted')
    print "classification_report: \n ", metrics.classification_report(trainTags, predicted)
    print "confusion_matrix:\n ", metrics.confusion_matrix(trainTags, predicted)
    fig, ax = plot_confusion_matrix(conf_mat=metrics.confusion_matrix(trainTags, predicted))
    plt.show()

    return


def removeStopwords(tokenizedWords):
    for i in range(1, len(tokenizedWords) + 1):
        filteredWords = [word for word in tokenizedWords[i] if word not in stopwords.words('english')]
        tokenizedWords[i] = filteredWords
    # print(tokenizedWords)
    return tokenizedWords


def stemWords(tokenizedWords):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    for i in range(1, len(tokenizedWords) + 1):
        # filteredWords=[word for word in tokenizedWords[i] if word not in stopwords.words('english')]
        for word in tokenizedWords[i]:
            # print(filteredWords)
            filteredWords = [word for word in tokenizedWords[i] if word in stemmer.stem(word)]
            tokenizedWords[i] = filteredWords
    return tokenizedWords


def naivebayesfunction(tokenizedWords, dictSent, dictStar):
    print"#" * 70
    print"\n NB Classification without any processing"
    print "#" * 70
    lexicon, tfVector = createTfIdfMatrix(tokenizedWords)
    print "TF Matrix Created..."
    print "length of vector : ", len(tfVector[1])
    tags = createTags(dictSent)
    trainVecs = np.array(tfVector.values())
    trainTags = np.array(tags)
    nbclassification(trainVecs, trainTags)
    print"#" * 70
    print"\n NB Classification after removing stop words"
    print "#" * 70
    tokenizedWords = removeStopwords(tokenizedWords)
    lexicon, tfVector = createTfIdfMatrix(tokenizedWords)
    print "TF Matrix Created..."
    print "length of vector : ", len(tfVector[1])
    tags = createTags(dictSent)
    trainVecs = np.array(tfVector.values())
    trainTags = np.array(tags)
    nbclassification(trainVecs, trainTags)
    print"#" * 70
    print"\n NB Classification after removing stop words+stemming"
    print "#" * 70
    # tokenizedWords = removeStopwords(tokenizedWords)
    tokenizedWords = stemWords(tokenizedWords)
    lexicon, tfVector = createTfIdfMatrix(tokenizedWords)
    print "TF Matrix Created..."
    print "length of vector : ", len(tfVector[1])
    tags = createTags(dictSent)
    trainVecs = np.array(tfVector.values())
    trainTags = np.array(tags)
    nbclassification(trainVecs, trainTags)
    print"#" * 70
    print "NB Classification into 5 Classes"
    print"#" * 70
    tags = createTags(dictStar)
    trainTags = np.array(tags)
    nbclassification(trainVecs, trainTags)


def svmfunction(tokenizedWords, dictSent, dictStar):
    print"\n SVC Classification without any processing"
    print "#" * 70
    lexicon, tfVector = createTfIdfMatrix(tokenizedWords)
    print "TF Matrix Created..."
    print "length of vector : ", len(tfVector[1])
    tags = createTags(dictSent)
    trainVecs = np.array(tfVector.values())
    trainTags = np.array(tags)
    svcclassification(trainVecs, trainTags)
    print"\n SVC Classification after removing stop words"
    print "#" * 70
    tokenizedWords = removeStopwords(tokenizedWords)
    lexicon, tfVector = createTfIdfMatrix(tokenizedWords)
    print "TF Matrix Created..."
    print "length of vector : ", len(tfVector[1])
    tags = createTags(dictSent)
    trainVecs = np.array(tfVector.values())
    trainTags = np.array(tags)
    svcclassification(trainVecs, trainTags)
    print"#" * 70
    print"\n SVC Classification after removing stop words+stemming"
    tokenizedWords = stemWords(tokenizedWords)
    lexicon, tfVector = createTfIdfMatrix(tokenizedWords)
    print "TF Matrix Created..."
    print "length of vector : ", len(tfVector[1])
    tags = createTags(dictSent)
    trainVecs = np.array(tfVector.values())
    trainTags = np.array(tags)
    svcclassification(trainVecs, trainTags)
    print "SVC Classification into 5 Classes"
    tags = createTags(dictStar)
    trainTags = np.array(tags)
    svcclassification(trainVecs, trainTags)

def logisticfunction(tokenizedWords, dictSent, dictStar):
    print"\n SVC Classification without any processing"
    print "#" * 70
    lexicon, tfVector = createTfIdfMatrix(tokenizedWords)
    print "TF Matrix Created..."
    print "length of vector : ", len(tfVector[1])
    tags = createTags(dictSent)
    trainVecs = np.array(tfVector.values())
    trainTags = np.array(tags)
    lrclassification(trainVecs, trainTags)
    print"\n NB Classification after removing stop words"
    print "#" * 70
    tokenizedWords = removeStopwords(tokenizedWords)
    lexicon, tfVector = createTfIdfMatrix(tokenizedWords)
    print "TF Matrix Created..."
    print "length of vector : ", len(tfVector[1])
    tags = createTags(dictSent)
    trainVecs = np.array(tfVector.values())
    trainTags = np.array(tags)
    lrclassification(trainVecs, trainTags)
    print"\n SVC Classification after removing stop words+stemming"
    tokenizedWords = stemWords(tokenizedWords)
    lexicon, tfVector = createTfIdfMatrix(tokenizedWords)
    print "TF Matrix Created..."
    print "length of vector : ", len(tfVector[1])
    tags = createTags(dictSent)
    trainVecs = np.array(tfVector.values())
    trainTags = np.array(tags)
    lrclassification(trainVecs, trainTags)
    print "SVC Classification into 5 Classes"
    tags = createTags(dictStar)
    trainTags = np.array(tags)
    lrclassification(trainVecs, trainTags)




def main():
    import re
    f = open("review.json")

    line = f.readline()
    dictSent = {}
    dictStar = {}
    reviewList = []
    i = 1
    sentiment = 'negative'
    ps = PorterStemmer()
    while line:
        line = f.readline()
        review = json.loads(line)
        index = i
        star = review["stars"]
        text = review["text"]
        text = text.lower()
        #		text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)

        if star > 3:
            sentiment = '1.0'  # positive
        else:
            sentiment = '0.0'  # negative
        dictSent[index] = sentiment
        dictStar[index] = star
        reviewList.append([index, text])
        i += 1
        if i == 5000:
            break
    f.close()

    print "Dataset Loaded..."
    tokenizedWords = tokenizeReview(reviewList)
    print "Reviews Tokenized..."
    #naivebayesfunction(tokenizedWords, dictSent, dictStar)
    #svmfunction(tokenizedWords, dictSent, dictStar)
    logisticfunction(tokenizedWords, dictSent, dictStar)

# #reviewPosTag=tokenizeReview(reviewList)

if __name__ == "__main__":
    main()
