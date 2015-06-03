import numpy as np
from collections import Counter

from pprint import pprint
from time import time
import pdb
import pickle

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVR

from prepare_data import *


corpus_filename = '../data_processing/data_all.pickle'
labels_filename = '../scraping/people_with_vectors_746'

count_vect = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer()


# Returns the mean-squared error between 2 vectors
def MSE(a, b):
    return np.mean((a - b) ** 2)

# Clips x to be in the range [-k, k]
def clip(x, k):
    for i in xrange(len(x)):
        if x[i] > k:
            x[i] = k
        elif x[i] < -1.0 * k:
            x[i] = -1.0 * k
    return x

# http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
def print_top20_multiclass(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    print "Printing top 10 features..."
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top20 = np.argsort(np.abs(clf.coef_[i]))[-20:]
        print "%s: %s" % (class_label, " ".join(feature_names[j] for j in top20))

# http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
def print_top20_binary(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)


def save_transformed_data(X_train, X_dev, X_test):
    for (dataset_label, X) in [("train", X_train), ("dev", X_dev), ("test", X_test)]:
        count_vect = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2))
        X_counts = count_vect.fit_transform(X)
        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_counts)

        output = open('X_tfidf_%s.pickle' % dataset_label, 'wb')
        pickle.dump(X_tfidf, output)
        output.close()

# Returns [X_train, X_dev, X_test]
def load_transformed_data():
    print "Loading tfidf-transformed data..."
    result = []
    for dataset_label in ["train", "dev", "test"]:
        pkl_file = open('X_tfidf_%s.pickle' % dataset_label, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        result.append(data)
    return result



def predict_party((X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test)):
    print "========= Party affiliation ========="

    print "TFIDF with bigrams"

    #clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=10, n_jobs=-1, random_state=42)

    #text_clf = clf.fit(X_tfidf_train, parties_train)

    vect = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2))
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=10, n_jobs=-1, random_state=42)

    X_train_tfidf = vect.fit_transform(X_train)
    text_clf = clf.fit(X_train_tfidf, parties_train)

    # dev
    X_dev_tfidf = vect.transform(X_dev)
    predicted = text_clf.predict(X_dev_tfidf)
    acc = np.mean(predicted == parties_dev)   
    print "Accuracy is %f" % acc
    print metrics.confusion_matrix(parties_dev, predicted)

    print "Most informative features..."
    print_top20_binary(vect, text_clf)


    # print "\nTFIDF, unigrams"

    # pipeline = Pipeline([ \
    #     ('vect', CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1))), \
    #     ('tfidf', TfidfTransformer()), \
    #     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=10, n_jobs=-1, random_state=42)) \
    # ])

    # text_clf = pipeline.fit(X_train, parties_train)

    # # dev
    # predicted = text_clf.predict(X_dev)
    # acc = np.mean(predicted == parties_dev)   
    # print "Accuracy is %f" % acc
    # print metrics.confusion_matrix(parties_dev, predicted)


    # print "\nNO TFIDF, unigrams"

    # pipeline = Pipeline([ \
    #     ('vect', CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1))), \
    #     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=10, n_jobs=-1, random_state=42)) \
    # ])

    # text_clf = pipeline.fit(X_train, parties_train)

    # # dev
    # predicted = text_clf.predict(X_dev)
    # acc = np.mean(predicted == parties_dev)   
    # print "Accuracy is %f" % acc
    # print metrics.confusion_matrix(parties_dev, predicted)


# Regression
def predict_20_attr((X_tfidf_train, X_tfidf_dev, X_tfidf_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test)):
    clf = SVR(kernel='linear', degree=3, gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=50)

    for i in xrange(20):
        print "\n========= Attribute %d =========" % i

        text_clf = clf.fit(X_tfidf_train, vectors_train[:, i])

        # dev
        correct_output = vectors_dev[:, i]
        predicted = text_clf.predict(X_tfidf_dev)

        # clip the predictions to be in (-2, 2)
        predicted = clip(predicted, 2.0)

        mse = MSE(predicted, correct_output)   
        print "  MSE is %f" % mse


def predict_20_attr_all_zeros((X_tfidf_train, X_tfidf_dev, X_tfidf_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test)):
    print "If we were predicting all zeros..."

    for i in xrange(20):
        print "\n========= Attribute %d =========" % i

        # dev
        correct_output = vectors_dev[:, i]
        predicted = np.zeros((correct_output.shape[0],))

        mse = MSE(predicted, correct_output)   
        print "  MSE is %f" % mse


def predict_20_attr_classification((X_tfidf_train, X_tfidf_dev, X_tfidf_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test)):
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=10, n_jobs=-1, random_state=42)

    for i in xrange(20):
        print "\n========= Attribute %d =========" % i

        text_clf = clf.fit(X_tfidf_train, vectors_train[:, i])

        # dev
        predicted = text_clf.predict(X_tfidf_dev)
        acc = np.mean(predicted == vectors_dev[:, i])   
        print "Accuracy is %f" % acc
        print metrics.confusion_matrix(vectors_dev[:, i], predicted)


def run_classifier():
    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename)
    (X, parties, vectors, names) = make_data(data, labels)
    (X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test) = train_test_split(X, parties, vectors)

    ctr = Counter(parties_train)
    cdev = Counter(parties_dev)
    print ctr
    print cdev

    #[X_tfidf_train, X_tfidf_dev, X_tfidf_test] = load_transformed_data()

    predict_party((X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test))
    #predict_20_attr((X_tfidf_train, X_tfidf_dev, X_tfidf_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test))
    #predict_20_attr_classification((X_tfidf_train, X_tfidf_dev, X_tfidf_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test))
    #predict_20_attr_all_zeros((X_tfidf_train, X_tfidf_dev, X_tfidf_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test))


def tune_params():
    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename)
    (X, parties, vectors, names) = make_data(data, labels)
    (X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test) = train_test_split(X, parties, vectors)

    pipeline = Pipeline([ \
        ('vect', CountVectorizer(strip_accents='ascii', stop_words='english')), \
        ('tfidf', TfidfTransformer()), \
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, n_jobs=1, random_state=42)) \
    ])

    ctr = Counter(parties_train)
    cdev = Counter(parties_dev)
    print ctr
    print cdev

    # multiprocessing requires the fork to happen in a __main__ protected
    # block
    parameters = {
        'vect__max_df': [0.75, 1.0],
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': [(1, 1), (1, 2)],  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': [1e-2, 1e-3, 1e-4, 1e-5],
        'clf__penalty': ['l2'],
        'clf__n_iter': [5],
    }

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, parties_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))



if __name__ == "__main__":
    run_classifier()

    # data = load_corpus(corpus_filename)
    # labels = get_labels(labels_filename)
    # (X, parties, vectors, names) = make_data(data, labels)
    # (X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test) = train_test_split(X, parties, vectors)

    # save_transformed_data(X_train, X_dev, X_test)
