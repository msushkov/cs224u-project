import numpy as np
from collections import Counter

from pprint import pprint
from time import time

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

from prepare_data import *


corpus_filename = '../data_processing/data_all.pickle'
labels_filename = '../scraping/all_people'


def run_classifier():
    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename)
    (X, parties, vectors) = make_data(data, labels)
    (X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test) = train_test_split(X, parties, vectors)

    pipeline = Pipeline([ \
        ('vect', CountVectorizer(strip_accents='ascii', stop_words='english')), \
        ('tfidf', TfidfTransformer()), \
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, n_jobs=-1, random_state=42)) \
    ])

    ctr = Counter(parties_train)
    cdev = Counter(parties_dev)
    print ctr
    print cdev

    text_clf = pipeline.fit(X_train, parties_train)

    # dev
    predicted = text_clf.predict(X_dev)
    acc = np.mean(predicted == parties_dev)   
    print "accuracy is %f" % acc
    #print(metrics.classification_report(parties_dev, predicted))
    print metrics.confusion_matrix(parties_dev, predicted)

if __name__ == "__main__":
    #run_classifier()

    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename)
    (X, parties, vectors) = make_data(data, labels)
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
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1)

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



