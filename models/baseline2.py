import numpy as np
from collections import Counter

from pprint import pprint
from time import time
import pdb
import cPickle as pickle

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVR

import os.path

from prepare_data import *
from baseline import *


VECTORS_FILE = '../data_processing/data_split_by_speech_nonzero_vectors_only.pickle'
labels_filename2 = '../scraping/fixed_people_with_vectors_234'
corpus_filename = '../data_processing/data_all.pickle'

def run_classifier():
	# if VECTORS_FILE is not found, run this
	if not os.path.isfile(VECTORS_FILE):
		data = load_corpus(corpus_filename)
		save_data_split_by_speech(data, labels_filename2, VECTORS_FILE)

	# list of dicts
    data = load_corpus(VECTORS_FILE)

    (X, parties, vectors, speech_ids) = make_data_split_by_speech(data)
    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test) = train_test_split(X, parties, vectors)

    predict_party((X_train, X_test, parties_train, parties_test, vectors_train, vectors_test))
    predict_20_attr_classification((X_train, X_test, parties_train, parties_test, vectors_train, vectors_test))


if __name__ == "__main__":
    run_classifier()

