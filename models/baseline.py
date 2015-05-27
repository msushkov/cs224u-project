import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

from prepare_data import *


corpus_filename = '../data_processing/data.pickle'
labels_filename = '../scraping/all_people'


def run_classifier():
	data = load_corpus(corpus_filename)
	labels = get_labels(labels_filename)
	(X, parties, vectors) = make_data(data, labels)
	(X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test) = train_test_split(X, parties, vectors)

	text_clf = Pipeline([ \
		('vect', CountVectorizer()), \
		('tfidf', TfidfTransformer()), \
		('clf', MultinomialNB()) \
	])

	text_clf = text_clf.fit(X_train, parties_train)

	# dev
	predicted = text_clf.predict(X_dev)
	acc = np.mean(predicted == parties_dev)   
	print "accuracy is %f" % acc
	print(metrics.classification_report(parties_dev, predicted))
	print metrics.confusion_matrix(parties_dev, predicted)
