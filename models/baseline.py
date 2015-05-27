from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from prepare_data import *


corpus_filename = '../data_processing/data.pickle'
labels_filename = '../scraping/all_people'


def prepare_corpus():
	(X, parties, vectors) = get_corpus(corpus_filename, labels_filename)

	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(X)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)