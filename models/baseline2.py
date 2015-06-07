import numpy as np
from collections import Counter

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
from nltk.tokenize import word_tokenize
import os.path
from time import time

from gensim.models.doc2vec import Doc2Vec, LabeledSentence

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


# Combine the labels of all the politician's speeches to get a single prediction for a given politician
def combine_politician_speeches():
    # before doing the voting
    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename2)
    (X, parties, vectors, names) = make_data(data, labels)
    (X_train_1, X_test_1, parties_train_1, parties_test_1, vectors_train_1, vectors_test_1) = train_test_split(X, parties, vectors)

    name_to_index = {}
    for i, name in enumerate(names):
        name_to_index[name] = i

    # if VECTORS_FILE is not found, run this
    if not os.path.isfile(VECTORS_FILE):
        data = load_corpus(corpus_filename)
        save_data_split_by_speech(data, labels_filename2, VECTORS_FILE)

    # list of dicts
    data = load_corpus(VECTORS_FILE)

    X_new = [None] * len(X)
    parties_new = [None] * len(parties)
    vectors_new = [None] * len(vectors)

    # name -> tuple of (party_label, vector, text)
    data_per_politician = {}

    for curr_point in data:
        name = curr_point['name']
        vector = curr_point['vector']
        party_label = curr_point['party_label']
        speech_text = curr_point['speech_text']

        if name not in data_per_politician:
            data_per_politician[name] = []

        data_per_politician[name].append((party_label, vector, speech_text))

    
    # combine labels for each name
    for name in data_per_politician:
        curr_lst = data_per_politician[name]
        parties = [x[0] for x in curr_lst]
        most_frequent_party = Counter(parties).most_common(1)[0][0]

        new_vector = []

        for i in range(20):
            labels = [x[1][i] for x in curr_lst]
            most_frequent_label = Counter(labels).most_common(1)[0][0]
            new_vector.append(most_frequent_label)

        index = name_to_index[name]
        X_new[index] = curr_lst[2]
        parties_new[index] = most_frequent_party
        vectors_new[index] = new_vector

    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test) = train_test_split(X_new, parties_new, vectors_new)

    print "party affiliation..."
    acc = np.mean(parties_test_1 == parties_test)   
    print "test accuracy is %f" % acc

    acc = np.mean(parties_train_1 == parties_train)
    print "train accuracy is %f" % acc

    for i in xrange(20):
        print "\n========= Attribute %d =========" % i
        acc = np.mean(vectors_test_1[:, i] == vectors_test[:, i])   
        print "test ccuracy is %f" % acc
        acc = np.mean(vectors_train_1[:, i] == vectors_train[:, i])   
        print "train ccuracy is %f" % acc




# http://radimrehurek.com/2014/12/doc2vec-tutorial/
def train_paragraph_vector(num_epochs=10):
    print "In train_paragraph_vector()..."

    speech_ids = []

    # if VECTORS_FILE is not found, run this
    if not os.path.isfile(VECTORS_FILE):
        data = load_corpus(corpus_filename)
        save_data_split_by_speech(data, labels_filename2, VECTORS_FILE)

    # list of dicts
    data = load_corpus(VECTORS_FILE)
    speeches = []

    print "Loaded data. Creating labeled sentence objects..."

    for curr_point in data:
        speech_id = curr_point['speech_id']
        name = curr_point['name']
        vector = curr_point['vector']
        party_label = curr_point['party_label']
        speech_text = curr_point['speech_text']
        curr_speech = LabeledSentence(words=word_tokenize(speech_text), labels=[speech_id])
        speech_ids.append(speech_id)
        speeches.append(curr_speech)

    # doc2vec stuff
    model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    model.build_vocab(speeches)

    print "Starting training..."

    for epoch in range(num_epochs):
        print "Epoch %d" % epoch
        curr = time()

        model.train(speeches)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

        print "  time = %s mins" % str((time() - curr) / 60.0)

    print "Done training."

    # save the model
    model.save('model_0.025_decr_by_0.002_epochs_10.doc2vec')

    # save the speech ids file
    f = open('speech_ids.pickle', 'wb')
    pickle.dump(speech_ids, f)
    f.close()


def load_doc2vec_model_and_speech_ids(filename='model_0.025_decr_by_0.002_epochs_10.doc2vec'):
    return (Doc2Vec.load(filename), pickle.load(open('speech_ids.pickle', 'rb')))


if __name__ == "__main__":
    #run_classifier()
    #train_paragraph_vector()
    combine_politician_speeches()

