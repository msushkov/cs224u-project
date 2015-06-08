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
from sklearn.externals import joblib

from gensim.models.doc2vec import Doc2Vec, LabeledSentence

from prepare_data import *
from baseline import *


VECTORS_FILE = '../data_processing/data_split_by_speech_nonzero_vectors_only.pickle'
VECTORS_FILE_SOME_MISSING = '../data_processing/data_split_by_speech_some_missing.pickle'
labels_filename2 = '../scraping/fixed_people_with_vectors_234'
labels_filename3 = '../scraping/fixed_people_with_vectors_745'
corpus_filename = '../data_processing/data_all.pickle'


def run_classifier():
    # if VECTORS_FILE is not found, run this
    if not os.path.isfile(VECTORS_FILE):
        data = load_corpus(corpus_filename)
        save_data_split_by_speech(data, labels_filename2, VECTORS_FILE)

    # list of dicts
    data = load_corpus(VECTORS_FILE)

    (X, parties, vectors, speech_ids, names) = make_data_split_by_speech(data)
    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test) = train_test_split(X, parties, vectors)

    predict_party((X_train, X_test, parties_train, parties_test, vectors_train, vectors_test))
    predict_20_attr_classification((X_train, X_test, parties_train, parties_test, vectors_train, vectors_test))


# Combine the labels of all the politician's speeches to get a single prediction for a given politician
def combine_politician_speeches():
    # if VECTORS_FILE is not found, run this
    if not os.path.isfile(VECTORS_FILE):
        data = load_corpus(corpus_filename)
        save_data_split_by_speech(data, labels_filename2, VECTORS_FILE)

    # get the predictions for the test speeches
    # list of dicts
    data = load_corpus(VECTORS_FILE)

    labels = get_labels(labels_filename2)

    (X, parties, vectors, speech_ids, names) = make_data_split_by_speech(data)
    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, speech_ids_train, speech_ids_test, names_train, names_test) = \
        train_test_split_2(X, parties, vectors, speech_ids, names)

    # predict party
    vect = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2))
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=10, n_jobs=-1, random_state=42)

    X_train_tfidf = vect.fit_transform(X_train)
    text_clf = clf.fit(X_train_tfidf, parties_train)
    X_tfidf_test = vect.transform(X_test)
    predicted_parties = text_clf.predict(X_tfidf_test) # shape is (num_speeches_in_test_set,)

    baseline.print_top20_binary(vect, text_clf)

    # save classifier and vectorizer
    joblib.dump(text_clf, '../saved_svm_models/party.pkl')
    joblib.dump(vect, '../saved_svm_models/vect.pkl')

    # predict issues (shape will be (num_issues, num_speeches_in_test_set))
    predicted_issues = []

    for i in xrange(20):
        print "Issue %d" % i

        text_clf = clf.fit(X_train_tfidf, vectors_train[:, i])
        predicted = text_clf.predict(X_tfidf_test)
        predicted_issues.append(predicted)

        baseline.print_top20_multiclass(vect, text_clf, [-2, -1, 1, 2])

        joblib.dump(text_clf, '../saved_svm_models/issue_%d.pkl' % i)

    issues_pred = np.array(predicted_issues).T # now shape is (num_speeches_in_test_set, num_issues)

    # group by name: name -> { 'pred' : [list of (predicted, actual)], 'actual' : (party, labels) }
    by_name = {}

    # iterate over the speeches in the test set; the politician names will be repeated
    for i, test_name in enumerate(names_test):
        predicted_party = predicted_parties[i]
        predicted_issue_labels = issues_pred[i, :]

        actual_party = labels[test_name][0]
        actual_issue_labels = labels[test_name][1]

        if test_name not in by_name:
            by_name[test_name] = {}
            by_name[test_name]['pred'] = []
            by_name[test_name]['actual'] = None

        curr = (predicted_party, predicted_issue_labels)
        by_name[test_name]['pred'].append(curr)
        by_name[test_name]['actual'] = (actual_party, actual_issue_labels)

    # consolidate the labels for each name
    party_correct = 0
    issues_correct = Counter() # will have 20 entries

    for name in by_name:
        pred_lst = by_name[name]['pred']
        (actual_party, actual_issue_labels) = by_name[name]['actual']

        # predicted counters
        party_counter = Counter() # count of party labels ()
        issue_counter = {} # 20 entries
        for i in range(20):
            issue_counter[i] = Counter()

        for (predicted_party, predicted_issue_labels) in pred_lst:
            party_counter[predicted_party] += 1
            for i in range(20):
                curr_issue_prediction = predicted_issue_labels[i]
                issue_counter[i][curr_issue_prediction] += 1

        # tally up the correct combined guesses

        most_frequent_party = party_counter.most_common(1)[0][0]
        if most_frequent_party == actual_party:
            party_correct += 1

        for i in range(20):
            curr_freq = issue_counter[i].most_common(1)[0][0]
            if curr_freq == actual_issue_labels[i]:
                issues_correct[i] += 1

    print "Accuracy for party prediction = %s" % str(float(party_correct) / len(by_name))

    for i in range(20):
        print "Accuracy for issue %d prediction = %s" % (i, str(float(issues_correct[i]) / len(by_name)))


# Use as a test set a set of politicians who aren't in the original 234
# test_split: what fraction of this unseen data do we want to test on?
def combine_politician_speeches_experiment1(test_split=1.0):
    # if VECTORS_FILE_SOME_MISSING is not found, run this
    if not os.path.isfile(VECTORS_FILE_SOME_MISSING):
        data = load_corpus(corpus_filename)
        save_data_split_by_speech(data, labels_filename3, VECTORS_FILE_SOME_MISSING)

    # load the vectorizer
    vect = joblib.load('../saved_svm_models/vect.pkl')

    data = load_corpus(VECTORS_FILE_SOME_MISSING)

    (X, parties, vectors, speech_ids, names) = make_data_split_by_speech(data)
    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, speech_ids_train, speech_ids_test, names_train, names_test) = \
        train_test_split_2(X, parties, vectors, speech_ids, names, split=test_split) 

    labels = get_labels(labels_filename3)

    X_tfidf_test = vect.transform(X_test)
    predicted_parties = text_clf.predict(X_tfidf_test) # shape is (num_speeches_in_test_set,)

    # predict issues (shape will be (num_issues, num_speeches_in_test_set))
    predicted_issues = []

    for i in xrange(20):
        text_clf = joblib.load('../saved_svm_models/issue_%d.pkl' % i)
        predicted = text_clf.predict(X_tfidf_test)
        predicted_issues.append(predicted)

    issues_pred = np.array(predicted_issues).T # now shape is (num_speeches_in_test_set, num_issues)

    # group by name: name -> { 'pred' : [list of (predicted, actual)], 'actual' : (party, labels) }
    by_name = {}

    # iterate over the speeches in the test set; the politician names will be repeated
    for i, test_name in enumerate(names_test):
        predicted_party = predicted_parties[i]
        predicted_issue_labels = issues_pred[i, :]

        actual_party = labels[test_name][0]
        actual_issue_labels = labels[test_name][1]

        if test_name not in by_name:
            by_name[test_name] = {}
            by_name[test_name]['pred'] = []
            by_name[test_name]['actual'] = None

        curr = (predicted_party, predicted_issue_labels)
        by_name[test_name]['pred'].append(curr)
        by_name[test_name]['actual'] = (actual_party, actual_issue_labels)

    # consolidate the labels for each name
    party_correct = 0
    issues_correct = Counter() # will have 20 entries
    issues_total = Counter() # will have 20 entries

    for name in by_name:
        pred_lst = by_name[name]['pred']
        (actual_party, actual_issue_labels) = by_name[name]['actual']

        # predicted counters
        party_counter = Counter() # count of party labels ()
        issue_counter = {} # 20 entries
        for i in range(20):
            issue_counter[i] = Counter()

        for (predicted_party, predicted_issue_labels) in pred_lst:
            party_counter[predicted_party] += 1
            for i in range(20):
                curr_issue_prediction = predicted_issue_labels[i]
                issue_counter[i][curr_issue_prediction] += 1

        # tally up the correct combined guesses

        most_frequent_party = party_counter.most_common(1)[0][0]
        if most_frequent_party == actual_party:
            party_correct += 1

        for i in range(20):
            curr_freq = issue_counter[i].most_common(1)[0][0]

            # ignore datapoints where the truth label is 0 (missing)
            if actual_issue_labels[i] != 0:
                issues_total[i] += 1
                if curr_freq == actual_issue_labels[i]:
                    issues_correct[i] += 1

    print "Accuracy for party prediction = %s" % str(float(party_correct) / len(by_name))

    for i in range(20):
        print "Accuracy for issue %d prediction = %s" % (i, str(float(issues_correct[i]) / issues_total[i]))



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
    #combine_politician_speeches_experiment1()

