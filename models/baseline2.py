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
from gensim import corpora, models

from prepare_data import *
from baseline import *
from sim import *


VECTORS_FILE = '../data_processing/data_split_by_speech_nonzero_vectors_only.pickle'
VECTORS_FILE_SOME_MISSING = '../data_processing/data_split_by_speech_some_missing.pickle'
labels_filename2 = '../scraping/fixed_people_with_vectors_234'
labels_filename3 = '../scraping/fixed_people_with_vectors_745'
corpus_filename = '../data_processing/data_all.pickle'


# from nltk
ENGLISH_STOPWORD_SET = set(['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', \
    'before', 'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', \
    'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', \
    'where', 'few', 'because', 'doing', 'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', \
    'does', 'above', 'between', 't', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', \
    'against', 's', 'or', 'own', 'into', 'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', \
    'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', \
    'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can', 'theirs', 'my', 'and', 'then', 'is', \
    'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', \
    'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'yours', 'so', 'the', 'having', 'once'])


# Don't split up by speech
# Only take the 234 datapoints
def run_classifier_dont_split_by_speech():
    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename2, True, False)

    (X, parties, vectors, names) = make_data(data, labels)
    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test) = train_test_split(X, parties, vectors)

    predict_party((X_train, X_test, parties_train, parties_test, vectors_train, vectors_test))
    predict_20_attr_classification((X_train, X_test, parties_train, parties_test, vectors_train, vectors_test))


# Don't split up by speech
# Take all the datapoints (including missing) but filter by label to include the training points that dont have missing values
def run_classifier_dont_split_by_speech_filter_all():
    print "run_classifier_dont_split_by_speech_filter_all()..."

    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename3, False, False) # don't skip anything

    (X, parties, vectors, names) = make_data(data, labels)
    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test) = train_test_split_4(X, parties, vectors)

    predict_party((X_train['party'], X_test['party'], parties_train, parties_test, vectors_train, vectors_test))
    
    # issues

    vect = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2))
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=10, n_jobs=-1, random_state=42)

    for i in xrange(20):
        print "\n========= Attribute %d =========" % i

        X_train_curr = X_train[i]
        X_test_curr = X_test[i]
        curr_labels_train = vectors_train[i]
        curr_labels_test = vectors_test[i]

        X_train_tfidf = vect.fit_transform(X_train_curr)

        print "%d training points, %d test points" % (len(X_train_curr), len(X_test_curr))
        print "Distribution of train labels:"
        print Counter(curr_labels_train)
        print "Distribution of test labels:"
        print Counter(curr_labels_test)

        text_clf = clf.fit(X_train_tfidf, curr_labels_train)

        # dev
        X_tfidf_test = vect.transform(X_test_curr)
        predicted = text_clf.predict(X_tfidf_test)
        acc = np.mean(predicted == curr_labels_test)   
        print "Accuracy is %f" % acc
        print metrics.confusion_matrix(curr_labels_test, predicted)


# Input is a bunch of dictionaries...
def make_predictions(X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test, labels):
    vect = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2))
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=10, n_jobs=-1, random_state=42)

    # PARTY

    X_train_curr = X_train['party']
    X_test_curr = X_test['party']
    names_test_curr = names_test['party']
    names_train_curr = names_train['party']

    print "Party: num_train = %d, num_test = %d" % (len(X_train_curr), len(X_test_curr))

    X_train_tfidf = vect.fit_transform(X_train_curr)
    text_clf = clf.fit(X_train_tfidf, parties_train)
    X_tfidf_test = vect.transform(X_test_curr)
    predicted_parties = text_clf.predict(X_tfidf_test) # shape is (num_speeches_in_test_set,)

    print "%d training points, %d test points" % (len(X_train_curr), len(X_test_curr))
    print "Distribution of train party labels:"
    print Counter(parties_train)
    print "Distribution of test party labels:"
    print Counter(parties_test)

    print_top20_binary(vect, text_clf)

    print "Test accuracy for party prediction before grouping by name = %s" % str(np.mean(predicted_parties == parties_test))

    predicted_parties_train = text_clf.predict(X_train_tfidf)
    print "Train accuracy for party prediction before grouping by name = %s" % str(np.mean(predicted_parties_train == parties_train))    

    # group by name: name -> { 'pred' : [list of predicted_party], 'actual' : party }
    by_name = {}

    # iterate over the speeches in the test set; the politician names will be repeated
    for i, test_name in enumerate(names_test_curr):
        if test_name not in labels:
            continue

        predicted_party = predicted_parties[i]
        actual_party = labels[test_name][0]

        if test_name not in by_name:
            by_name[test_name] = {}
            by_name[test_name]['pred'] = []
            by_name[test_name]['actual'] = None

        by_name[test_name]['pred'].append(predicted_party)
        by_name[test_name]['actual'] = actual_party

    # consolidate the labels for each name
    party_correct = 0
    for name in by_name:
        pred_lst = by_name[name]['pred']
        actual_party = by_name[name]['actual']
        party_counter = Counter(pred_lst)
        most_frequent_party = party_counter.most_common(1)[0][0]
        if most_frequent_party == actual_party:
            party_correct += 1

    print "Test accuracy for party prediction after grouping by name = %s" % str(float(party_correct) / len(by_name))


    # group by name: name -> { 'pred' : [list of predicted_party], 'actual' : party }
    by_name = {}

    # iterate over the speeches in the test set; the politician names will be repeated
    for i, train_name in enumerate(names_train_curr):
        if train_name not in labels:
            continue

        predicted_party = predicted_parties_train[i]
        actual_party = labels[train_name][0]

        if train_name not in by_name:
            by_name[train_name] = {}
            by_name[train_name]['pred'] = []
            by_name[train_name]['actual'] = None

        by_name[train_name]['pred'].append(predicted_party)
        by_name[train_name]['actual'] = actual_party

    # consolidate the labels for each name
    party_correct = 0
    for name in by_name:
        pred_lst = by_name[name]['pred']
        actual_party = by_name[name]['actual']
        party_counter = Counter(pred_lst)
        most_frequent_party = party_counter.most_common(1)[0][0]
        if most_frequent_party == actual_party:
            party_correct += 1

    print "Train accuracy for party prediction after grouping by name = %s" % str(float(party_correct) / len(by_name))


    # ISSUES

    for i in xrange(20):
        X_train_curr = X_train[i]
        X_test_curr = X_test[i]
        names_test_curr = names_test[i]
        curr_vectors_train = vectors_train[i]

        print "Issue %d: num_train = %d, num_test = %d" % (i, len(X_train_curr), len(X_test_curr))

        X_train_tfidf = vect.fit_transform(X_train_curr)
        X_tfidf_test = vect.transform(X_test_curr)
        text_clf = clf.fit(X_train_tfidf, curr_vectors_train)
        predicted = text_clf.predict(X_tfidf_test) # gives a single label for each of the test points

        print_top20_multiclass(vect, text_clf, [-2, -1, 1, 2])

        print "Distribution of train labels:"
        print Counter(curr_vectors_train)
        print "Distribution of test labels:"
        print Counter(vectors_test[i])

        # group by name: name -> { 'pred' : [list of predicted_issue], 'actual' : issue_labels }
        by_name = {}

        # iterate over the speeches in the test set; the politician names will be repeated
        for j, test_name in enumerate(names_test_curr):
            predicted_issue_label = predicted[j] # for current test point
            actual_issue_label = labels[test_name][1][i]

            if test_name not in by_name:
                by_name[test_name] = {}
                by_name[test_name]['pred'] = []
                by_name[test_name]['actual'] = None

            by_name[test_name]['pred'].append(predicted_issue_label)
            by_name[test_name]['actual'] = actual_issue_label

        # consolidate the labels for each name
        issues_correct = 0
        for name in by_name:
            pred_lst = by_name[name]['pred']
            actual_issue = by_name[name]['actual']
            issue_counter = Counter(pred_lst)
            most_frequent_issue = issue_counter.most_common(1)[0][0]
            if most_frequent_issue == actual_issue:
                issues_correct += 1

        print "Accuracy for issue %d prediction = %s" % (i, str(float(issues_correct) / len(by_name)))



# Input is a bunch of dictionaries...
def make_predictions_using_doc2vec(X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test, labels):
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=10, n_jobs=-1, random_state=42)

    # PARTY

    X_train_curr = X_train['party']
    X_test_curr = X_test['party']
    names_test_curr = names_test['party']

    print "Party: num_train = %d, num_test = %d" % (len(X_train_curr), len(X_test_curr))

    text_clf = clf.fit(X_train_curr, parties_train)
    predicted_parties = text_clf.predict(X_test_curr) # shape is (num_speeches_in_test_set,)

    print "%d training points, %d test points" % (len(X_train_curr), len(X_test_curr))
    print "Distribution of train party labels:"
    print Counter(parties_train)
    print "Distribution of test party labels:"
    print Counter(parties_test)

    # group by name: name -> { 'pred' : [list of predicted_party], 'actual' : party }
    by_name = {}

    # iterate over the speeches in the test set; the politician names will be repeated
    for i, test_name in enumerate(names_test_curr):
        if test_name not in labels: continue

        predicted_party = predicted_parties[i]
        actual_party = labels[test_name][0]

        if test_name not in by_name:
            by_name[test_name] = {}
            by_name[test_name]['pred'] = []
            by_name[test_name]['actual'] = None

        by_name[test_name]['pred'].append(predicted_party)
        by_name[test_name]['actual'] = actual_party

    # consolidate the labels for each name
    party_correct = 0
    for name in by_name:
        pred_lst = by_name[name]['pred']
        actual_party = by_name[name]['actual']
        party_counter = Counter(pred_lst)
        most_frequent_party = party_counter.most_common(1)[0][0]
        if most_frequent_party == actual_party:
            party_correct += 1

    print "Accuracy for party prediction = %s" % str(float(party_correct) / len(by_name))


    # ISSUES

    for i in xrange(20):
        X_train_curr = X_train[i]
        X_test_curr = X_test[i]
        names_test_curr = names_test[i]
        curr_vectors_train = vectors_train[i]

        print "Issue %d: num_train = %d, num_test = %d" % (i, len(X_train_curr), len(X_test_curr))

        text_clf = clf.fit(X_train_curr, curr_vectors_train)
        predicted = text_clf.predict(X_test_curr) # gives a single label for each of the test points

        print "Distribution of train labels:"
        print Counter(curr_vectors_train)
        print "Distribution of test labels:"
        print Counter(vectors_test[i])

        # group by name: name -> { 'pred' : [list of predicted_issue], 'actual' : issue_labels }
        by_name = {}

        # iterate over the speeches in the test set; the politician names will be repeated
        for j, test_name in enumerate(names_test_curr):
            predicted_issue_label = predicted[j] # for current test point
            actual_issue_label = labels[test_name][1][i]

            if test_name not in by_name:
                by_name[test_name] = {}
                by_name[test_name]['pred'] = []
                by_name[test_name]['actual'] = None

            by_name[test_name]['pred'].append(predicted_issue_label)
            by_name[test_name]['actual'] = actual_issue_label

        # consolidate the labels for each name
        issues_correct = 0
        for name in by_name:
            pred_lst = by_name[name]['pred']
            actual_issue = by_name[name]['actual']
            issue_counter = Counter(pred_lst)
            most_frequent_issue = issue_counter.most_common(1)[0][0]
            if most_frequent_issue == actual_issue:
                issues_correct += 1

        print "Accuracy for issue %d prediction = %s" % (i, str(float(issues_correct) / len(by_name)))


# First divide the politicians into train and test, then split those up by speech, then take majority vote after classifying
# Take all the datapoints (including missing) but filter by label to include the training points that dont have missing values
def run_classifier_split_by_speech():
    print "run_classifier_split_by_speech()..."

    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename3, False, False) # dont skip anything

    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test) = \
        make_data_split_by_speech3(data, labels)
    
    make_predictions(X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test, labels)




# Split up by speech
def run_classifier():
    # if VECTORS_FILE is not found, run this
    if not os.path.isfile(VECTORS_FILE):
        data = load_corpus(corpus_filename)
        save_data_split_by_speech(data, labels_filename2, VECTORS_FILE, True, False)

    # list of dicts
    data = load_corpus(VECTORS_FILE)

    (X, parties, vectors, speech_ids, names) = make_data_split_by_speech(data)
    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test) = train_test_split(X, parties, vectors)

    predict_party((X_train, X_test, parties_train, parties_test, vectors_train, vectors_test))
    predict_20_attr_classification((X_train, X_test, parties_train, parties_test, vectors_train, vectors_test))



def run_filter_by_similarity(sim_threshold=0.5):
    print "run_filter_by_similarity()..."

    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename3, False, False) # dont skip anything

    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test) = \
        make_data_split_by_speech3(data, labels, jaccard_sim)

    make_predictions(X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test, labels)


# Combine the labels of all the politician's speeches to get a single prediction for a given politician
# Use doc2vec instead of tfidf as vector for speeches
def combine_politician_speeches_use_doc2vec():
    data = load_corpus(corpus_filename)
    labels = get_labels(labels_filename3, False, False) # dont skip anything

    (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test) = \
        make_data_split_by_speech5(data, labels)
    
    make_predictions_using_doc2vec(X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test, labels)


# http://radimrehurek.com/2014/12/doc2vec-tutorial/
# Train doc2vec on the corpus
def train_paragraph_vector(num_epochs=10):
    print "In train_paragraph_vector()..."

    speech_ids = []

    # if VECTORS_FILE is not found, run this
    if not os.path.isfile(VECTORS_FILE_SOME_MISSING):
        data = load_corpus(corpus_filename)
        save_data_split_by_speech(data, labels_filename2, VECTORS_FILE_SOME_MISSING, False, False) # false, false -> dont ignore anything

    # list of dicts
    data = load_corpus(VECTORS_FILE_SOME_MISSING)
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


def run_lda(num_topics=20):
    # if VECTORS_FILE is not found, run this
    if not os.path.isfile(VECTORS_FILE_SOME_MISSING):
        data = load_corpus(corpus_filename)
        save_data_split_by_speech(data, labels_filename2, VECTORS_FILE_SOME_MISSING, False, False) # false, false -> dont ignore anything

    # list of dicts
    data = load_corpus(VECTORS_FILE_SOME_MISSING)
    
    tokenized_speeches = []
    speech_ids = []

    print "Loaded data. Processing..."

    for curr_point in data:
        speech_id = curr_point['speech_id']
        speech_text = curr_point['speech_text']
        speech_ids.append(speech_id)
        speech_tokens = [w for w in word_tokenize(speech_text) if w.lower() not in ENGLISH_STOPWORD_SET]
        tokenized_speeches.append(speech_tokens)

    print "Starting LDA..."

    dictionary = corpora.Dictionary(tokenized_speeches)
    corpus = [dictionary.doc2bow(text) for text in tokenized_speeches]
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=1000, passes=1)

    lda.print_topics(num_topics=num_topics, num_words=20)


if __name__ == "__main__":
    #run_classifier_dont_split_by_speech()
    #run_classifier_dont_split_by_speech_filter_all()
    run_classifier_split_by_speech()
    #run_classifier()
    #train_paragraph_vector()
    #combine_politician_speeches()
    #combine_politician_speeches_experiment1()
    #run_filter_by_similarity(0.0)
    #run_lda()
    #combine_politician_speeches_use_doc2vec()


