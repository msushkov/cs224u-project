from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cPickle as pickle
import os

from baseline2 import load_doc2vec_model_and_speech_ids

ISSUES = [
    "abortion pregnancy fetus",
    "hiring minorities",
    "gay same-sex homosexual marriage",
    "god religion church",
    "ObamaCare healthcare health insurance",
    "social security welfare",
    "education school funding",
    "pollution air water",
    "crime punishment",
    "gun firearm nra",
    "tax wealthy",
    "citizenship alien immigration immigrant",
    "trade economy tariff",
    "U.N. sovereignty embargo isolationism cooperation",
    "military base downsize 9/11 defense",
    "vote campaign",
    "iran",
    "energy environment green warming",
    "marijuana pot weed drug",
    "stimulus market recovery stock dividend bond budget"
]

ISSUES_1 = [
    "abortion",
    "minorities",
    "marriage",
    "religion",
    "healthcare",
    "welfare",
    "schools",
    "pollution",
    "crime",
    "gun",
    "taxes",
    "citizenship",
    "trade",
    "U.N.",
    "military",
    "voting",
    "iran",
    "energy",
    "marijuana",
    "stimulus"
]

stemmer = PorterStemmer()
ISSUES_ = []
for issue_i in range(0, 20):    
    ISSUES_.append([stemmer.stem(w) for w in word_tokenize(ISSUES[issue_i])])

# cache of tokenized speeches
# speech_id -> list of tokens
tokenized_speeches = {}

if os.path.isfile('tokenized_speeches.pickle'):
    pkl_file = open('tokenized_speeches.pickle', 'rb')
    tokenized_speeches = pickle.load(pkl_file)
    pkl_file.close()


def get_issue_str(i):
    return ISSUES[i]

def stem_tokens(s_tks):
    return [stemmer.stem(w) for w in s_tks]

def jaccard_sim(speech_text, speech_id, issue_i, threshold=0.0):
    global tokenized_speeches

    tokenized_speech = None
    if speech_id in tokenized_speeches:
        tokenized_speech = tokenized_speeches[speech_id]
    else:
        tokenized_speech = set([stemmer.stem(w) for w in word_tokenize(speech_text)])
        tokenized_speeches[speech_id] = tokenized_speech

    common_w = tokenized_speech & set(ISSUES_[issue_i])
    sim_value = float(len(common_w)) / len(ISSUES_[issue_i])
    
    return sim_value > threshold

def cosine_sim(tfidf_vec, i, vect):
    issue_tfidf = vect.transform(ISSUES[i])
    return cosine_similarity(tfidf_vec, issue_tfidf)[0][0]

(doc2vec_model, speech_ids) = load_doc2vec_model_and_speech_ids()

def get_speech_vector(speech_id):
    return doc2vec_model[speech_id]

def doc2vec_sim(speech_id, i):
    issue_word = ISSUES[i]
    # TODO




