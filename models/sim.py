from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from baseline2 import *

ISSUES = [
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

def get_issue_str(i):
    return ISSUES[i]

def stem_tokens(s_tks):
    return [stemmer.stem(w) for w in s_tks]

def jaccard_sim(speech_tk, issue_i):
    common_w = list(set(speech_tk) & set(ISSUES_[issue_i]))
    return float(len(common_w)) / len(ISSUES_[issue_i])

def cosine_sim(tfidf_vec, i, vect):
    issue_tfidf = vect.transform(ISSUES[i])
    return cosine_similarity(tfidf_vec, issue_tfidf)[0][0]

(doc2vec_model, speech_ids) = load_doc2vec_model_and_speech_ids()

def get_speech_vector(speech_id):
    return model[speech_id]

def doc2vec_sim(speech_id, i):
    issue_word = ISSUES[i]
    # TODO




