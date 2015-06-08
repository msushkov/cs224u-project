from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ISSUES = [
    "abortion",
    "hiring women minorities",
    "same-sex marriage",
    "god public sphere",
    "obamacare",
    "social security",
    "vouchers school choice",
    "clean air water",
    "crime punishment",
    "gun ownership",
    "taxes",
    "citizenship illegal aliens",
    "free trade",
    "un",
    "military",
    "voting rights",
    "iran",
    "green energy",
    "marijuana",
    "stimulus market"
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


