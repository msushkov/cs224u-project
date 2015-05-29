from prepare_data import *
from collections import Counter
import pdb
import numpy as np


corpus_filename = '../data_processing/data_all.pickle'
labels_filename = '../scraping/people_with_vectors_746'


data = load_corpus(corpus_filename)
labels = get_labels(labels_filename)
(X, parties, vectors) = make_data(data, labels)

c = Counter()

for vec in vectors:
	for x in vec:
		c[x] += 1
print c

c2 = Counter()
for p in parties:
	c2[p] += 1
print c2

# distribution of word count for each politician
wc = []
for speech in X:
	words = speech.split()
	wc.append(len(words))
print wc[:25]
print "mean: %f" % np.mean(wc)
print "std: %f" % np.std(wc)
print "median: %d" % np.median(wc)
