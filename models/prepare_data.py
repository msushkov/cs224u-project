import random
import pickle
import pdb
import json
import numpy as np

# mapping from what we see in the vector data to what we actually record
vector_mapping = {
	-5 : -2,
	-3 : -1,
	0 : 0,
	2 : 1,
	5 : 2
}


# Load the labels for each politician
def get_labels(filename):
	print 'Loading labels...'

	labels = {}
	f = open(filename, 'r')
	data = json.load(f)

	num_not_D_or_R = 0
	num_without_vectors = 0
	num_without_party = 0

	for elem in data:
		last_name = elem['name']['last'].strip()

		# get rid of the nickname (e.g. ' Walter "Wally"' -> 'Walter')
		first_name = elem['name']['first'].strip().split()[0].strip()
		
		state = None
		try:
			state = elem['state'].strip()
		except AttributeError:
			pass
		
		vector_data = []
		try:
			vector_data = np.array(elem['vector'])
		except KeyError:
			num_without_vectors += 1
			continue

		# make sure the vector is in [-5, 5]
		if max(vector_data) > 5:
			vector_data = vector_data - 5.0

		vector = []
		for x in vector_data:
			vector.append(vector_mapping[x])
		vector = np.array(vector)

		# vector should now be in [-2, 2]
		assert min(vector) >= -2.0
		assert max(vector) <= 2.0

		party = None
		try:
			party = elem['party'].strip()
		except AttributeError:
			num_without_party += 1
			continue

		party_label = -1
		if party == 'D':
			party_label = 0
		elif party == 'R':
			party_label = 1
		else:
			num_not_D_or_R += 1
			continue

		full_name = ' '.join([first_name, last_name])
		labels[full_name] = (party_label, vector)

	print "There are %d politicians that are not D or R." % num_not_D_or_R
	print "There are %d politicians without vectors." % num_without_vectors
	print "There are %d politicians without a party." % num_without_party
	return labels

# Returns the object that was pickled.
def load_corpus(corpus_filename):
	print 'Loading pickled corpus...'
	pkl_file = open(corpus_filename, 'rb')
	data = pickle.load(pkl_file)
	pkl_file.close()
	return data

# Basically do a join between the data and the labels (join key is the politician name).
def make_data(data, labels):
	print 'Consolidating...'

	X = []
	parties = []
	vectors = []

	num_datapoints = 0
	num_missing = 0

	for name in data:
		if name not in labels:
			num_missing += 1
			continue

		(party_label, vector) = labels[name]
		speeches = data[name]['speech']
		pos = data[name]['pos']
		single_pos = ' '.join(pos)
		single_speech = ' '.join(speeches)
		
		X.append(single_speech)
		parties.append(party_label)
		vectors.append(vector)
		num_datapoints += 1

	print 'Total datapoints: %d' % num_datapoints
	print 'Missing datapoints: %d' % num_missing

	return (X, parties, vectors)


def train_test_split(X, parties, vectors, split=0.15, random_state=123):
	print 'Shuffling...'

	zipped = zip(X, parties, vectors)

	random.seed(random_state)
	random.shuffle(zipped)

	combined = [list(t) for t in zip(*zipped)]
	X = combined[0]
	parties = combined[1]
	vectors = combined[2]

	num_train = int(len(X) * (1.0 - split))
	num_dev = int((len(X) - num_train) / 2.0)
	X_train = X[:num_train]
	X_dev = X[num_train:(num_train + num_dev)]
	X_test = X[(num_train + num_dev):]
	parties_train = parties[:num_train]
	parties_dev = parties[num_train:(num_train + num_dev)]
	parties_test = parties[(num_train + num_dev):]
	vectors_train = vectors[:num_train]
	vectors_dev = vectors[num_train:(num_train + num_dev)]
	vectors_test = vectors[(num_train + num_dev):]

	# (X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test)
	result = (X_train, X_dev, X_test, parties_train, parties_dev, parties_test, \
		np.array(vectors_train), np.array(vectors_dev), np.array(vectors_test))
	return result


# Test this code.
def test():
	import pdb

	corpus_filename = '../data_processing/data_100.pickle'
	labels_filename = '../scraping/all_people'
	labels_filename2 = '../scraping/people_with_vectors_108'

	data = load_corpus(corpus_filename)
	labels = get_labels(labels_filename)
	(X, parties, vectors) = make_data(data, labels)
	(X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test) = train_test_split(X, parties, vectors)

	pdb.set_trace()

# if __name__ == '__main__':
# 	test()

