import random
import pickle
import pdb
import json

# Load the labels for each politician
def get_labels(filename):
	print 'Loading labels...'

	labels = {}
	f = open(filename, 'r')
	data = json.load(f)
	for elem in data:
		last_name = elem['name']['last'].strip()
		first_name = elem['name']['first'].strip()
		state = None
		try:
			state = elem['state'].strip()
		except AttributeError:
			pass
		vector = []
		try:
			vector = elem['vector'].strip()
		except KeyError:
			pass
		party = None
		try:
			party = elem['party'].strip()
		except AttributeError:
			pass
		full_name = ' '.join([first_name, last_name])
		labels[full_name] = (party, vector)
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

		vector = None
		(party, vector) = labels[name]
		party_label = -1
		if party == 'D':
			party_label = 0
		elif party == 'R':
			party_label = 1
		else:
			party_label = 2

		num_datapoints += 1

		speeches = data[name]['speech']
		pos = data[name]['pos']
		single_pos = ' '.join(pos)
		single_speech = ' '.join(speeches)
		
		X.append(single_speech)
		parties.append(party_label)
		vectors.append(vector)

	print 'Total datapoints: %d' % num_datapoints
	print 'Missing datapoints: %d' % num_missing

	return (X, parties, vectors)


def train_test_split(X, parties, vectors, split=0.15, random_state=123):
	print 'Shuffling...'
	random.seed(random_state)
	random.shuffle(X)
	random.seed(random_state)
	random.shuffle(parties)
	random.seed(random_state)
	random.shuffle(vectors)

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

	return (X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test)

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

