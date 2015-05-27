import ast
import random
import pickle
import pdb

# Load the labels for each politician
def get_labels(filename):
	print 'Loading labels...'

	labels = {}
	f = open(filename, 'r')
	data = f.read().strip()
	labels_list = ast.literal_eval(data)
	for elem in labels_list:
		last_name = elem['name']['last'].strip()
		first_name = elem['name']['first'].strip()
		state = elem['state'].strip()
		vector = []
		try:
			vector = elem['vector'].strip()
		except KeyError:
			pass
		party = elem['party'].strip()
		full_name = ' '.join([first_name, last_name])
		labels[full_name] = (party, ast.literal_eval(vector))
	return labels

# Load the corpus without the labels (corpus_filename is the pickled file)
def get_corpus(corpus_filename, labels_filename):
	print 'Loading pickled corpus...'

	pkl_file = open(corpus_filename, 'rb')
	data = pickle.load(pkl_file)
	pkl_file.close()
	labels = get_labels(labels_filename)

	X = []
	parties = []
	vectors = []

	print 'Consolidating...'

	for name in labels:
		(state, vector, party) = labels[name]
		party_label = -1
		if party == 'D':
			party_label = 0
		elif party == 'R':
			party_label = 1
		else:
			party_label = 2
		speeches = data[name]
		single_text = ' '.join(speeches)
		X.append(single_text)
		parties.append(party_label)
		vectors.append(vector)

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

	corpus_filename = '../data_processing/data_10.pickle'
	labels_filename = '../scraping/all_people'
	labels_filename2 = '../scraping/people_with_vectors_108'

	(X, parties, vectors) = get_corpus(corpus_filename, labels_filename)
	(X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test) = train_test_split(X, parties, vectors)

	pdb.set_trace()

if __name__ == '__main__':
	test()

