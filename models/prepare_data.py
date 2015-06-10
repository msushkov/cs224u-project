import random
import cPickle as pickle
import pdb
import json
import numpy as np
from collections import Counter
import sys
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sim import *

MIN_SPEECH_LENGTH = 100

# mapping from what we see in the vector data to what we actually record
vector_mapping = {
	-5 : -2,
	-3 : -1,
	0 : 0,
	2 : 1,
	5 : 2
}

def my_sign(v):
	for i in xrange(len(v)):
		if v[i] < 0:
			v[i] = 0.0
		else:
			v[i] = 1.0
	return v


# Load the labels for each politician
def get_labels(filename, ignore_0_vec=False, ignore_no_missing=True):
	print 'Loading labels...'

	labels = {}
	f = open(filename, 'r')
	data = json.load(f)

	num_not_D_or_R = 0
	num_without_vectors = 0
	num_without_party = 0
	count = 0

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

		# do we ignore datapoints that have 0 in their vectors? 0 means missing data
		if ignore_0_vec:
			if 0 in set(vector_data):
				continue

		# do we ignore datapoints that have no 0's in their vectors?
		# in other words, we only want datapoints with missing values
		if ignore_no_missing:
			if 0 not in set(vector_data):
				continue

		vector = []
		for x in vector_data:
			vector.append(vector_mapping[x])
		vector = np.array(vector)

		# vector should now be in [-2, 2]

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
		count += 1

	print "Loaded %d labels." % count
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

# politician_name -> list of speeches
# Basically do a join between the data and the labels (join key is the politician name).
def make_data(data, labels, join_speeches=True):
	print 'Consolidating...'

	names = []
	X = []
	parties = []
	vectors = []

	num_datapoints = 0
	num_missing = 0

	politician_to_num_speeches = Counter()
	label_distribution = Counter()
	party_distribution = Counter()

	for name in data:
		if name not in labels:
			num_missing += 1
			continue

		(party_label, vector) = labels[name]
		speeches = data[name]['speech']
		pos = data[name]['pos']

		if join_speeches:
			single_pos = ' '.join(pos)
			single_speech = ' '.join(speeches)
			
			names.append(name)
			X.append(single_speech)
			parties.append(party_label)
			vectors.append(vector)
			num_datapoints += 1
		else:
			for speech in speeches:
				# skip short speeches
				if len(speech.split()) >= MIN_SPEECH_LENGTH:
					names.append(name)
					X.append(speech)
					parties.append(party_label)
					vectors.append(vector)
					num_datapoints += 1
					politician_to_num_speeches[name] += 1

					party_distribution[party_label] += 1
					for val in vector:
						label_distribution[val] += 1


	print 'Total datapoints: %d' % num_datapoints
	print 'Missing datapoints: %d' % num_missing
	#print politician_to_num_speeches
	print label_distribution
	print party_distribution

	return (X, parties, vectors, names)


# Given a saved list of dict objects (in binary format), load that and convert that to 
# a matrix format that is understood by scikit-learn.
# data is loaded from something like '../data_processing/data_split_by_speech_all.pickle' (list of dict objects)
def make_data_split_by_speech(data, labels=None):
	X = []
	parties = []
	vectors = []
	speech_ids = []
	names = []

	for curr_point in data:
		speech_id = curr_point['speech_id']
		name = curr_point['name']
		vector = curr_point['vector']
		party_label = curr_point['party_label']
		speech_text = curr_point['speech_text']

		# only return speeches of politicians for which we have provided labels
		if labels is not None and name not in labels:
			continue

		# skip short speeches
		if len(speech_text.split()) >= MIN_SPEECH_LENGTH:
			X.append(speech_text)
			vectors.append(vector)
			parties.append(party_label)
			speech_ids.append(speech_id)
			names.append(name)

	return (X, parties, vectors, speech_ids, names)


def make_data_split_by_speech2(data, labels, split=0.7, random_state=123):
	names = data.keys()
	random.seed(random_state)
	random.shuffle(names)
	num_train = int(len(names) * (1.0 - split))
	names_train = set(names[:num_train])
	names_test = set(names[num_train:])
	
	X_train = []
	X_test = []
	parties_train = []
	parties_test = []
	vectors_train = []
	vectors_test = []

	num_datapoints = 0
	num_missing = 0

	for name in data:
		if name not in labels:
			num_missing += 1
			continue

		(party_label, vector) = labels[name]
		speeches = data[name]['speech']
		pos = data[name]['pos']

		for speech in speeches:
			# skip short speeches
			if len(speech.split()) >= MIN_SPEECH_LENGTH:
				if name in names_train:
					X_train.append(speech)
					parties_train.append(party_label)
					vectors_train.append(vector)
					num_datapoints += 1
				elif name in names_test:
					X_test.append(speech)
					parties_test.append(party_label)
					vectors_test.append(vector)
					num_datapoints += 1

	print 'Total datapoints: %d' % num_datapoints
	print 'Missing datapoints: %d' % num_missing

	return (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test)


# Return a training and test set for each attribute
def make_data_split_by_speech3(data, labels, split=0.7, random_state=123):
	X_train = {}
	X_test = {}
	vectors_train = {} # issue id -> 1D list of labels for each of the train points
	vectors_test = {}
	names_train = {}
	names_test = {}

	names = data.keys()
	random.seed(random_state)
	random.shuffle(names)
	num_train = int(len(names) * (1.0 - split))
	names_train['party'] = names[:num_train]
	names_test['party'] = names[num_train:]
	
	parties_train = []
	parties_test = []

	X_train['party'] = []
	X_test['party'] = []
	
	X_i = {}
	names_i = {}
	vectors_i = {}
	for i in range(20):
		X_i[i] = []
		names_i[i] = []
		vectors_i[i] = []

	for name in data:
		if name not in labels:
			continue

		(party_label, vector) = labels[name]
		speeches = data[name]['speech']
		pos = data[name]['pos']

		for speech in speeches:
			# skip short speeches
			if len(speech.split()) >= MIN_SPEECH_LENGTH:
				# party
				if name in set(names_train['party']):
					X_train['party'].append(speech)
					parties_train.append(party_label)
				elif name in set(names_test['party']):
					X_test['party'].append(speech)
					parties_test.append(party_label)

				# issues
				for i in range(20):
					if vector[i] != 0:
						X_i[i].append(speech)
						names_i[i].append(name)
						vectors_i[i].append(vector[i])

	for i in range(20):
		curr_X = X_i[i]
		num_train = int(len(curr_X) * (1.0 - split))
		X_train[i] = curr_X[:num_train]
		X_test[i] = curr_X[num_train:]
		curr_vector = vectors_i[i]
		vectors_train[i] = curr_vector[:num_train]
		vectors_test[i] = curr_vector[num_train:]
		curr_names = names_i[i]
		names_train[i] = curr_names[:num_train]
		names_test[i] = curr_names[num_train:]

	return (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test)


# Return a training and test set for each attribute
def make_data_split_by_speech4(data, labels, split=0.7, random_state=123):
	X_train = {}
	X_test = {}
	vectors_train = {} # issue id -> 1D list of labels for each of the train points
	vectors_test = {}
	names_train = {}
	names_test = {}
	speech_ids_train = {}
	speech_ids_test = {}

	names = data.keys()
	random.seed(random_state)
	random.shuffle(names)
	num_train = int(len(names) * (1.0 - split))
	names_train['party'] = names[:num_train]
	names_test['party'] = names[num_train:]
	
	parties_train = []
	parties_test = []

	X_train['party'] = []
	X_test['party'] = []
	
	X_i = {}
	names_i = {}
	vectors_i = {}
	for i in range(20):
		X_i[i] = []
		names_i[i] = []
		vectors_i[i] = []

	for name in data:
		if name not in labels:
			continue

		(party_label, vector) = labels[name]
		speeches = data[name]['speech']
		pos = data[name]['pos']

		for speech in speeches:
			# skip short speeches
			if len(speech.split()) >= MIN_SPEECH_LENGTH:
				# party
				if name in set(names_train['party']):
					X_train['party'].append(speech)
					parties_train.append(party_label)
				elif name in set(names_test['party']):
					X_test['party'].append(speech)
					parties_test.append(party_label)

				# issues
				for i in range(20):
					if vector[i] != 0:
						X_i[i].append(speech)
						names_i[i].append(name)
						vectors_i[i].append(vector[i])

	for i in range(20):
		curr_X = X_i[i]
		num_train = int(len(curr_X) * (1.0 - split))
		X_train[i] = curr_X[:num_train]
		X_test[i] = curr_X[num_train:]
		curr_vector = vectors_i[i]
		vectors_train[i] = curr_vector[:num_train]
		vectors_test[i] = curr_vector[num_train:]
		curr_names = names_i[i]
		names_train[i] = curr_names[:num_train]
		names_test[i] = curr_names[num_train:]

	return (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test, names_train, names_test)


# Call this with the labels filename
# corpus is loaded from data_*.pickle
def save_data_split_by_speech(corpus, labels_filename, output_filename='../data_processing/data_split_by_speech_nonzero_vectors_only.pickle', ignore_0_vec=True, ignore_no_missing=False):
	# dict of label tuples by name
	labels = get_labels(labels_filename, ignore_0_vec, ignore_no_missing)

	# list of dicts
	data = []

	count = 0
	num_names = 0
	for name in corpus:
		if name not in labels:
			continue

		num_names += 1

		(party_label, vector) = labels[name]
		speeches = corpus[name]['speech']

		for speech_text in speeches:
			# skip short speeches
			if len(speech_text.split()) >= MIN_SPEECH_LENGTH:
				curr_point = {}
				curr_point['speech_id'] = 'SPEECH_%d' % count
				curr_point['name'] = name
				curr_point['vector'] = vector # numpy array
				curr_point['party_label'] = party_label # 0 (D) or 1 (R)
				curr_point['speech_text'] = speech_text
				count += 1
				
				data.append(curr_point)

	print "%d names out of %d" % (num_names, len(corpus))
	print "%d datapoints" % len(data)
	print "saving binary file..."

	# save data as a binary file
	output = open(output_filename, 'wb')
	pickle.dump(data, output)
	output.close()


def train_test_split(X, parties, vectors, split=0.30, random_state=123):
	print "Number of total datapoints: %d" % len(X)

	zipped = zip(X, parties, vectors)

	random.seed(random_state)
	random.shuffle(zipped)

	combined = [list(t) for t in zip(*zipped)]
	X = combined[0]
	parties = combined[1]
	vectors = combined[2]

	num_train = int(len(X) * (1.0 - split))
	#num_dev = len(X) - num_train
	X_train = X[:num_train]
	#X_test = X[num_train:(num_train + num_dev)]
	X_test = X[num_train:]
	#X_test = X[(num_train + num_dev):]
	parties_train = parties[:num_train]
	parties_test = parties[num_train:]
	#parties_dev = parties[num_train:(num_train + num_dev)]
	#parties_test = parties[(num_train + num_dev):]
	vectors_train = vectors[:num_train]
	vectors_test = vectors[num_train:]
	#vectors_dev = vectors[num_train:(num_train + num_dev)]
	#vectors_test = vectors[(num_train + num_dev):]

	# (X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test)
	#result = (X_train, X_dev, X_test, parties_train, parties_dev, parties_test, \
	#	np.array(vectors_train), np.array(vectors_dev), np.array(vectors_test))
	
	result = (X_train, X_test, parties_train, parties_test, np.array(vectors_train), np.array(vectors_test))
	
	return result

def train_test_split_2(X, parties, vectors, speech_ids, names, split=0.30, random_state=123):
	print "Number of total datapoints: %d" % len(X)

	zipped = zip(X, parties, vectors, speech_ids, names)

	random.seed(random_state)
	random.shuffle(zipped)

	combined = [list(t) for t in zip(*zipped)]
	X = combined[0]
	parties = combined[1]
	vectors = combined[2]
	speech_ids = combined[3]
	names = combined[4]

	num_train = int(len(X) * (1.0 - split))
	X_train = X[:num_train]
	X_test = X[num_train:]
	parties_train = parties[:num_train]
	parties_test = parties[num_train:]
	vectors_train = vectors[:num_train]
	vectors_test = vectors[num_train:]
	speech_ids_train = speech_ids[:num_train]
	speech_ids_test = speech_ids[num_train:]
	names_train = names[:num_train]
	names_test = names[num_train:]
	
	result = (X_train, X_test, parties_train, parties_test, np.array(vectors_train), np.array(vectors_test), speech_ids_train, speech_ids_test, names_train, names_test)
	
	return result


# Only use speeches that are more similar than sim_threshold to the topic i
def train_test_split_3(X, parties, vectors, speech_ids, names, sim_threshold=0.5, similarity_measure='jaccard', split=0.30, random_state=123):
	print "Number of total datapoints: %d" % len(X)

	zipped = zip(X, parties, vectors, speech_ids, names)

	random.seed(random_state)
	random.shuffle(zipped)

	combined = [list(t) for t in zip(*zipped)]
	X = combined[0]
	parties = combined[1]
	vectors = combined[2]
	speech_ids = combined[3]
	names = combined[4]

	num_train = int(len(X) * (1.0 - split))

	result = {}

	X_train = X[:num_train]
	X_test = X[num_train:]
	parties_train = parties[:num_train]
	parties_test = parties[num_train:]
	speech_ids_train = speech_ids[:num_train]
	speech_ids_test = speech_ids[num_train:]
	names_train = names[:num_train]
	names_test = names[num_train:]

	result['party'] = (X_train, X_test, parties_train, parties_test, speech_ids_train, speech_ids_test, names_train, names_test)

	X_tfidf = None
	vect = None
	if similarity_measure == 'cosine':
		vect = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2))
		X_tfidf = vect.fit_transform(X)

	# if 'cosine': speech_id -> (tfidf vector, index into X)
	# if 'jaccard': speech_id -> (speech_tokens, index into X)
	# if 'dic2vec': speech_id -> (index into X, index into X)
	processed_speeches = {}
	print "Stemming and tokenizing speeches..."
	for j in range(len(X)):
		if j % 10000 == 0: print j
		curr_speech = X[j]
		curr_speech_id = speech_ids[j]
		
		if similarity_measure == 'cosine':
			processed_speeches[curr_speech_id] = (X_tfidf[j], j)
		elif similarity_measure == 'jaccard':
			speech_tkns = word_tokenize(curr_speech)
			speech_tkns_stemmed = stem_tokens(speech_tkns)
			processed_speeches[curr_speech_id] = (speech_tkns_stemmed, j)
		elif similarity_measure == 'doc2vec':
			processed_speech[curr_speech_id] = (j, j)
		

	for i in range(20):
		print 'Issue %d...' % i

		# find the speech ids that are within sim_threshold of topic i

		X_i = []
		vectors_i = []
		speech_ids_i = []
		names_i = []

		for speech_id in speech_ids:
			(processed_speech, index_into_X) = processed_speeches[speech_id]

			curr_sim = -1.0
			if similarity_measure == 'cosine':
				curr_sim = cosine_sim(processed_speech, i, vect)
			elif similarity_measure == 'jaccard':
				curr_sim = jaccard_sim(processed_speech, i)
			elif similarity_measure == 'doc2vec':
				curr_sim = doc2vec_sim(speech_id, i)
			
			if curr_sim > sim_threshold:
				X_i.append(X[index_into_X])
				vectors_i.append(vectors[index_into_X])
				speech_ids_i.append(speech_id)
				names_i.append(names[index_into_X])

		print "For issue %d, found %d relevant speeches." % (i, len(X_i))

		# now split into train and test
		X_i_train = X_i[:num_train]
		X_i_test = X_i[num_train:]
		vectors_i_train = vectors_i[:num_train]
		vectors_i_test = vectors_i[num_train:]
		speech_ids_i_train = speech_ids_i[:num_train] 
		speech_ids_i_test = speech_ids_i[num_train:]
		names_i_train = names_i[:num_train]
		names_i_test = names_i[num_train:]

		result[i] = (X_i_train, X_i_test, vectors_i_train, vectors_i_test, speech_ids_i_train, speech_ids_i_test, names_i_train, names_i_test)
	
	return result


def train_test_split_4(X, parties, vectors, split=0.30, random_state=123):
	print "Number of total datapoints: %d" % len(X)

	zipped = zip(X, parties, vectors)

	random.seed(random_state)
	random.shuffle(zipped)

	combined = [list(t) for t in zip(*zipped)]
	X = combined[0]
	parties = combined[1]
	vectors = combined[2]

	num_train = int(len(X) * (1.0 - split))

	X_train = {}
	X_test = {}
	vectors_train = {} # issue id -> 1D list of labels for each of the train points
	vectors_test = {}

	X_train['party'] = X[:num_train]
	X_test['party'] = X[num_train:]
	parties_train = parties[:num_train]
	parties_test = parties[num_train:]		

	for i in range(20):
		print 'Issue %d...' % i

		# filter the training and test points to include only the ones where we have test values
		X_i = []
		issue_labels_i = [] # a single value for each datapoint

		for j in range(len(X)):
			curr_issue_label = vectors[j][i]
			if curr_issue_label != 0:
				X_i.append(X[j])
				issue_labels_i.append(curr_issue_label)

		num_train = int(len(X_i) * (1.0 - split))
		X_i_train = X_i[:num_train]
		X_i_test = X_i[num_train:]
		issue_labels_i_train = issue_labels_i[:num_train]
		issue_labels_i_test = issue_labels_i[num_train:]

		print "Num train = %d, num test = %d" % (len(X_i_train), len(X_i_test))
		
		X_train[i] = X_i_train
		X_test[i] = X_i_test
		vectors_train[i] = issue_labels_i_train
		vectors_test[i] = issue_labels_i_test

	return (X_train, X_test, parties_train, parties_test, vectors_train, vectors_test)



# Test this code.
def test():
	import pdb

	corpus_filename = '../data_processing/data_all.pickle'
	labels_filename = '../scraping/all_people'
	labels_filename2 = '../scraping/fixed_people_with_vectors_234'
	labels_filename3 = '../scraping/fixed_people_with_vectors_745'

	data = load_corpus(corpus_filename)

	# save both files, one for predicting all vectors, one for predicting party only
	save_data_split_by_speech(data, labels_filename2, '../data_processing/data_split_by_speech_nonzero_vectors_only.pickle', True, False)
	save_data_split_by_speech(data, labels_filename3, '../data_processing/data_split_by_speech_some_missing.pickle', False, False) # no datapoints skipped


	#(X, parties, vectors) = make_data(data, labels)
	#(X_train, X_dev, X_test, parties_train, parties_dev, parties_test, vectors_train, vectors_dev, vectors_test) = train_test_split(X, parties, vectors)

	#pdb.set_trace()

if __name__ == '__main__':
	test()

