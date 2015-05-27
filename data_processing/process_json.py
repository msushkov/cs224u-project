import os
import json
from pprint import pprint
import pdb
import pickle

DATA_DIR = os.getenv('DATA_DIR')

# map from policitian name (string) to their speeches (list of strings)
data = {}

# Processes the JSON object that is contained in each file.
def process_json(json_obj):
	global data
	for transcript in json_obj['transcripts']:
		last_name = transcript['speaker']['name']['last'].strip()
		first_name = transcript['speaker']['name']['first'].strip()
		speech = transcript['speech'].strip()
		pos = transcript['pos'].strip()
		
		full_name = ' '.join([first_name, last_name])

		if full_name not in data:
			data[full_name] = {}
			data[full_name]['speech'] = []
			data[full_name]['pos'] = []

		data[full_name]['speech'].append(speech)
		data[full_name]['pos'].append(pos)

# Processes a single file.
def process_file(filename):
	f = open(filename)
	json_obj = json.loads(f.read())
	f.close()
	process_json(json_obj)

# Call process_file() on each file in the given directory.
def process_directory(directory=DATA_DIR, num_files=float('inf')):
	count = 0
	total_files = len(os.listdir(directory))
	for fname in os.listdir(directory):
		if count % 10 == 0:
			print "Processing file %d out of %d" % (count, total_files)

		if count >= num_files:
			break
		full_path = os.path.join(directory, fname)
		process_file(full_path)
		count += 1


# Return a dictionary with k of the entries of dict_arg.
def take(dict_arg, k):
	result = {}
	c = 0
	for key in dict_arg:
		result[key] = dict_arg[key]
		c += 1
		if c == k:
			return result
	return result


if __name__ == '__main__':
	process_directory()

	# save 'data' as a binary file for later use
	output = open('data_all.pickle', 'wb')
	pickle.dump(data, output)
	output.close()

	output = open('data_10.pickle', 'wb')
	data_10 = take(data, 10)
	pickle.dump(data_10, output)
	output.close()

	output = open('data_100.pickle', 'wb')
	data_100 = take(data, 100)
	pickle.dump(data_100, output)
	output.close()

	'''
	to load a pickle file:
	pkl_file = open('data.pkl', 'rb')
	data1 = pickle.load(pkl_file)
	pkl_file.close()
	'''
