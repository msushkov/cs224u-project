import os
import json
from pprint import pprint
import pdb
from collections import defaultdict

DATA_DIR = os.getenv('DATA_DIR')

# map from policitian name (string) to their speeches (list of strings)
data = defaultdict(list)

# Processes the JSON object that is contained in each file.
def process_json(json_obj):
	global data
	for transcript in json_obj['transcripts']:
		last_name = transcript['speaker']['name']['last']
		first_name = transcript['speaker']['name']['first']
		speech = transcript['speech']
		
		full_name = ' '.join([first_name, last_name])
		data[full_name].append(speech)

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

if __name__ == '__main__':
	process_directory()
	pdb.set_trace()