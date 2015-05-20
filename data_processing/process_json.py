import os
import json
from pprint import pprint

DATA_DIR = os.getenv('DATA_DIR')

# map from policitian name (string) to their speeches (list of strings)
data = {}

# Processes the JSON object that is contained in each file.
def process_json(json_obj):
	pass

# Processes a single file.
def process_file(filename):
	f = open(filename)
	json_obj = json.loads(f.read())
	f.close()
	process_json(json_obj)

# Call process_file() on each file in the given directory.
def process_directory(directory=DATA_DIR, num_files=float('inf')):
	count = 0
	for fname in os.listdir(directory):
		if count >= num_files:
			break
		full_path = os.path.join(directory, fname)
		process_file(full_path)
		count += 1

if __name__ == '__main__':
	process_directory(num_files=1)