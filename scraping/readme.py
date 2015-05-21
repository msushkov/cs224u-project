#people_with_vectors is a list of pople objects

import json

f = open("people_with_vectors_108")
people = json.load(f)

for p in people:
    print p["name"], p["vector"] # this is a 20D vector
