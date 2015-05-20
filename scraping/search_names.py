#!/usr/bin/python
import json
import urllib
import random
from time import sleep

def showsome(searchfor):
  query = urllib.urlencode({'q': searchfor})
  url = 'http://ajax.googleapis.com/ajax/services/search/web?v=1.0&%s' % query
  search_response = urllib.urlopen(url)
  search_results = search_response.read()
  results = json.loads(search_results)
  data = results['responseData']
  if not data:
    return ""
  print 'Total results: %s' % data['cursor']['estimatedResultCount']
  hits = data['results']
  return hits[0]['url']
  #print 'Top %d hits:' % len(hits)
  #for h in hits: print ' ', h['url']
  #print 'For more results, see %s' % data['cursor']['moreResultsUrl']

f = open("all_names")
lines = f.readlines()
random.shuffle(lines)

hit = 0
miss = 0
total = 0

for l in lines[0:50]:
    parts = l.split(" ")
    query = l + " www.ontheissues.org"
    res = showsome(query)
    if res == "":
      continue
    total += 1
    if "ontheissues" in res and (parts[0] in res or parts[1] in res):
        hit += 1
        print "HIT: " + l + " " + res
    else:
        miss +=1
        print "MISS: " + l + " " + res
    sleep(30)

print hit
print miss
print total

