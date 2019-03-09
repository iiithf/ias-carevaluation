#!/usr/bin/env python
from http.client import HTTPConnection
import optparse
import json
import time


def parse_addr(addr):
  i = addr.find(':')
  host = '' if i<0 else addr[0:i]
  port = int(addr if i<0 else addr[i+1:])
  return (host, port)

def input_json(buying, maint, doors, persons, lug_boot, safety):
  return {'buying': buying, 'maint': maint, 'doors': doors,
  'persons': persons, 'lug_boot': lug_boot, 'safety': safety}

def classify(conn, model, method, example):
  path = '/v1/models/%s:%s' % (model, method)
  conn.request('POST', path, body=json.dumps({'examples': [example]}))
  resp = conn.getresponse()
  data = json.loads(resp.read())
  if 'error' in data:
    raise Exception(data['error'])
  return data['results']


p = optparse.OptionParser()
p.set_defaults(service='127.0.0.1:8501',model='model',method='classify')
p.add_option('--service', dest='service', help='set tensorflow serving address')
p.add_option('--model', dest='model', help='set model name to use')
p.add_option('--method', dest='method', help='set method to use')
(o, args) = p.parse_args()

host, port = parse_addr(o.service)
conn = HTTPConnection(host, port)
print('CarEvaluation TF serving demo:\n')
buying = 'vhigh' # input('buying price (vhigh, high, med, low):')
maint = 'vhigh' # input('price of the maintenance (vhigh, high, med, low):')
doors = '2' # input('number of doors (2, 3, 4, 5more):')
persons = '2' # input('capacity in terms of persons to carry (2, 4, more):')
lug_boot = 'small' # input('the size of luggage boot (small, med, big):')
safety = 'low' # input('estimated safety of the car (low, med, high):')
example = input_json(buying, maint, doors, persons, lug_boot, safety)

start = time.time()
results = classify(conn, o.model, o.method, example)
end = time.time()
print('\nresult in %3f seconds:' % (end-start))
for r in results[0]:
  print('%s:' % r[0], r[1])
