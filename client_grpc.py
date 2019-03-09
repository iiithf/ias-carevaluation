#!/usr/bin/env python3
from http.client import HTTPConnection
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf
import optparse
import json
import grpc
import time


def example_make(example):
  features = {}
  for k, v in example.items():
    lst = tf.train.BytesList(value=[v.encode('ascii')])
    feature = tf.train.Feature(bytes_list=lst)
    features[k] = feature
  return tf.train.Example(features=tf.train.Features(feature=features))

def example_json(buying, maint, doors, persons, lug_boot, safety):
  return {'buying': buying, 'maint': maint, 'doors': doors,
  'persons': persons, 'lug_boot': lug_boot, 'safety': safety}

def classify(conn, model, method, example):
  channel = grpc.insecure_channel('127.0.0.1:8500')
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'model'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['inputs'].CopyFrom(make_tensor_proto(example_make(example).SerializeToString(), shape=[1]))
  return stub.Predict(request, 10.0)


p = optparse.OptionParser()
p.set_defaults(service='127.0.0.1:8500',model='model',method='serving_default')
p.add_option('--service', dest='service', help='set tensorflow serving address')
p.add_option('--model', dest='model', help='set model name to use')
p.add_option('--method', dest='method', help='set method to use')
(o, args) = p.parse_args()

channel = grpc.insecure_channel(o.service)
print('CarEvaluation gRPC TF client:\n')
buying = 'vhigh' # input('buying price (vhigh, high, med, low):')
maint = 'vhigh' # input('price of the maintenance (vhigh, high, med, low):')
doors = '2' # input('number of doors (2, 3, 4, 5more):')
persons = '2' # input('capacity in terms of persons to carry (2, 4, more):')
lug_boot = 'small' # input('the size of luggage boot (small, med, big):')
safety = 'low' # input('estimated safety of the car (low, med, high):')
example = example_json(buying, maint, doors, persons, lug_boot, safety)

print()
start = time.time()
result = classify(channel, o.model, o.method, example)
end = time.time()
classes = result.outputs['classes'].string_val
scores = result.outputs['scores'].float_val
print('\nresult in %3f seconds:' % (end-start))
for i in range(len(classes)):
  print('%s: %s' % (classes[i].decode('ascii'), scores[i]))
