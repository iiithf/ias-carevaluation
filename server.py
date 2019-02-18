#!/usr/bin/env python3
from service_stub import ServiceStub, ServiceProcedure
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import shutil
import csv
import os

MODEL = 'build'
NBUYING = ['vhigh', 'high', 'med', 'low']
NMAINT = ['vhigh', 'high', 'med', 'low']
NDOORS = ['2', '3', '4', '5more']
NPERSONS = ['2', '4', 'more']
NLUG_BOOT = ['small', 'med', 'big']
NSAFETY = ['low', 'med', 'high']
NACCEPT = ['unacc', 'acc', 'good', 'vgood']


def attrib_numbers(row):
  buying = float(NBUYING.index(row[0]))
  maint = float(NMAINT.index(row[1]))
  doors = float(NDOORS.index(row[2]))
  persons = float(NPERSONS.index(row[3]))
  lug_boot = float(NLUG_BOOT.index(row[4]))
  safety = float(NSAFETY.index(row[5]))
  return [buying, maint, doors, persons, lug_boot, safety]

def class_numbers(accept):
  numbers = [0.0, 0.0, 0.0, 0.0]
  numbers[NACCEPT.index(accept)] = 1.0
  return numbers

def ann_layer(x, size):
  w = tf.Variable(tf.truncated_normal(size))
  b = tf.Variable(tf.truncated_normal(size[-1:]))
  return tf.add(tf.matmul(x, w), b)

def ann_network(x):
  h1 = tf.nn.relu(ann_layer(x, [6, 48]))
  h2 = tf.nn.sigmoid(ann_layer(h1, [48, 48]))
  h3 = tf.nn.relu(ann_layer(h2, [48, 48]))
  return ann_layer(h3, [48, 4])


def get_data(name, test_per):
  x, y = ([], [])
  with open(name, 'r') as f:
    for row in csv.reader(f):
      x.append(attrib_numbers(row))
      y.append(class_numbers(row[6]))
  x, y = shuffle(x, y)
  return train_test_split(x, y, test_size=test_per)

def prediction(sess, y, row):
  test_x = [attrib_numbers(row)]
  test_y = [[1.0, 0.0, 0.0, 0.0]]
  return NACCEPT[sess.run(tf.argmax(y,1), {x: test_x, y_: test_y})[0]]


print('reading dataset:')
inps, outs = (6, 4)
rate, epochs = (0.1, 200)
train_x, test_x, train_y, test_y = get_data('car.data', 0.2)
print('%d train rows, %d test rows' % (len(train_x), len(test_x)))

print('\ndefining ann:')
x = tf.placeholder(tf.float32, [None, inps])
y_ = tf.placeholder(tf.float32, [None, outs])
y = ann_network(x)
cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(rate).minimize(cost_func)

print('\nloading model:')
sess = tf.Session()
savr = tf.train.Saver()
sess.run(tf.global_variables_initializer())
savr.restore(sess, MODEL+'/car')

print('\nstarting server:')
def predict(buying, maint, doors, persons, lug_boot, safety):
  return prediction(sess, y, [buying, maint, doors, persons, lug_boot, safety])

def close():
  exit()

def service_setup(name=''):
  serv = ServiceStub(name)
  serv.add('str predict(str buying, str maint, str doors, str persons, str lug_boot, str safety)', predict)
  serv.add('str close()', close)
  return serv

addr = ('', 1992)
midw = ('127.0.0.1', 1992)
service = service_setup('memkart')
print('Starting service on %s -> %s' % (addr, midw))
service.start(addr)
