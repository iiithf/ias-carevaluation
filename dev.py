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

print('\nstarting training:')
if os.path.exists(MODEL):
  shutil.rmtree(MODEL)
os.mkdir(MODEL)
sess = tf.Session()
# savr = tf.train.Saver()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
  sess.run(train_step, {x: train_x, y_: train_y})
  pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accr = tf.reduce_mean(tf.cast(pred, tf.float32))
  accr_v = sess.run(accr, {x: train_x, y_: train_y})
  print('Epoch %d: %f accuracy' % (epoch, accr_v))
# savr.save(sess, MODEL+'/car')
