import  data as ut
import numpy as np
import tensorflow as tf
# 3 GPU limit memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)
tf.config.experimental.set_memory_growth(physical_devices[2], True)

from model import  Model
#------------------------------------------
# tx,ty=ut.load_data()
# print(type(tx))
# print(tf.shape(tx))

#-----------------------------------------
# d=ut.get_samples()
# #
# # print(d.shape)
#-----------------------------------------

# model=Model()
#-----------------------------------------


b = tf.Variable(tf.ones([1,20,20,1]))
wc1 = tf.Variable(tf.ones([5,5,1,1]))



m1 = tf.nn.conv2d(b, wc1, strides=[1,5,5,1], padding='SAME')
m2 = tf.nn.conv2d(b, wc1, strides=[1,5,5,1], padding='VALID')
m3 = tf.nn.conv2d(b, wc1, strides=[1,5,5,1])
print(tf.get_shape(m1))
print(tf.get_shape(m2))
print(tf.get_shape(m3))

