# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:25:05 2018

@author: Administrator
"""



import os
os.environ['KERAS_BACKEND'] = 'theano'
#print os.environ
import numpy as np

import keras.models as models
from keras.layers import Input, merge
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *

from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model


img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 调整数据集格式
x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print np.min(x_train), np.max(x_train)
print ('x_train shape: ', x_train.shape)
print ('number of train samples: ', x_train.shape[0])
print ('number of test samples: ', x_test.shape[0])


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        

shp = x_train.shape[1:]
dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

# 建立生成器
nch = 200
g_input = Input(shape=(100,))
H = Dense(nch*14*14, init='')
