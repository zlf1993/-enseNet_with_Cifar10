from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Activation
import numpy as np


def DenseNet121_withImagenet():
    model = DenseNet121(include_top=False, weights='imagenet', input_shape=(32,32,3), pooling='avg')
    return model


def DenseNet121():
    model = DenseNet121(include_top=False, weights=None, input_shape=(32, 32, 3), pooling='avg')
    return model
