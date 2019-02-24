# -*- coding:utf-8 -*-
import time

import keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img
from keras.datasets import cifar10
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Flatten
from keras.models import Model
from sklearn.svm import SVC

batch_size = 500
check_point_file = save_dir = os.path.join(os.getcwd(), 'saved_models1/model_95-0.85.hdf5')

def eval_model():
    basemodel = DenseNet121(include_top=False, weights=None, input_shape=(32, 32, 3), pooling='avg')
    deep_feature = basemodel.output
    x = Dense(100, activation='sigmoid', name='fc100')(deep_feature)
    prediction = Dense(10, activation='softmax', name='fc10')(x)
    model = Model(input=basemodel.input, output=prediction)

    model.load_weights(check_point_file)
    submodel = Model(input=basemodel.input, output=deep_feature)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    deep_feature1 = submodel.predict(x_train)
    deep_feature2 = submodel.predict(x_test)

    start = time.clock()

    clf = SVC(kernel='poly')
    clf.fit(deep_feature1, y_train)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

    score = clf.score(deep_feature2, y_test)
    print(" score: {:.6f}".format(score))


if __name__ =='__main__':
    eval_model()