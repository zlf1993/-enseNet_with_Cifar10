# -*- coding:utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Callback
from keras.models import load_model
from data_input.data_input import getDataGenerator, loadcifar10, generate_batch_data_random
from keras.applications.densenet import DenseNet121
from keras.layers import Dense
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

# define DenseNet parms
ROWS = 32
COLS = 32
CHANNELS = 3
nb_classes = 10
batch_size = 500
samples_per_epoch = 50000 // 500
nb_epoch = 100
img_dim = (ROWS, COLS, CHANNELS)
densenet_depth = 40
densenet_growth_rate = 12

# define filepath parms
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


def main(resume=False):
    print('Now,we start compiling DenseNet model...')

    if resume == False:
        basemodel = DenseNet121(include_top=False, weights=None, input_shape=(32, 32, 3), pooling='avg')
    else:
        basemodel = DenseNet121(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')

    # x = basemodel.layers[-1].output
    x = basemodel.output
    x = Dense(100, activation='sigmoid', name='fc100')(x)
    prediction = Dense(nb_classes, activation='softmax', name='fc10')(x)
    model = Model(input=basemodel.input, output=prediction)

    if resume == False:
        for layer in basemodel.layers:
            layer.trainable = True
    else:
        print("Setting Trainable Permission")
        for layer in basemodel.layers:
            layer.trainable = True
        basemodel.layers[1].trainable = False
        basemodel.layers[2].trainable = False
        basemodel.layers[3].trainable = False
        basemodel.layers[4].trainable = False
        basemodel.layers[5].trainable = False
        basemodel.layers[6].trainable = False
        """ for i in range(16):
            basemodel.layers[(i+1)].trainable = False"""

    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # start loading data
    # start loading data
    x_train, y_train, x_test, y_test = loadcifar10(nb_classes)
    train_datagen = getDataGenerator(train_phase=True)
    train_datagen = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_datagen = getDataGenerator(train_phase=False)
    validation_datagen = validation_datagen.flow(x_test, y_test, batch_size=batch_size)

    # defining callback functions
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = "model_{epoch:02d}-{val_acc:.2f}.hdf5"
    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.1, cooldown=0, patience=7, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor="val_acc", save_best_only=True,
                                       save_weights_only=True, verbose=1, mode='auto', period=1)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0)
    history = LossHistory()
    # callbacks = [lr_reducer, model_checkpoint, tensorboard, history]
    callbacks = [model_checkpoint, history, lr_reducer]

    # training
    """model.fit_generator(generator=train_datagen, steps_per_epoch=samples_per_epoch, epochs=nb_epoch, verbose=1,
                                    callbacks=callbacks,
                                    validation_data=generate_batch_data_random(x_test, y_test, batch_size),
                                    nb_val_samples=x_test.shape[0] // batch_size)"""
    model.fit_generator(generator=train_datagen, steps_per_epoch=samples_per_epoch, epochs=nb_epoch, verbose=1,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test))

    history.loss_plot('epoch')
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            plt.savefig("./result.jpg")

if __name__ == '__main__':
    K.set_image_data_format('channels_last')

    main(resume=False)
