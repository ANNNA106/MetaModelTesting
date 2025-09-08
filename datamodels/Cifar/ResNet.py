# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
# Copyright 2024 The Meta-Model Testing Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Residual Network."""

import functools
from typing import Iterable

import tensorflow as tf
import tensorflow_probability as tfp
from keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from absl import app
import os
from tensorflow.keras.utils import to_categorical
from datamodels.LoadImages import load_adv_test_data
from keras.callbacks import LearningRateScheduler

def lr_scheduler(epoch, lr):
    new_lr = lr
    if epoch <= 91:
        pass
    elif epoch > 91 and epoch <= 137:
        new_lr = lr * 0.1
    else:
        new_lr = lr * 0.01
    return new_lr

def lr_scheduler_aug(epoch, lr):
    new_lr = lr
    if epoch == 80 or epoch == 120 or epoch == 160:
        new_lr = lr * 0.1
    else:
        pass
    return new_lr


BatchNormalization = functools.partial(
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)


def Conv2D(filters, seed=None, **kwargs):
  default_kwargs = {
      'kernel_size': 3,
      'padding': 'same',
      'use_bias': False,
      # Note that we need to use the class constructor for the initializer to
      # get deterministic initialization.
      'kernel_initializer': tf.keras.initializers.HeNormal(seed=seed),
  }
  # Override defaults with the passed kwargs.
  default_kwargs.update(kwargs)
  return tf.keras.layers.Conv2D(filters, **default_kwargs)


def basic_block(
    inputs: tf.Tensor,
    filters: int,
    strides: int,
    conv_l2: float,
    bn_l2: float,
    seed: int) -> tf.Tensor:
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    conv_l2: L2 regularization coefficient for the conv kernels.
    bn_l2: L2 regularization coefficient for the batch norm layers.
    seed: random seed used for initialization.

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs

  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 3)[:, 0]

  y = Conv2D(filters,
             strides=strides,
             seed=seeds[0],
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)

  y = Conv2D(filters,
             strides=1,
             seed=seeds[1],
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  y = BatchNormalization()(y)

  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters,
               kernel_size=1,
               strides=strides,
               seed=seeds[2],
               kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(x)
  x = tf.keras.layers.add([x, y])
  x = tf.keras.layers.Activation('relu')(x)

  return x


def group(inputs, filters, strides, num_blocks, conv_l2, bn_l2, seed, skip_conv=False):
  """Group of residual blocks."""
  seeds = tf.random.experimental.stateless_split(
      [seed, seed + 1], num_blocks)[:, 0]
  x = inputs
  if (not skip_conv):
    x = basic_block(
      x,
      filters=filters,
      strides=strides,
      conv_l2=conv_l2,
      bn_l2=bn_l2,
      seed=seeds[0])
  for i in range(num_blocks - 1):
    x = basic_block(
        x,
        filters=filters,
        strides=1,
        conv_l2=conv_l2,
        bn_l2=bn_l2,
        seed=seeds[i + 1])
  return x


def resnet_32(
    input_shape: Iterable[int],
    num_classes: int,
    l2: float,
    seed: int = 42,
    prob_last_layer = False) -> tf.keras.models.Model:
  """Builds ResNet.

  Args:
    input_shape: tf.Tensor.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    seed: random seed used for initialization.

  Returns:
    tf.keras.Model.
  """
  l2_reg = tf.keras.regularizers.l2
  l2_reg_value = l2
  weight_decay_value = 1e-4

  block_layers = [5, 5, 5]

  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 5)[:, 0]

  inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32')
  x = inputs

  # Initial Conv layer along with maxPool
  x = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same')(x)

  x = group(x,
            filters=16,
            strides=1,
            num_blocks=block_layers[0],
            conv_l2=weight_decay_value,
            bn_l2=l2_reg_value,
            seed=seeds[0],
            skip_conv = True)
  x = group(x,
            filters=32,
            strides=2,
            num_blocks=block_layers[1],
            conv_l2=weight_decay_value,
            bn_l2=l2_reg_value,
            seed=seeds[1])
  x = group(x,
            filters=64,
            strides=2,
            num_blocks=block_layers[2],
            conv_l2=weight_decay_value,
            bn_l2=l2_reg_value,
            seed=seeds[2])

  x = tf.keras.layers.AveragePooling2D((8,8), padding='valid')(x)
  x = tf.keras.layers.Flatten()(x)
  logits = tf.keras.layers.Dense(num_classes, name='logits')(x)

  if (prob_last_layer):
    outputs = tfp.layers.DistributionLambda(lambda a: tfp.distributions.Categorical(logits=a),
                                          convert_to_tensor_fn=tfp.distributions.Distribution.sample)(logits)
  else:
    outputs = tf.keras.layers.Activation('softmax', name='predictions')(logits)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet32')

def Resnet_dataset_and_model(train=False, learning_rate=0.1,
                               batch_size=32, validation_freq=5,
                               training_steps=5000,
                               output_dir="", l2_reg=0., data_augmentation=True, prob_last_layer=False ):

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if output_dir == "":
        cwd = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(cwd, "model/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not data_augmentation:
        filename = os.path.join(output_dir, 'TestedResnetModel.h5')
    else:
        filename = os.path.join(output_dir, 'TestedResnetAugModel.h5')


    nb_classes = 10

    # input image dimensions
    img_rows, img_cols = 32, 32

    # The CIFAR10 images are RGB.
    img_channels = 3

    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255.
    X_test /= 255.

    n_train = X_train.shape[0]
    nb_epoch = 182
    model = resnet_32( input_shape=(img_rows, img_cols, img_channels),
                        num_classes= nb_classes,
                        l2=l2_reg,
                        seed = seed,
                        prob_last_layer=prob_last_layer)

    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr_scheduler(0, learning_rate), momentum=0.9),
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.CategoricalAccuracy(),
                     tf.keras.metrics.CategoricalCrossentropy()])

    if train:
        if not data_augmentation:

            model.fit(
                x=X_train,
                y=Y_train,
                batch_size=batch_size,
                epochs=nb_epoch,
                validation_split=0.2,
                validation_freq=max(
                    (validation_freq * batch_size) // n_train, 1),
                verbose=1)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            num_train = int(X_train.shape[0] * 0.9)
            num_val = X_train.shape[0] - num_train
            mask = list(range(num_train, num_train + num_val))
            X_val = X_train[mask]
            Y_val = Y_train[mask]

            mask = list(range(num_train))
            X_train = X_train[mask]
            Y_train = Y_train[mask]
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=4,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=4,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            datagen.fit(X_train)
            reduce_lr = LearningRateScheduler(lr_scheduler_aug)

            model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                steps_per_epoch=X_train.shape[0] // batch_size,
                                validation_data=(X_val, Y_val),
                                epochs=nb_epoch, verbose=1, callbacks=[reduce_lr])
        model.save_weights(filename)
    else:
        model.load_weights(filename)
    return model, X_test, Y_test, filename

