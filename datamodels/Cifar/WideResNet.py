# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Wide Residual Network."""

import functools
from typing import Dict, Iterable, Optional

import tensorflow as tf
import tensorflow_probability as tfp
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
from absl import app
import os
from datamodels.LoadImages import load_adv_test_data
from keras.callbacks import LearningRateScheduler

from wandb.keras import WandbCallback

_HP_KEYS = ('bn_l2', 'input_conv_l2', 'group_1_conv_l2', 'group_2_conv_l2',
            'group_3_conv_l2', 'dense_kernel_l2', 'dense_bias_l2')

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)


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

def get_wide_resnet_hp_keys():
  """Returns the hyperparameter keys used in the wide ResNet model."""
  return _HP_KEYS


def Conv2D(filters, seed=None, **kwargs):  # pylint: disable=invalid-name
  """Conv2D layer that is deterministically initialized."""
  default_kwargs = {
      'kernel_size': 3,
      'padding': 'same',
      'use_bias': False,
      'kernel_initializer': tf.keras.initializers.HeNormal(seed=seed),
  }
  # Override defaults with the passed kwargs.
  default_kwargs.update(kwargs)
  return tf.keras.layers.Conv2D(filters, **default_kwargs)


def basic_block(inputs: tf.Tensor, filters: int, strides: int, conv_l2: float,
                bn_l2: float, seed: int, version: int) -> tf.Tensor:
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    conv_l2: L2 regularization coefficient for the conv kernels.
    bn_l2: L2 regularization coefficient for the batch norm layers.
    seed: random seed used for initialization.
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  if version == 2:
    y = BatchNormalization(
        beta_regularizer=tf.keras.regularizers.l2(bn_l2),
        gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(
            y)
    y = tf.keras.layers.Activation('relu')(y)
  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 3)[:, 0]

  y = Conv2D(
      filters,
      strides=strides,
      seed=seeds[0],
      kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(
          y)
  y = BatchNormalization(
      beta_regularizer=tf.keras.regularizers.l2(bn_l2),
      gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(
          y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(
      filters,
      strides=1,
      seed=seeds[1],
      kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(
          y)
  if version == 1:
    y = BatchNormalization(
        beta_regularizer=tf.keras.regularizers.l2(bn_l2),
        gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(
            y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(
        filters,
        kernel_size=1,
        strides=strides,
        seed=seeds[2],
        kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(
            x)
  x = tf.keras.layers.add([x, y])
  if version == 1:
    x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, strides, num_blocks, conv_l2, bn_l2, version, seed):
  """Group of residual blocks."""
  seeds = tf.random.experimental.stateless_split([seed, seed + 1], num_blocks)[:, 0]

  x = basic_block(
      inputs,
      filters=filters,
      strides=strides,
      conv_l2=conv_l2,
      bn_l2=bn_l2,
      version=version,
      seed=seeds[0])
  for i in range(num_blocks - 1):
    x = basic_block(
        x,
        filters=filters,
        strides=1,
        conv_l2=conv_l2,
        bn_l2=bn_l2,
        version=version,
        seed=seeds[i + 1])
  return x


def _parse_hyperparameters(l2: float, hps: Dict[str, float]):
  """Extract the L2 parameters for the dense, conv and batch-norm layers."""

  assert_msg = ('Ambiguous hyperparameter specifications: either l2 or hps '
                'must be provided (received {} and {}).'.format(l2, hps))
  is_specified = lambda h: bool(h) and all(v is not None for v in h.values())
  only_l2_is_specified = l2 is not None and not is_specified(hps)
  only_hps_is_specified = l2 is None and is_specified(hps)
  assert only_l2_is_specified or only_hps_is_specified, assert_msg
  if only_hps_is_specified:
    assert_msg = 'hps must contain the keys {}!={}.'.format(
        _HP_KEYS, hps.keys())
    assert set(hps.keys()).issuperset(_HP_KEYS), assert_msg
    return hps
  else:
    return {k: l2 for k in _HP_KEYS}


def wide_resnet(
    input_shape: Iterable[int],
    depth: int,
    width_multiplier: float,
    num_classes: int,
    l2: float,
    version: int = 2,
    seed: int = 42,
    prob_last_layer=False,
    hps: Optional[Dict[str, float]] = None) -> tf.keras.models.Model:
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).
    seed: random seed used for initialization.
    hps: Fine-grained specs of the hyperparameters, as a Dict[str, float].

  Returns:
    tf.keras.Model.
  """
  l2_reg = tf.keras.regularizers.l2
  hps = _parse_hyperparameters(l2, hps)

  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 5)[:, 0]
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = Conv2D(
      16,
      strides=1,
      seed=seeds[0],
      kernel_regularizer=l2_reg(hps['input_conv_l2']))(
          inputs)
  if version == 1:
    x = BatchNormalization(
        beta_regularizer=l2_reg(hps['bn_l2']),
        gamma_regularizer=l2_reg(hps['bn_l2']))(
            x)
    x = tf.keras.layers.Activation('relu')(x)
  x = group(
      x,
      filters=round(16 * width_multiplier),
      strides=1,
      num_blocks=num_blocks,
      conv_l2=hps['group_1_conv_l2'],
      bn_l2=hps['bn_l2'],
      version=version,
      seed=seeds[1])
  x = group(
      x,
      filters=round(32 * width_multiplier),
      strides=2,
      num_blocks=num_blocks,
      conv_l2=hps['group_2_conv_l2'],
      bn_l2=hps['bn_l2'],
      version=version,
      seed=seeds[2])
  x = group(
      x,
      filters=round(64 * width_multiplier),
      strides=2,
      num_blocks=num_blocks,
      conv_l2=hps['group_3_conv_l2'],
      bn_l2=hps['bn_l2'],
      version=version,
      seed=seeds[3])
  if version == 2:
    x = BatchNormalization(
        beta_regularizer=l2_reg(hps['bn_l2']),
        gamma_regularizer=l2_reg(hps['bn_l2']))(
            x)
    x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  logits = tf.keras.layers.Dense(
      num_classes, name='logits',
      kernel_initializer=tf.keras.initializers.HeNormal(seed=seeds[4]),
      kernel_regularizer=l2_reg(hps['dense_kernel_l2']),
      bias_regularizer=l2_reg(hps['dense_bias_l2']))(
          x)

  if (prob_last_layer):
    outputs = tfp.layers.DistributionLambda(lambda a: tfp.distributions.Categorical(logits=a),
                                          convert_to_tensor_fn=tfp.distributions.Distribution.sample)(logits)
  else:
    outputs = tf.keras.layers.Activation('softmax', name='predictions')(logits)

  return tf.keras.Model(
      inputs=inputs,
      outputs=outputs,
      name='wide_resnet-{}-{}'.format(depth, width_multiplier))


def WideResnet_dataset_and_model(train=False, learning_rate=0.1,
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
        filename = os.path.join(output_dir, 'TestedWideResnetModel.h5')
    else:
        filename = os.path.join(output_dir, 'TestedWideResnetAugModel.h5')


    nb_classes = 100

    # input image dimensions
    img_rows, img_cols = 32, 32

    # The CIFAR100 images are RGB.
    img_channels = 3

    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255.
    X_test /= 255.

    n_train = X_train.shape[0]
    nb_epoch = 200
    #nb_epoch = (batch_size * training_steps) // n_train,
    model = wide_resnet(input_shape=(img_rows, img_cols, img_channels),
                        depth=28,
                        width_multiplier=10,
                        num_classes=nb_classes,
                        l2=l2_reg,
                        seed=seed,
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
                verbose=1,
                callbacks=[WandbCallback(save_model=False)])
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
                                epochs=nb_epoch, verbose=1, callbacks=[reduce_lr, WandbCallback(save_model=False)])
        model.save_weights(filename)
    else:
        model.load_weights(filename)
    return model, X_test, Y_test, filename

