import os
import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from keras.utils.np_utils import to_categorical
from datamodels.LoadImages import load_adv_test_data
from absl import app
import tensorflow_probability as tfp

def lenet1(input_shape, num_classes, prob_last_layer=False):
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(4,
                                 kernel_size=5,
                                 padding='valid',
                                 activation='relu')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2])(conv1)
    conv2 = tf.keras.layers.Conv2D(12,
                                 kernel_size=5,
                                 padding='valid',
                                 activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv2)
    flatten = tf.keras.layers.Flatten()(pool2)
    logits = tf.keras.layers.Dense(num_classes, name='logits')(flatten)
    if prob_last_layer:
        outputs = tfp.layers.DistributionLambda(lambda x: tfp.distributions.Categorical(logits=x),
                                                convert_to_tensor_fn=tfp.distributions.Distribution.sample)(logits)
    else:
        outputs = tf.keras.layers.Activation('softmax', name='predictions')(logits)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def lenet5(input_shape, num_classes, prob_last_layer=False):
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(6,
                                 kernel_size=5,
                                 padding='SAME',
                                 activation='relu')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv1)
    conv2 = tf.keras.layers.Conv2D(16,
                                 kernel_size=5,
                                 activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv2)
    flatten = tf.keras.layers.Flatten()(pool2)
    dense0 = tf.keras.layers.Dense(120, activation=tf.nn.relu)(flatten)
    dense1 = tf.keras.layers.Dense(84, activation=tf.nn.relu)(dense0)
    logits = tf.keras.layers.Dense(num_classes, name='logits')(dense1)
    if prob_last_layer:
        outputs = tfp.layers.DistributionLambda(lambda x: tfp.distributions.Categorical(logits=x),
                                                convert_to_tensor_fn=tfp.distributions.Distribution.sample)(logits)
    else:
        outputs = tf.keras.layers.Activation('softmax', name='predictions')(logits)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def LeNet_dataset_and_model(dataset, train=False, learning_rate=0.001,
                               batch_size=256, validation_freq=5,
                               training_steps=5000,
                               output_dir="", lenet_family='Lenet5'):
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if output_dir == "":
        cwd = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(cwd, "model/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset == 'mnist':
        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.
        x_test /= 255.

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    n_train = 60000

    num_classes = 10

    if lenet_family == "Lenet5":
        model = lenet5(x_train.shape[1:], num_classes)
        filename = os.path.join(output_dir, 'TestedLenet5Model.h5')
    else:
        model = lenet1(x_train.shape[1:], num_classes)
        filename = os.path.join(output_dir, 'TestedLenet1Model.h5')

    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.CategoricalAccuracy(),
                     tf.keras.metrics.CategoricalCrossentropy()])

    if train:
        model.fit(
                x=x_train,
                y=y_train,
                batch_size=batch_size,
                epochs=(batch_size * training_steps) // n_train,
                validation_split=0.2,
                validation_freq=max(
                    (validation_freq * batch_size) // n_train, 1),
                verbose=1)
        model.save_weights(filename)
    else:
        model.load_weights(filename)
    return model, x_test, y_test, filename

