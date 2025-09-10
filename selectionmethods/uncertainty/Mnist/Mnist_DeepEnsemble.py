import os

from absl import app
from absl import flags

import numpy as np
from keras.datasets import mnist
import tensorflow as tf
from selectionmethods import UncertaintyUtils
from datamodels.mnist.LeNet import lenet5, lenet1

for name in ('dataset', 'ensemble_size', 'batch_size', 'learning_rate', 'validation_freq', 'output_dir',
             'train', 'training_steps'):
    if name in flags.FLAGS:
        delattr(flags.FLAGS, name)

flags.DEFINE_enum('dataset', 'mnist',
                  enum_values=['mnist'],
                  help='Name of the image dataset.')
flags.DEFINE_integer('ensemble_size', 4, 'Number of ensemble members.')
flags.DEFINE_integer('training_steps', 5000, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('validation_freq', 5, 'Validation frequency in steps.')
flags.DEFINE_string('output_dir', './model/deepensemble/',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_bool('train', False, 'Whether to train models.')

FLAGS = flags.FLAGS


def generate_dataset_and_model(dataset, ensemble_size=1,
                               train=False, learning_rate=0.001,
                               batch_size=256, validation_freq=5,
                               training_steps=5000,
                               output_dir="", lenet_family='Lenet5'):
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if output_dir == "":
        cwd = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(cwd, "model/deepensemble/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        n_train = 60000

    num_classes = int(np.amax(y_train)) + 1
    seed_list = np.arange(ensemble_size)
    ensemble_filenames = []
    for i in range(ensemble_size):
        member_dir = os.path.join(output_dir, 'member_' + str(i))
        member_filename = os.path.join(member_dir, lenet_family+'model.h5')
        ensemble_filenames.append(member_filename)

        if (train) or ((not train) and (i==0)):
            if lenet_family == "Lenet5":
                model = lenet5(x_train.shape[1:], num_classes, prob_last_layer=True)
            else:
                model = lenet1(x_train.shape[1:], num_classes, prob_last_layer=True)

            def negative_log_likelihood(y_true, y_pred):
                return -tf.reduce_mean(y_pred.log_prob(tf.squeeze(y_true)))

            def accuracy(y_true, y_pred):
                return tf.reduce_mean(tf.cast(tf.math.equal(
                                            tf.math.argmax(input=y_pred.logits, axis=1),
                                            tf.cast(tf.squeeze(y_true), tf.int64)), tf.float32))

            def log_likelihood(y_true, y_pred):
                return tf.reduce_mean(y_pred.log_prob(tf.squeeze(y_true)))

            def cross_entropy(y_true, y_pred):
                return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.squeeze(y_true),
                                                                                  y_pred.logits,
                                                                                  from_logits=True))

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=negative_log_likelihood,
                metrics=[UncertaintyUtils.MeanMetricWrapper(log_likelihood, name='log_likelihood'),
                         UncertaintyUtils.MeanMetricWrapper(accuracy, name='accuracy'),
                         UncertaintyUtils.MeanMetricWrapper(cross_entropy, name='cross_entropy')])

            if train:
                model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=(batch_size * training_steps) // n_train,
                    validation_data=(x_test, y_test),
                    validation_freq=max((validation_freq * batch_size) // n_train, 1),
                    verbose=1)
                if not os.path.exists(member_dir):
                    os.makedirs(member_dir)

                model.save_weights(member_filename)
    print("deep enssemble initialized")

    return model, x_test, y_test, ensemble_filenames

