import os

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
from selectionmethods import UncertaintyUtils
from keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from datamodels.Cifar.WideResNet import wide_resnet

for name in ('ensemble_size', 'batch_size', 'learning_rate', 'output_dir', 'seed', 'train',
             'nb_epochs', 'l2_reg', 'data_aug', 'validation_freq'):
    if name in flags.FLAGS:
        delattr(flags.FLAGS, name)

flags.DEFINE_integer('ensemble_size', 5, 'Number of ensemble members.')
flags.DEFINE_bool('train', True, 'Whether to train models.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('validation_freq', 5, 'Validation frequency in steps.')
flags.DEFINE_integer('nb_epochs', 200, 'Number of training epochs.')
flags.DEFINE_string('output_dir', './model/deepensemble/',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_float('l2_reg', 1e-4, 'L2 regularization.')
flags.DEFINE_bool('data_augmentation', True, 'Whether to train with augmented data.')

FLAGS = flags.FLAGS


# Setting LR for different number of Epochs
def lr_schedule(epoch, lr):
    new_lr = FLAGS.learning_rate
    if epoch <= 80:
        pass
    elif epoch > 80 and epoch <= 120:
        new_lr = lr * 0.1
    elif epoch > 120 and epoch <= 160:
        new_lr = lr * 0.01
    else:
        new_lr = lr * 0.001
    return new_lr

def lr_scheduler_aug(epoch, lr):
    new_lr = lr
    if epoch == 80 or epoch == 120 or epoch == 160:
        new_lr = lr * 0.1
    else:
        pass
    return new_lr

def generate_CIFAR100_dataset_and_model(ensemble_size=1,
                               train=False, learning_rate=0.1,
                               batch_size=128, validation_freq=5,
                               nb_epoch = 120,
                               output_dir="", l2_reg=0., data_augmentation=True):
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if output_dir == "":
        cwd = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(cwd, "model/deepensemble/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

   # input image dimensions
    img_rows, img_cols = 32, 32
    # The CIFAR100 images are RGB.
    img_channels = 3

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    nb_classes = 100

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    n_train = x_train.shape[0]

    num_classes = nb_classes

    seed_list = np.arange(ensemble_size)
    ensemble_filenames = []
    for i in range(ensemble_size):

        member_dir = os.path.join(output_dir, 'member_' + str(i))
        if not data_augmentation:
            member_filename = os.path.join(member_dir, 'cifar100model.h5')
        else:
            member_filename = os.path.join(member_dir, 'cifar100model_aug.h5')

        ensemble_filenames.append(member_filename)

        if (train) or ((not train) and (i==0)):

            model = wide_resnet(input_shape=(img_rows, img_cols, img_channels),
                                depth=28,
                                width_multiplier=10,
                                num_classes=num_classes,
                                l2=l2_reg,
                                seed=seed_list[i],
                                prob_last_layer=True)

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
                optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule(0, learning_rate), momentum=0.9),
                loss=cross_entropy,
                metrics=[UncertaintyUtils.MeanMetricWrapper(log_likelihood, name='log_likelihood'),
                         UncertaintyUtils.MeanMetricWrapper(accuracy, name='accuracy'),
                         UncertaintyUtils.MeanMetricWrapper(cross_entropy, name='cross_entropy')])

        if train:
            # Prepare callbacks for model saving and for learning rate adjustment.
            lr_scheduler = LearningRateScheduler(lr_scheduler_aug)

            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                           cooldown=0,
                                           patience=5,
                                           min_lr=0.5e-6)

            callbacks = [lr_scheduler]

            if not data_augmentation:
                model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    validation_split=0.2,
                    validation_freq=max(
                        (validation_freq * batch_size) // n_train, 1),
                    verbose=1,
                    callbacks=callbacks)
            else:
                print('Using real-time data augmentation.')
                # This will do preprocessing and realtime data augmentation:
                num_train = int(x_train.shape[0] * 0.9)
                num_val = x_train.shape[0] - num_train
                mask = list(range(num_train, num_train + num_val))
                x_val = x_train[mask]
                y_val = y_train[mask]

                mask = list(range(num_train))
                x_train = x_train[mask]
                y_train = y_train[mask]

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

                datagen.fit(x_train)

                # Fit the model on the batches generated by datagen.flow().
                model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                    steps_per_epoch=x_train.shape[0] // batch_size,
                                    validation_data=(x_val, y_val),
                                    epochs=nb_epoch, verbose=1,
                                    callbacks=callbacks)

            if not os.path.exists(member_dir):
                os.makedirs(member_dir)

            model.save_weights(member_filename)

    return model, x_test, y_test, ensemble_filenames

