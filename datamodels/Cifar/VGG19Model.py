from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import tensorflow_probability as tfp
from keras.datasets import cifar10

from absl import flags

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from absl import app
import os

flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('epochs', 150, 'Number of training epochs.')
flags.DEFINE_float('init_learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_bool('data_augmentation', True, 'Whether to use augmented data during training')
flags.DEFINE_bool('variational', False, 'Whether to use probabilistic last layer')
flags.DEFINE_bool('batch_norm', True, 'Whether to use batch normalization layers')
FLAGS = flags.FLAGS

num_classes = 10

def VGG_Model(input_shape, learning_rate=0.001, batch_norm=False, prob_last_layer=False):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, name='block1_conv1'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv4'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv4'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv4'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, name='fc2'))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, name='logits'))
    model.add(BatchNormalization()) if batch_norm else None

    if (prob_last_layer):
        model.add(tfp.layers.DistributionLambda(lambda a: tfp.distributions.Categorical(logits=a),
                                                convert_to_tensor_fn=tfp.distributions.Distribution.sample))
    else:
        model.add(Activation('softmax', name='predictions'))

    sgd = optimizers.SGD(lr=learning_rate, decay=0, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def VGG_model_and_dataset(nb_epoch=150, learning_rate=0.001,
                    batch_size=32, data_augmentation=True,
                    train=True, batch_norm=True, prob_last_layer=False, output_dir=""):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # x_train - training data(images), y_train - labels(digits)

    # Convert and pre-processing
    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    model = VGG_Model(input_shape=X_train.shape[1:], learning_rate=learning_rate,
                      batch_norm=batch_norm, prob_last_layer=prob_last_layer)
    if output_dir == "":
        cwd = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(cwd, "model/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not data_augmentation:
        filename = os.path.join(output_dir, 'TestedVGG16Model.h5')
    else:
        filename = os.path.join(output_dir, 'TestedVGG16AugModel.h5')

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

            def lr_scheduler(epoch):
                return learning_rate * (0.5 ** (epoch // 40))

            reduce_lr = LearningRateScheduler(lr_scheduler)

            model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                steps_per_epoch=X_train.shape[0] // batch_size,
                                validation_data=(X_val, Y_val),
                                epochs=nb_epoch, verbose=1, callbacks=[reduce_lr, WandbCallback(save_model=False)])
        model.save_weights(filename)
    else:
        model.load_weights(filename)

    return model, X_test, Y_test, filename

