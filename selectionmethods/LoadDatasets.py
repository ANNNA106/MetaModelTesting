from keras.datasets import cifar10, mnist, cifar100

def loadCifarDataSet(datasetname):
    if (datasetname == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    return x_train, y_train, x_test, y_test

def loadMnistDataSet():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train[:100]
    # y_train = y_train[:100]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    return x_train, y_train, x_test, y_test