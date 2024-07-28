import numpy as np
import tensorflow as tf
from keras import layers
import os
from PIL import Image
from datamodels.LoadImages import load_adv_test_data
import tensorflow_io as tfio
transformationtype = ['Translate', 'Brightness', 'Rotate', 'Blur']

def deprocess_image(x, img_rows, img_cols, img_depth):
  x = np.clip(x, 0, 255).astype('uint8')
  if (img_depth==1):
    return x.reshape(img_rows, img_cols)
  else:
    return x.reshape(img_rows, img_cols, img_depth)

def transform_images(dataset, trans_type, sample_size, image_directory):

    cwd = os.path.dirname(os.path.abspath(__file__))
    image_directory = os.path.join(cwd, image_directory)
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    if dataset == 'cifar100':
        img_rows, img_cols, img_depth = 32, 32, 3
        orig_img_path = os.path.join(cwd, '../datamodels/Cifar/data/images/cifar100/cifar100_seed')
        selected_images, selected_labels = load_adv_test_data('cifar100', None, None,
                                                orig_img_path, 'cifarorig', sample_size, load_all_data=False)
        selected_images = selected_images * 255.0
    if dataset == 'cifar10':
        img_rows, img_cols, img_depth = 32, 32, 3
        orig_img_path = os.path.join(cwd, '../datamodels/Cifar/data/images/cifar10/cifar10_seed')
        selected_images, selected_labels = load_adv_test_data('cifar10', None, None,
                                                orig_img_path, 'cifarorig', sample_size, load_all_data=False)
        selected_images = selected_images * 255.0
    if dataset == 'mnist':
        img_rows, img_cols, img_depth = 28, 28, 1
        orig_img_path = os.path.join(cwd, '../datamodels/mnist/data/images/mnist_seed')
        selected_images, selected_labels = load_adv_test_data('mnist', None, None,
                                                orig_img_path, 'mnistorig', sample_size, load_all_data=False)
        selected_images = selected_images * 255.0

    if trans_type == 'Rotate':
        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(factor=0.08)
        ])
    if trans_type == 'Translate':
        data_augmentation = tf.keras.Sequential([
            layers.RandomTranslation(0.2, 0.2, fill_mode='constant', fill_value=0)
        ])

    for i in range(len(selected_images)):
        filename = trans_type + "_" + str(selected_labels[i]) + "_" + str(i) + ".png"
        image_file_name = os.path.join(image_directory, filename)
        reshaped_image = np.array(selected_images[i]).reshape(1,img_rows, img_cols,img_depth)

        if trans_type in ['Rotate', 'Translate']:
            transformed_image = np.array(data_augmentation(reshaped_image))
        elif trans_type == 'Brightness':
            seed = (1, 2)
            image_to_transform = selected_images[i] / 255
            transformed_image = np.array(tf.image.stateless_random_brightness(image_to_transform, 0.1, seed))
            transformed_image = transformed_image * 255
        elif trans_type == 'Blur':
            transformed_image = tfio.experimental.filter.gaussian(input=reshaped_image, ksize=[6,6], sigma=[1,1])

        deprocessed_img = deprocess_image(transformed_image, img_rows, img_cols, img_depth)
        img = Image.fromarray(deprocessed_img)
        img.save(image_file_name)

