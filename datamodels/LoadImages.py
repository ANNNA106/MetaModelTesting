import os
from PIL import Image
import numpy as np
import re


def load_adv_images(image_directory, num_rows, num_cols, num_chnls, transformation, features_data, label_data, adv_count=100, load_all_data=False):
    index = 0
    aug_features_data = features_data
    aug_label_data = label_data
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        file_name_split = re.split("[_ \.]", file)
        if ".png" in file and file_name_split[0] == transformation:
            image_file_name = os.path.join(image_directory, file)
            if (num_chnls == 1):
                img = Image.open(image_file_name).convert("L")
            else:
                img = Image.open(image_file_name)
            img = np.resize(img, (num_rows,num_cols,num_chnls))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,num_rows,num_cols,num_chnls)
            im2arr = im2arr.astype('float32')
            im2arr /= 255

            if aug_features_data is None:
                aug_features_data = np.zeros((1, num_rows, num_cols, num_chnls), dtype=float)
                aug_features_data[0] = im2arr
            else:
                aug_features_data = np.append(aug_features_data, im2arr, axis=0)

            image_label = int(file_name_split[1])
            if aug_label_data is None:
                aug_label_data = np.zeros((1,), dtype=np.uint8)
                aug_label_data[0] = image_label
            else:
                aug_label_data = np.append(aug_label_data, [image_label], axis=0)

            index = index+1

        if (not load_all_data) and (index >= adv_count):
            break
    print("number of adv samples are : " + str(index))
    return aug_features_data, aug_label_data

def load_adv_test_data(dataset, x_test, y_test, adv_img_path, transformation, adv_count, load_all_data = False):
    if (dataset == 'mnist'):
        adv_x_test, adv_y_test = load_adv_images(adv_img_path, 28, 28, 1, transformation, x_test, y_test, adv_count, load_all_data)
    else:
        adv_x_test, adv_y_test = load_adv_images(adv_img_path, 32, 32, 3, transformation,
                                                 x_test, y_test, adv_count, load_all_data)
    return adv_x_test, adv_y_test


def deprocess_image(x, img_rows, img_cols, img_depth):
  x = np.clip(x, 0, 255).astype('uint8')
  return x.reshape(img_rows, img_cols, img_depth)

