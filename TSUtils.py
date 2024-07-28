from absl import logging
import os
import datetime
import absl
import numpy as np
from PIL import Image

def saveimage(img_arr, img_rows, img_cols, img_chnls, filename, output_dir):
    img_arr = img_arr * 255
    img_arr = np.clip(img_arr, 0, 255).astype('uint8')
    if (img_chnls == 1):
        img_arr = img_arr.reshape(img_rows, img_cols)
    else:
        img_arr = img_arr.reshape(img_rows, img_cols, img_chnls)
    img = Image.fromarray(img_arr)
    ct = datetime.datetime.now()
    filename = filename + "_" + str(ct) + ".png"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name_path = os.path.join(output_dir, filename)
    img.save(file_name_path)