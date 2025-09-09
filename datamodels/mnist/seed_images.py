# run this once (e.g., in a notebook cell from /content/MetaModelTesting)
import os, numpy as np
from keras.datasets import mnist
from PIL import Image

seed_size = 5000
out_dir = "/content/MetaModelTesting/datamodels/mnist/data/images/mnist_seed"
os.makedirs(out_dir, exist_ok=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Take a class-balanced subset from test set:
per_class = seed_size // 10
counts = {k:0 for k in range(10)}
i = 0
saved = 0
while saved < seed_size and i < len(x_test):
    lbl = int(y_test[i])
    if counts[lbl] < per_class:
        img = Image.fromarray(x_test[i].astype('uint8'))
        img.save(os.path.join(out_dir, f"mnistorig_{lbl}_{counts[lbl]}.png"))
        counts[lbl] += 1
        saved += 1
    i += 1

print("Wrote", saved, "seed images to", out_dir)
