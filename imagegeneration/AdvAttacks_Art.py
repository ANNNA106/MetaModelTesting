
import art
from keras.datasets import mnist, cifar10, cifar100
import numpy as np
import csv
from keras.models import load_model
import argparse
import tensorflow as tf
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentTensorFlowV2, ProjectedGradientDescentNumpy, CarliniL2Method
from absl import app, flags
from keras.utils import to_categorical
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TSUtils import saveimage
from datamodels.Cifar.ResNet import Resnet_dataset_and_model
from datamodels.Cifar.WideResNet import WideResnet_dataset_and_model
from datamodels.Cifar.VGG19Model import VGG_model_and_dataset
from datamodels.LoadImages import load_adv_test_data
from datamodels.mnist.LeNet import LeNet_dataset_and_model

tf.compat.v1.disable_eager_execution()

def art_attack_cifar(data_type='cifar10', modelname='ResNet', attack_type='fgsm', output_dir= "./"):

    seed_size = 5000

    if data_type == 'cifar10':
        number_of_classes = 10
        img_rows, img_cols, img_depth = 32, 32, 3
        if (modelname == 'ResNet'):
            model, _, _, _ = Resnet_dataset_and_model(train=False, prob_last_layer=False)
        else:
            model, _, _, _ = VGG_model_and_dataset(train=False)
        cwd = os.path.dirname(os.path.abspath(__file__))
        orig_img_path = '../datamodels/Cifar/data/images/cifar10/cifar10_seed'
        orig_img_path = os.path.join(cwd, orig_img_path)

        x_selected, y_selected = load_adv_test_data('cifar10', None, None,
                                                orig_img_path, 'cifarorig', seed_size, load_all_data=False)
        y_selected_cat = to_categorical(y_selected, number_of_classes)

    if data_type == 'cifar100':
        number_of_classes = 100
        img_rows, img_cols, img_depth = 32, 32, 3
        model, _, _, _ = WideResnet_dataset_and_model(train=False)
        cwd = os.path.dirname(os.path.abspath(__file__))
        orig_img_path = '../datamodels/Cifar/data/images/cifar100/cifar100_seed'
        orig_img_path = os.path.join(cwd, orig_img_path)

        x_selected, y_selected = load_adv_test_data('cifar100', None, None,
                                                orig_img_path, 'cifarorig', seed_size, load_all_data=False)
        y_selected_cat = to_categorical(y_selected, number_of_classes)

    if data_type == 'mnist':
        number_of_classes = 10
        img_rows, img_cols, img_depth = 28, 28, 1
        model, _, _, _ = LeNet_dataset_and_model("mnist", train=False, lenet_family=modelname)
        cwd = os.path.dirname(os.path.abspath(__file__))
        orig_img_path = '../datamodels/mnist/data/images/mnist_seed'
        orig_img_path = os.path.join(cwd, orig_img_path)

        x_selected, y_selected = load_adv_test_data('mnist', None, None,
                                                orig_img_path, 'mnistorig', seed_size, load_all_data=False)
        y_selected_cat = to_categorical(y_selected, number_of_classes)

    #Create the ART classifier
    classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    if attack_type == 'pgd':
        classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
        attack = ProjectedGradientDescentNumpy(estimator=classifier, norm=np.inf, eps=0.1, eps_step=0.02, max_iter=40, targeted=False)
        prefix = "PGD"
    if attack_type == 'cw':
        model_logits = tf.keras.Model(inputs=model.input, outputs=model.get_layer('logits').output)
        classifier = KerasClassifier(model=model_logits, clip_values=(0, 1), use_logits=True)
        attack = CarliniL2Method(classifier=classifier, learning_rate=0.01, binary_search_steps=10, max_iter=10,
                                  initial_const=0.01, batch_size=32, targeted=False)
        prefix="CW"
    if attack_type == 'fgsm':
        classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
        attack = FastGradientMethod(estimator=classifier, norm=np.inf, eps=0.1, batch_size=32, eps_step=0.1, targeted=False)
        prefix = "FGSM"

    predictions = classifier.predict(x_selected)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_selected_cat, axis=1)) / len(y_selected_cat)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Generate adversarial test examples
    x_test_adv = attack.generate(x=x_selected, y=y_selected_cat)

    # Evaluate the classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_selected_cat, axis=1)) / len(y_selected_cat)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


    cwd = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(cwd, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(seed_size):
        label = y_selected[i]
        filename = prefix + "_" + str(label)
        saveimage(x_test_adv[i], img_rows, img_cols, img_depth, filename, output_dir)


# Add main block to generate adversarial images when script is run directly
if __name__ == "__main__":
    # Example: generate MNIST adversarial images using FGSM and save to mnist_test directory
    art_attack_cifar(
        data_type='mnist',
        modelname='LeNet',
        attack_type='fgsm',
        output_dir='../datamodels/mnist/data/images/mnist_test/'
    )

