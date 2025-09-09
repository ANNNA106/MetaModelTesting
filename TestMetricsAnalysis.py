import numpy
import numpy as np

from datamodels.mnist.LeNet import *
from datamodels.Cifar.ResNet import *
from datamodels.Cifar.WideResNet import WideResnet_dataset_and_model
from datamodels.Cifar.VGG19Model import VGG_model_and_dataset

from datamodels.LoadImages import load_adv_test_data
from TestSelection import select_test_data, statistic_eval
from tensorflow.keras.utils import to_categorical

import datetime
from selectionmethods.uncertainty.Mnist.Mnist_DeepEnsemble import *
from selectionmethods.uncertainty.Cifar.Cifar_DeepEnsemble_VGG import generate_VGG_CIFAR_dataset_and_model
from selectionmethods.uncertainty.Cifar.Cifar_DeepEnsemble import *
from selectionmethods.uncertainty.Cifar.Cifar_DE_WideResnet import generate_CIFAR100_dataset_and_model
from absl import app
from absl import flags


flags.DEFINE_enum('tested_dataset', 'mnist',
                  enum_values=['mnist', 'cifar', 'cifar100'],
                  help='Name of the image dataset.')

flags.DEFINE_enum('tested_model', 'Lenet5',
                  enum_values=['Lenet5', 'Lenet1', 'ResNet', 'VGG', 'WideResNet'],
                  help='Name of the models.')
flags.DEFINE_bool('OOD_data', False, 'Whether to use OOD data or not.')
flags.DEFINE_list('query_methods', ['Entropy', 'Confidence', 'Margin',
                     'Deep_Ensemble_Entropy', 'Deep_Ensemble_Confidence', 'Deep_Ensemble_Margin',
                    'Deep_Ensemble_Combined_Metric_1', 'Deep_Ensemble_Combined_Metric_2', 'Deep_Ensemble_Combined_Metric_3',
                     'DeepGini', 'MCP',  'DSA', 'LSA'], 'list of query methods')

flags.DEFINE_list('data_groups', ['Orig', 'GAN', 'Aug', 'Attack', 'All'], "tested data types")

FLAGS = flags.FLAGS


def print_statistics(x, y, rauc_results, apfd_values, biserial_results, fault_percentage_all_results, file_path, file_name):

    output_path_results = file_path
    output_directory_results = os.path.join(output_path_results, "")
    if not os.path.exists(output_directory_results):
        os.mkdir(output_directory_results)
    exp_results_file_prefix = output_directory_results + file_name
    ct = datetime.datetime.now()

    for data_group_index in range(len(x)):
        np.savetxt(exp_results_file_prefix + "fault_percentage_" + str(data_group_index+1) + "_" + str(ct) + ".csv",
                   fault_percentage_all_results[data_group_index], delimiter=',')
        np.savetxt(exp_results_file_prefix + "rauc_result_" + str(data_group_index+1) + "_" + str(ct) + ".csv",
                   rauc_results[data_group_index], delimiter=',')

    np.savetxt(exp_results_file_prefix + "APFD_" + str(ct) + ".csv", apfd_values, delimiter=',')
    np.savetxt(exp_results_file_prefix + "biserial_results_" + str(ct) + ".csv", biserial_results, delimiter=',')

def getfaultcounts(tested_model, transformed_data_group_list, transformed_label_group_list):
    total_faults_in_test_sets = np.zeros(5, int)
    index = 0
    for x_test in transformed_data_group_list:
        y_test = transformed_label_group_list[index]
        class_probs = tested_model.predict(x_test)
        class_pred = np.argmax(class_probs, axis=1)
        misclassification_list = np.logical_not(np.equal(class_pred, y_test))
        total_faults_in_test_sets[index] = int(np.sum(misclassification_list))
        index = index + 1

    return total_faults_in_test_sets


def main(argv):
    without_ood_data = (FLAGS.OOD_data == False)
    if (FLAGS.tested_dataset == 'mnist'):
        tested_model, x_test, y_cat_test, weight_filename = LeNet_dataset_and_model("mnist", train=True, lenet_family=FLAGS.tested_model)
        num_labels = 10
        input_shape = (28, 28, 1)
        adv_GAN_img_path = os.path.join('./datamodels/mnist/data/images/generated_inputs_GAN', "")
        adv_aug_img_path = os.path.join('./datamodels/mnist/data/images/generated_inputs_aug', "")
        if (FLAGS.tested_model == 'Lenet1'):
            adv_attack_img_path = os.path.join('./datamodels/mnist/data/images/generated_inputs_adv_lenet1', "")
        else:
            adv_attack_img_path = os.path.join('./datamodels/mnist/data/images/generated_inputs_adv_lenet5', "")
        orig_img_path = os.path.join('./datamodels/mnist/data/images/mnist_test', "")
        seed_img_path = os.path.join('./datamodels/mnist/data/images/mnist_seed', "")
    elif (FLAGS.tested_dataset == 'cifar'):
        if (FLAGS.tested_model == 'ResNet'):
            tested_model, x_test, y_cat_test, weight_filename = Resnet_dataset_and_model(train=False, prob_last_layer=False)
            adv_attack_img_path = os.path.join('./datamodels/Cifar/data/images/cifar10/generated_inputs_adv_Resnet', "")
        else:
            tested_model, x_test, y_cat_test, weight_filename = VGG_model_and_dataset(train=False)
            adv_attack_img_path = os.path.join('./datamodels/Cifar/data/images/cifar10/generated_inputs_adv_VGG', "")
        num_labels = 10
        input_shape = (32, 32, 3)
        adv_GAN_img_path = os.path.join('./datamodels/Cifar/data/images/cifar10/generated_inputs_gan', "")
        adv_aug_img_path = os.path.join('./datamodels/Cifar/data/images/cifar10/generated_inputs_aug', "")
        orig_img_path = os.path.join('./datamodels/Cifar/data/images/cifar10/cifar10_test', "")
        seed_img_path = os.path.join('./datamodels/Cifar/data/images/cifar10/cifar10_seed', "")
    elif (FLAGS.tested_dataset == 'cifar100'):
        if (FLAGS.tested_model == 'WideResNet'):
            tested_model, x_test, y_cat_test, weight_filename = WideResnet_dataset_and_model(train=False, prob_last_layer=False)
            adv_attack_img_path = os.path.join('./datamodels/Cifar/data/images/cifar100/generated_inputs_adv', "")
        num_labels = 100
        input_shape = (32, 32, 3)
        adv_GAN_img_path = os.path.join('./datamodels/Cifar/data/images/cifar100/generated_inputs_gan', "")
        adv_aug_img_path = os.path.join('./datamodels/Cifar/data/images/cifar100/generated_inputs_aug', "")
        orig_img_path = os.path.join('./datamodels/Cifar/data/images/cifar100/cifar100_test', "")
        seed_img_path = os.path.join('./datamodels/Cifar/data/images/cifar100/cifar100_seed', "")

    if without_ood_data:
        withoud_ood_file_postfix = "ID"
        adv_GAN_img_path = os.path.join(adv_GAN_img_path, withoud_ood_file_postfix)
        adv_aug_img_path = os.path.join(adv_aug_img_path, withoud_ood_file_postfix)
        adv_attack_img_path = os.path.join(adv_attack_img_path, withoud_ood_file_postfix)

    if (FLAGS.tested_dataset == 'mnist'):
        transformation_list = ['GAN', 'ACGAN', 'DCGAN', 'Rotate', 'Translate', 'Brightness', 'Blur', 'CW', 'FGSM','PGD']
        gan_transformation_list = ['GAN', 'ACGAN', 'DCGAN']
        aug_transformation_list = ['Rotate', 'Translate', 'Brightness', 'Blur']
        adv_transformation_list = ['CW', 'FGSM', 'PGD']
    elif (FLAGS.tested_dataset == 'cifar'):
        transformation_list = ['Rotate', 'Translate', 'Brightness', 'Blur', 'DPGAN', 'STYLE2ADA', 'CW', 'FGSM', 'PGD']
        gan_transformation_list = ['STYLE2ADA', 'DPGAN']
        aug_transformation_list = ['Rotate', 'Translate', 'Brightness', 'Blur']
        adv_transformation_list = ['CW', 'FGSM', 'PGD']
    elif (FLAGS.tested_dataset == 'cifar100'):
        transformation_list = ['Rotate', 'Translate', 'Brightness', 'Blur', 'CW', 'FGSM', 'PGD', 'GAN']
        gan_transformation_list = ['GAN']
        aug_transformation_list = ['Rotate', 'Translate', 'Brightness', 'Blur']
        adv_transformation_list = ['CW', 'FGSM', 'PGD']

    query_methods = FLAGS.query_methods
    data_groups = FLAGS.data_groups

    row_count = len(query_methods)
    column_count = len(data_groups)

    apfd_results = np.zeros((row_count, column_count))
    biserial_results = np.zeros((row_count, column_count))
    fault_percentage_results = np.zeros((column_count, row_count, 10))
    rauc_results = np.zeros((column_count, row_count, 5))

    adv_all_gan_x = None
    adv_all_gan_y = None
    adv_all_aug_x = None
    adv_all_aug_y = None
    adv_all_attack_x = None
    adv_all_attack_y = None
    adv_all_trans_x = None
    adv_all_trans_y = None
    orig_all_x = None
    orig_all_y =None
    adv_all_trans_count = 0

    transformed_data_group_list = []
    transformed_label_group_list = []

    if (FLAGS.tested_dataset == 'mnist'):
        ensemble_model, _, _, ensemble_filenames = generate_dataset_and_model("mnist", ensemble_size=5, lenet_family=FLAGS.tested_model)
    elif (FLAGS.tested_dataset == 'cifar'):
        if (FLAGS.tested_model == 'ResNet'):
            ensemble_model, _, _, ensemble_filenames = generate_CIFAR_dataset_and_model(ensemble_size=5, train=False)
        else:
            ensemble_model, _, _, ensemble_filenames = generate_VGG_CIFAR_dataset_and_model(ensemble_size=5, train=False, variational=True)
    elif (FLAGS.tested_dataset == 'cifar100'):
        if (FLAGS.tested_model == 'WideResNet'):
            ensemble_model, _, _, ensemble_filenames = generate_CIFAR100_dataset_and_model(ensemble_size=5, train=False)

    #Load orig test data
    if (FLAGS.tested_dataset == 'mnist'):
        orig_all_x, orig_all_y = load_adv_test_data(FLAGS.tested_dataset, None, None,
                                                    orig_img_path, 'mnistorig',  0, load_all_data=True)
        seed_all_x, seed_all_y = load_adv_test_data(FLAGS.tested_dataset, None, None,
                                                    seed_img_path, 'mnistorig', 0, load_all_data=True)
    elif (FLAGS.tested_dataset == 'cifar') or (FLAGS.tested_dataset == 'cifar100'):
        orig_all_x, orig_all_y = load_adv_test_data(FLAGS.tested_dataset, None, None,
                                                orig_img_path, 'cifarorig', 0, load_all_data=True)
        seed_all_x, seed_all_y = load_adv_test_data(FLAGS.tested_dataset, None, None,
                                                seed_img_path, 'cifarorig', 0, load_all_data=True)


    #Load augmented test data
    for transformation in transformation_list:
        print("Transformation type : ", transformation)
        if transformation in gan_transformation_list:
            adv_all_gan_x, adv_all_gan_y = load_adv_test_data(FLAGS.tested_dataset, adv_all_gan_x,
                                                              adv_all_gan_y, adv_GAN_img_path,
                                                              transformation, 0, load_all_data=True)
        elif transformation in aug_transformation_list:
            adv_all_aug_x, adv_all_aug_y = load_adv_test_data(FLAGS.tested_dataset, adv_all_aug_x,
                                                              adv_all_aug_y, adv_aug_img_path,
                                                              transformation, 0, load_all_data=True)
        elif transformation in adv_transformation_list:
            adv_all_attack_x, adv_all_attack_y = load_adv_test_data(FLAGS.tested_dataset, adv_all_attack_x,
                                                                    adv_all_attack_y, adv_attack_img_path,
                                                                    transformation, 0, load_all_data=True)

    # Construct a data group inlcuding all generated images and original test images (seperated from meta-model training data)
    adv_all_trans_x = np.copy(orig_all_x)
    adv_all_trans_y = np.copy(orig_all_y)
    adv_all_trans_count = adv_all_trans_count + len(orig_all_x)
    if (adv_all_gan_x is not None):
        adv_all_trans_x = np.append(adv_all_trans_x, adv_all_gan_x, axis=0)
        adv_all_trans_y = np.append(adv_all_trans_y, adv_all_gan_y, axis=0)
        adv_all_trans_count = adv_all_trans_count + len(adv_all_gan_y)
    if (adv_all_aug_x is not None):
        adv_all_trans_x = np.append(adv_all_trans_x, adv_all_aug_x, axis=0)
        adv_all_trans_y = np.append(adv_all_trans_y, adv_all_aug_y, axis=0)
        adv_all_trans_count = adv_all_trans_count + len(adv_all_aug_y)
    if (adv_all_attack_x is not None):
        adv_all_trans_x = np.append(adv_all_trans_x, adv_all_attack_x, axis=0)
        adv_all_trans_y = np.append(adv_all_trans_y, adv_all_attack_y, axis=0)
        adv_all_trans_count = adv_all_trans_count + len(adv_all_attack_y)

    # Construct data groups Adversarial Attacks, Image Transformations and Generative Models
    adv_all_gan_x = np.append(adv_all_gan_x, seed_all_x, axis=0)
    adv_all_gan_y = np.append(adv_all_gan_y, seed_all_y, axis=0)
    adv_all_aug_x = np.append(adv_all_aug_x, seed_all_x, axis=0)
    adv_all_aug_y = np.append(adv_all_aug_y, seed_all_y, axis=0)
    adv_all_attack_x = np.append(adv_all_attack_x, seed_all_x, axis=0)
    adv_all_attack_y = np.append(adv_all_attack_y, seed_all_y, axis=0)

    for data_group in data_groups:
        if data_group == "Orig":
            transformed_data_group_list.append(orig_all_x)
            transformed_label_group_list.append(orig_all_y)
        if data_group == "GAN":
            transformed_data_group_list.append(adv_all_gan_x)
            transformed_label_group_list.append(adv_all_gan_y)
        if data_group == "Aug":
            transformed_data_group_list.append(adv_all_aug_x)
            transformed_label_group_list.append(adv_all_aug_y)
        if data_group == "Attack":
            transformed_data_group_list.append(adv_all_attack_x)
            transformed_label_group_list.append(adv_all_attack_y)
        if data_group == "All":
            transformed_data_group_list.append(adv_all_trans_x)
            transformed_label_group_list.append(adv_all_trans_y)

    fault_count_in_datasets = getfaultcounts(tested_model, transformed_data_group_list, transformed_label_group_list)

    #Perform test case prioritization for data groups
    column_index = 0
    for gen_x in transformed_data_group_list:
        if not (gen_x is None):
            row_index = 0
            gen_y = transformed_label_group_list[column_index]

            #Apply selection methods to data groups
            for query_method_name in query_methods:
                #Select all data according to selection method
                all_adv_count = len(gen_x)
                selected_samples_data, selected_samples_label,  metric_values, new_selected_samples_index = select_test_data(
                                        query_method_name, FLAGS.tested_model, num_labels, input_shape,
                                        gen_x, gen_y, all_adv_count, tested_model,
                                        ensemble_model, ensemble_filenames, data_groups[column_index])

                rauc_value, \
                apfd_value, \
                biserial, p_biserial, \
                fault_percentage, misclassification_list = statistic_eval(tested_model, selected_samples_data,
                                                                          selected_samples_label, metric_values,
                                                                          fault_count_in_datasets[column_index])

                apfd_results[row_index][column_index] = apfd_value
                biserial_results[row_index][column_index] = biserial
                fault_percentage_results[column_index][row_index] = fault_percentage
                rauc_results[column_index][row_index] = rauc_value

                row_index = row_index + 1
        column_index = column_index + 1

    x = data_groups
    y = query_methods
    file_path = './exp_results/'
    if FLAGS.OOD_data:
        file_name = FLAGS.tested_dataset + "_" + FLAGS.tested_model + "_"
    else:
        file_name = FLAGS.tested_dataset + "_withoutOOD_" + FLAGS.tested_model + "_"

    print_statistics(x, y, rauc_results, apfd_results, biserial_results,
                     fault_percentage_results, file_path=file_path, file_name=file_name)


if __name__ == '__main__':
    app.run(main)
