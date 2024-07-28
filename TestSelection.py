from selectionmethods.uncertainty.Mnist.Mnist_DeepEnsemble import *
from selectionmethods.SelectionList import *
from selectionmethods import UncertaintyUtils
from datamodels.mnist.LeNet import *
from analysis import statistic

def select_samples(selection_method, X_adv, Y_adv, data_count, data_group=""):
    current_selected, metric_values = selection_method.selecttestdata(X_adv, Y_adv, data_count, data_group)
    return current_selected, metric_values

def select_samples_DE(selection_method, X_adv, Y_adv, data_count, ensemble_filenames, metric, sampling_count):
    current_selected, metric_values = selection_method.selecttestdata_DE(X_adv, Y_adv, data_count,
                                                                     ensemble_filenames, metric, sampling_count)
    return current_selected, metric_values


def select_test_data(selection_method_name, model_name, num_labels, input_shape,
                     adv_all_x, adv_all_y, adv_count, model,
                     ensemble_model=None, ensemble_filenames=None, data_group=""):
    if 'Deep_Ensemble' in selection_method_name:
        DE_size = 5
        metric = selection_method_name[14:]
        if ensemble_model == None:
            ensemble_model, _, _, ensemble_filenames = generate_dataset_and_model("mnist", ensemble_size=DE_size)
        method = DeepEnsemble
        selection_method = method(ensemble_model, input_shape, num_labels, model_name=model_name)
        new_selected_samples_index, metric_values = select_samples_DE(selection_method, adv_all_x, adv_all_y,
                                                                           adv_count,
                                                                           ensemble_filenames, metric, sampling_count=1)
        selected_samples_data = np.copy(adv_all_x[new_selected_samples_index])
    elif selection_method_name == 'Entropy':
        method = EntropySampling
        selection_method = method(model, input_shape, num_labels, model_name=model_name)
        new_selected_samples_index, metric_values = select_samples(selection_method, adv_all_x, adv_all_y,
                                                                     adv_count)
        selected_samples_data = np.copy(adv_all_x[new_selected_samples_index, :])
    elif selection_method_name == 'Confidence':
        method = ConfidenceSampling
        selection_method = method(model, input_shape, num_labels, model_name=model_name)
        new_selected_samples_index, metric_values = select_samples(selection_method, adv_all_x, adv_all_y,
                                                                     adv_count)
        selected_samples_data = np.copy(adv_all_x[new_selected_samples_index, :])
    elif selection_method_name == 'Margin':
        method = MarginSampling
        selection_method = method(model, input_shape, num_labels, model_name=model_name)
        new_selected_samples_index, metric_values = select_samples(selection_method, adv_all_x, adv_all_y,
                                                                 adv_count)
        selected_samples_data = np.copy(adv_all_x[new_selected_samples_index, :])
    elif selection_method_name == 'Random':
        method = RandomSampling
        selection_method = method(model, input_shape, num_labels, model_name=model_name)
        new_selected_samples_index, metric_values = select_samples(selection_method, adv_all_x, adv_all_y,
                                                             adv_count)
    elif selection_method_name == 'DeepGini':
        method = DeepGiniSampling
        selection_method = method(model, input_shape, num_labels, model_name=model_name)
        new_selected_samples_index, metric_values = select_samples(selection_method, adv_all_x, adv_all_y,
                                                             adv_count)
    elif selection_method_name == 'MCP':
        method = MCPSampling
        selection_method = method(model, input_shape, num_labels, model_name=model_name)
        new_selected_samples_index, metric_values = select_samples(selection_method, adv_all_x, adv_all_y,
                                                             adv_count)
    elif selection_method_name == 'DSA':
        method = DSASampling
        selection_method = method(model, input_shape, num_labels, model_name=model_name)
        new_selected_samples_index, metric_values = select_samples(selection_method, adv_all_x, adv_all_y,
                                                             adv_count, data_group)
    elif selection_method_name == 'LSA':
        method = LSASampling
        selection_method = method(model, input_shape, num_labels, model_name=model_name)
        new_selected_samples_index, metric_values = pool_samples(selection_method, adv_all_x, adv_all_y,
                                                             adv_count, data_group)

    selected_samples_data = np.copy(adv_all_x[new_selected_samples_index, :])
    selected_samples_label = np.copy(adv_all_y[new_selected_samples_index])

    return selected_samples_data, selected_samples_label, metric_values, new_selected_samples_index

def statistic_eval(tested_model, selected_samples_data, selected_samples_label, metric_values, total_faults):
    class_probs = tested_model.predict(selected_samples_data)
    class_pred = np.argmax(class_probs, axis=1)
    misclassification_list = np.logical_not(np.equal(selected_samples_label, class_pred))
    total_faults_in_testcases = np.sum(misclassification_list)
    test_case_count = len(selected_samples_data)
    if (total_faults >= test_case_count):
        total_faults = test_case_count
    apfd_value = statistic.apfd(misclassification_list, number_of_test_cases=test_case_count, number_of_faults=total_faults)

    fault_catched = np.where(class_pred == selected_samples_label, 0, 1)
    rauc_100 = statistic.RAUC(total_faults, fault_catched, 100)
    rauc_300 = statistic.RAUC(total_faults, fault_catched, 300)
    rauc_500 = statistic.RAUC(total_faults, fault_catched, 500)
    rauc_1000 = statistic.RAUC(total_faults, fault_catched, 1000)
    rauc_all = statistic.RAUC(total_faults, fault_catched, test_case_count)
    rauc = [rauc_100, rauc_300, rauc_500, rauc_1000, rauc_all]

    biserialscore, p_biserial = statistic.computeCor(misclassification_list, metric_values)
    fault_percentage = statistic.faultpercentage(misclassification_list, number_of_test_cases=test_case_count,
                                                 number_of_faults=total_faults)
    return rauc, apfd_value, biserialscore, p_biserial, fault_percentage, misclassification_list

