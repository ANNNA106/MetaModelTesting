import os
import numpy as np
import ml_collections
from selectionmethods import UncertaintyUtils
from selectionmethods.LoadDatasets import loadCifarDataSet, loadMnistDataSet
from selectionmethods.MCP import select_only
from selectionmethods.SurpriseAdequacy import fetch_dsa, fetch_lsa


class SelectionMethod:

    def __init__(self, model, input_shape=(28,28), num_labels=10, model_name="Lenet1"):
        self.model = model
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_labels = num_labels

    def selecttestdata(self, X_test, Y_test, datacount, data_group=""):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param X_test: the test data
        :param Y_test: the labels of test data
        :param datacount: the count of test data to be selected
        :return: the new labeled indices (including the ones queried)
        """
        return NotImplemented

    def selecttestdata_DE(self, X_test, Y_test, data_count,ensemble_filenames, metric, sampling_count):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param X_test: the test data
        :param Y_test: the labels of test data
        :param datacount: the count of test data to be selected
        :return: the new labeled indices (including the ones queried)
        :ensemble_filenames: names of DE models
        :metric: uncertainty metric used with DE models
        :sampling_count: forward pass count
        """
        return NotImplemented

class RandomSampling(SelectionMethod):

    def __init__(self, model, input_shape, num_labels, model_name):
        super().__init__(model, input_shape, num_labels, model_name)

    def selecttestdata(self, X_test, Y_test, data_count, data_group=""):
        label_indices = np.arange(X_test.shape[0])
        metric_values = np.full(len(label_indices), 0.1, dtype=float)
        selected_index = np.random.choice(label_indices, data_count, replace=False)
        return selected_index, metric_values[selected_index]

class DeepEnsemble(SelectionMethod):

    def __init__(self, model, input_shape, num_labels, model_name):
        super().__init__(model, input_shape, num_labels, model_name)

    def selecttestdata_DE(self, X_test, Y_test, data_count,ensemble_filenames, metric, sampling_count):

        if metric in ["Combined_Metric_1", "Combined_Metric_2", "Combined_Metric_3", "Prediction_Disagreement"]:
            metric_results = UncertaintyUtils.query_samples_combinedmetrics(X_test, Y_test, self.model,
                                                                            sampling_count, ensemble_filenames,
                                                                            metric, self.model_name,
                                                                            "DeepEnsemble")
            sorted_indices = np.argsort(metric_results)
            new_selected_indices = np.transpose(sorted_indices)[-data_count:]
            new_selected_indices = np.flip(new_selected_indices)
        else:
            metric_results = UncertaintyUtils.query_samples(X_test, Y_test, self.model, sampling_count,
                                                            ensemble_filenames, metric)
            if metric in ["Entropy", "Margin", "Confidence"]:
                sorted_indices = np.argsort(metric_results)
                new_selected_indices = np.transpose(sorted_indices)[-data_count:]
                new_selected_indices = np.flip(new_selected_indices)

        return new_selected_indices, metric_results[new_selected_indices]

class DeepGiniSampling(SelectionMethod):

    def __init__(self, model, input_shape, num_labels, model_name):
        super().__init__(model, input_shape, num_labels, model_name)

    def selecttestdata(self, X_test, Y_test, data_count, data_group=""):

        predictions = self.model.predict(X_test)
        unlabeled_deepgini_values = np.sum(predictions**2, axis=1)

        new_selected_indices = np.argsort(unlabeled_deepgini_values)[:data_count]
        unlabeled_deepgini_values = np.array([1.0-x for x in unlabeled_deepgini_values])
        return new_selected_indices, unlabeled_deepgini_values[new_selected_indices]

class MCPSampling(SelectionMethod):

    def __init__(self, model, input_shape, num_labels, model_name):
        super().__init__(model, input_shape, num_labels, model_name)

    def selecttestdata(self, X_test, Y_test, data_count, data_group=""):
        if self.model_name in [ "Lenet5", "Lenet1", "ResNet", "VGG"]:
            num_classes = 10
        else:
            num_classes = 100

        new_selected_indices, new_selected_mcp_values = select_only(self.model, data_count, X_test, num_classes)

        return np.array(new_selected_indices), np.array(new_selected_mcp_values)

class DSASampling(SelectionMethod):

    def __init__(self, model, input_shape, num_labels, model_name):
        super().__init__(model, input_shape, num_labels, model_name)

    def selecttestdata(self, X_test, Y_test, data_count, data_group):
        args = ml_collections.ConfigDict()
        cwd = os.path.dirname(os.path.abspath(__file__))
        args.save_path = os.path.join(cwd, "for_dsa_VE")
        args.is_classification = True
        args.var_threshold = 0.00001
        args.model_type = self.model_name

        if (self.model_name == "ResNet") or (self.model_name == "VGG"):
            x_train, y_train, x_orig_test, y_orig_test = loadCifarDataSet('cifar10')
            args.layer_names = ['flatten']
            args.d = "cifar10"+"_"+self.model_name
            args.num_classes = 10
        elif (self.model_name == "Lenet5") or (self.model_name == "Lenet1"):
            x_train, y_train, x_orig_test, y_orig_test = loadMnistDataSet()
            x_train = x_train.reshape(x_train.shape[0],
                                      x_train.shape[1],
                                      x_train.shape[2],
                                      1)
            args.d = "mnist"+"_"+self.model_name
            args.num_classes = 10
            if (self.model_name == "Lenet5"):
                args.layer_names = ['dense']
            else:
                args.layer_names = ['flatten']
        elif (self.model_name == "WideResNet"):
            x_train, y_train, x_orig_test, y_orig_test = loadCifarDataSet('cifar100')
            args.layer_names = ['flatten']
            args.d = "cifar100" + "_" + self.model_name
            args.num_classes = 100

        args.data_type = data_group

        dsa_values = fetch_dsa(self.model, x_train, X_test, args.data_type, args.layer_names, args)
        dsa_values = np.array(dsa_values)
        sorted_indices = np.argsort(dsa_values)
        new_selected_indices = np.transpose(sorted_indices)[-data_count:]
        new_selected_indices = np.flip(new_selected_indices)

        return new_selected_indices, dsa_values[new_selected_indices]

class LSASampling(SelectionMethod):

    def __init__(self, model, input_shape, num_labels, model_name):
        super().__init__(model, input_shape, num_labels, model_name)

    def selecttestdata(self, X_test, Y_test, data_count, data_group):
        args = ml_collections.ConfigDict()
        cwd = os.path.dirname(os.path.abspath(__file__))
        args.save_path = os.path.join(cwd, "for_lsa_VE")
        args.is_classification = True
        args.var_threshold = 0.0001
        args.model_type = self.model_name

        if (self.model_name == "ResNet") or (self.model_name == "VGG"):
            x_train, y_train, x_orig_test, y_orig_test = loadCifarDataSet('cifar10')
            args.layer_names = ['flatten']
            args.d = "cifar10"+"_"+self.model_name
            args.num_classes = 10
        elif (self.model_name == "Lenet5") or (self.model_name == "Lenet1"):
            x_train, y_train, x_orig_test, y_orig_test = loadMnistDataSet()
            x_train = x_train.reshape(x_train.shape[0],
                                      x_train.shape[1],
                                      x_train.shape[2],
                                      1)
            args.d = "mnist"+"_"+self.model_name
            args.num_classes = 10
            if (self.model_name == "Lenet5"):
                args.layer_names = ['dense']
            else:
                args.layer_names = ['flatten']
        elif (self.model_name == "WideResNet"):
            x_train, y_train, x_orig_test, y_orig_test = loadCifarDataSet('cifar100')
            args.layer_names = ['flatten']
            args.d = "cifar100" + "_" + self.model_name
            args.num_classes = 100

        args.data_type = data_group

        lsa_values = fetch_lsa(self.model, x_train, X_test, args.data_type, args.layer_names, args)
        lsa_values = np.array(lsa_values)
        sorted_indices = np.argsort(lsa_values)
        new_selected_indices = np.transpose(sorted_indices)[-data_count:]
        new_selected_indices = np.flip(new_selected_indices)

        return new_selected_indices, lsa_values[new_selected_indices]

class EntropySampling(SelectionMethod):

    def __init__(self, model, input_shape, num_labels, model_name):
        super().__init__(model, input_shape, num_labels, model_name)

    def selecttestdata(self, X_test, Y_test, data_count, data_group=""):

        predictions = self.model.predict(X_test)
        unlabeled_predictions = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)

        sorted_indices = np.argsort(unlabeled_predictions)
        print("data count")
        print(data_count)
        new_selected_indices = np.transpose(sorted_indices)[-data_count:]
        new_selected_indices = np.flip(new_selected_indices)

        return new_selected_indices, unlabeled_predictions[new_selected_indices]

class ConfidenceSampling(SelectionMethod):
    def __init__(self, model, input_shape, num_labels, model_name):
        super().__init__(model, input_shape, num_labels, model_name)

    def selecttestdata(self, X_test, Y_test, data_count, data_group=""):

        predictions = self.model.predict(X_test)
        unlabeled_predictions = np.amax(predictions, axis=1)

        new_selected_indices = np.argsort(unlabeled_predictions)[:data_count]
        unlabeled_predictions = np.array([1.0-x for x in unlabeled_predictions])
        return new_selected_indices, unlabeled_predictions[new_selected_indices]

class MarginSampling(SelectionMethod):
    def __init__(self, model, input_shape, num_labels, model_name):
        super().__init__(model, input_shape, num_labels, model_name)

    def selecttestdata(self, X_test, Y_test, data_count, data_group=""):

        predictions = self.model.predict(X_test)

        prediction_sorted = np.sort(predictions, axis=1)
        margin_values = prediction_sorted[:, -1] - prediction_sorted[:, -2]
        margin_values = [1 - x for x in margin_values]
        margin_values = np.array(margin_values)

        sorted_indices = np.argsort(margin_values)
        new_selected_indices = np.transpose(sorted_indices)[-data_count:]
        new_selected_indices = np.flip(new_selected_indices)

        return new_selected_indices, margin_values[new_selected_indices]


