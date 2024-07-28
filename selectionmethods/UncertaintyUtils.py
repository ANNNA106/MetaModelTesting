import numpy as np
import scipy
import six
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from joblib import dump, load
from datamodels.Cifar.ResNet import Resnet_dataset_and_model
from datamodels.Cifar.WideResNet import WideResnet_dataset_and_model
from datamodels.Cifar.VGG19Model import VGG_model_and_dataset
from datamodels.mnist.LeNet import *

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def prob_entropy_all(p):
  epsilon = 1e-12
  entropy_values = -np.sum(p * np.log2(p + epsilon), axis=1)
  return entropy_values

def margin_all(p):
  prediction_sorted = np.sort(p, axis=1)
  margin_values = prediction_sorted[:, -1] - prediction_sorted[:, -2]
  margin_values = [1-x for x in margin_values]
  margin_values = np.array(margin_values)

  return margin_values

def fit_log_reg(X_features, Y_output, sc_filename, log_filename):
  sc_x = StandardScaler()
  sc_x.fit(X_features)
  xtrain = sc_x.transform(X_features)
  cwd = os.path.dirname(os.path.abspath(__file__))
  filename_scaler = os.path.join(cwd, "./"+ sc_filename)
  dump(sc_x, filename_scaler, compress=True)
  log_reg = sm.Logit(Y_output, sm.add_constant(xtrain.astype(float))).fit()
  filename_logistic = os.path.join(cwd, "./"+ log_filename)
  log_reg.save(filename_logistic)
  return log_reg

def calculate_different_predictions(de_class_pred, tested_class_preds):
  item_count = np.shape(tested_class_preds)[0]
  de_disagree_count = np.zeros(item_count)
  tested_model_disagree_count = np.zeros(item_count)
  for item in range(item_count):
    de_disagree_count[item] = len(np.unique(de_class_pred[:, item]))
    for de_model_index in range(np.shape(de_class_pred)[0]):
      if not(tested_class_preds[item] == de_class_pred[de_model_index][item]):
        tested_model_disagree_count[item] = tested_model_disagree_count[item] + 1
  return de_disagree_count, tested_model_disagree_count


def calculate_log_parameters(metric, model_name, tested_model, model, weight_files, n_modelsamplingsize, filename_prefix):

  #First load data that will be used in Meta-Model (LR model) training
  if (model_name == "ResNet") or (model_name == "VGG"):
    cwd = os.path.dirname(os.path.abspath(__file__))
    orig_not_seed_img_path = '../datamodels/Cifar/data/images/cifar10/cifar10_test_metamodel'
    orig_not_seed_img_path = os.path.join(cwd, orig_not_seed_img_path)
    x_test, y_test = load_adv_test_data("cifar10", None, None, orig_not_seed_img_path, "cifarorig",
                                        5000, load_all_data=True)
  elif (model_name == "Lenet5") or (model_name == "Lenet1"):
    cwd = os.path.dirname(os.path.abspath(__file__))
    orig_not_seed_img_path = '../datamodels/mnist/data/images/mnist_test_metamodel'
    orig_not_seed_img_path = os.path.join(cwd, orig_not_seed_img_path)
    x_test, y_test = load_adv_test_data("mnist", None, None, orig_not_seed_img_path, "mnistorig",
                                        5000, load_all_data=True)
  elif (model_name == "WideResNet"):
    cwd = os.path.dirname(os.path.abspath(__file__))
    orig_not_seed_img_path = '../datamodels/Cifar/data/images/cifar100/cifar100_test_metamodel'
    orig_not_seed_img_path = os.path.join(cwd, orig_not_seed_img_path)
    x_test, y_test = load_adv_test_data("cifar100", None, None, orig_not_seed_img_path, "cifarorig",
                                        5000, load_all_data=True)

  # Calculate tested model's prediction probabilities
  tested_model_probs = tested_model.predict(x_test)
  tested_model_pred = np.argmax(tested_model_probs, axis=1)
  tested_model_max_probs = np.amax(tested_model_probs, axis=1)

  # Calculate prediction probabilities of DE Ensembles
  de_test_ensemble_logits = []
  for filename in weight_files:
    model.load_weights(filename)
    de_test_ensemble_logits.extend([model(x_test).logits for _ in range(n_modelsamplingsize)])
  de_test_ensemble_logits = np.array(de_test_ensemble_logits)
  de_test_model_probs = scipy.special.softmax(de_test_ensemble_logits, axis=2)
  de_test_model_pred = np.argmax(de_test_model_probs, axis=2)

  # Calculate DE Variation Score metric
  de_disagree_count, tested_model_disagree_count = calculate_different_predictions(de_test_model_pred, tested_model_pred)
  de_disagree_count = np.reshape(de_disagree_count, (np.shape(de_disagree_count)[0], 1))
  tested_model_disagree_count = np.reshape(tested_model_disagree_count, (np.shape(tested_model_disagree_count)[0], 1))

  #Train Meta-Models from uncertainty metrics and save the results
  if (metric == "Combined_Metric_1"):
    tested_model_max_probs = np.amax(tested_model_probs, axis=1)
    test_class_probs = np.reshape(tested_model_max_probs, (np.shape(tested_model_max_probs)[0], 1))
    test_class_probs = np.array([1.0 - x for x in test_class_probs])
    orig_test_scores = np.hstack((test_class_probs, tested_model_disagree_count))
    scale_model_filename = filename_prefix + "_Scaler_DE-Dis_Conf.bin"
    logistic_model_filename = filename_prefix + "_Log_DE-Dis_Conf.pickle"
  if (metric == "Combined_Metric_2"):
    prediction_sorted = np.sort(tested_model_probs, axis=1)
    margin_values = prediction_sorted[:, -1] - prediction_sorted[:, -2]
    margin_values = np.reshape(margin_values, (np.shape(margin_values)[0], 1))
    orig_test_scores = np.hstack((margin_values, tested_model_disagree_count))
    scale_model_filename = filename_prefix + "_Scaler_DE-Dis_Margin.bin"
    logistic_model_filename = filename_prefix + "_Log_DE-Dis_Margin.pickle"
  if (metric == "Combined_Metric_3"):
    entropy_values = prob_entropy_all(tested_model_probs)
    entropy_values = np.reshape(entropy_values, (np.shape(entropy_values)[0], 1))
    orig_test_scores = np.hstack((entropy_values, tested_model_disagree_count))
    scale_model_filename = filename_prefix + "_Scaler_DE-Dis_Entropy.bin"
    logistic_model_filename = filename_prefix + "_Log_DE-Dis_Entropy.pickle"

  misclassification_list = np.logical_not(np.equal(np.squeeze(y_test), np.squeeze(tested_model_pred)))

  #Train logistic regression models
  fit_log_reg(orig_test_scores, misclassification_list, sc_filename=scale_model_filename, log_filename=logistic_model_filename)


def get_combined_scores(metric, adv_x_test, adv_y_test, model,  model_name, bayesian_type,
                           n_modelsamplingsize=1, weight_files=None):
  calculate_log = False

  if (model_name == "ResNet"):
    tested_model, _, _, weight_filename = Resnet_dataset_and_model(train=False, prob_last_layer=False)
    filename_prefix = "./Cifar_" + model_name
  elif (model_name == "VGG"):
      tested_model, _, _, weight_filename = VGG_model_and_dataset(train=False)
      filename_prefix = "./Cifar_" + model_name
  elif (model_name == "Lenet5") or (model_name == "Lenet1"):
    tested_model, _, _, weight_filename = LeNet_dataset_and_model("mnist", train=False,
                                                                                lenet_family=model_name)
    filename_prefix = "./Mnist_" + model_name
  elif (model_name == "WideResNet") :
    tested_model, _, _, weight_filename = WideResnet_dataset_and_model(train=False, prob_last_layer=False)
    filename_prefix = "./Cifar100_" + model_name

  #Following lines added for calculate logistic regression model's parameters
  if calculate_log:
    calculate_log_parameters(metric, model_name, tested_model, model, weight_files, n_modelsamplingsize, filename_prefix)

  #Calculate prediction probabilities of DE model's for test data
  adv_de_ensemble_logits = []
  for filename in weight_files:
    model.load_weights(filename)
    model_logits = tf.keras.Model(inputs=model.input, outputs=model.get_layer('logits').output)
    logits = model_logits.predict(adv_x_test)
    adv_de_ensemble_logits.extend([logits])

  adv_de_ensemble_logits = np.array(adv_de_ensemble_logits)
  adv_de_class_probs = scipy.special.softmax(adv_de_ensemble_logits, axis=2)
  adv_de_class_pred = np.argmax(adv_de_class_probs, axis=2)

  #Calculate tested model probabilities
  adv_tested_model_probs = tested_model.predict(adv_x_test)
  adv_tested_model_pred = np.argmax(adv_tested_model_probs, axis=1)
  adv_tested_model_max_probs = np.amax(adv_tested_model_probs, axis=1)

  #Calculate disagreement count is ensemble models
  adv_de_disagree_count, adv_tested_model_disagree_count = calculate_different_predictions(adv_de_class_pred, adv_tested_model_pred)

  #Calculate DE Variation Score
  if (metric == "Prediction_Disagreement"):
    test_combined_scores = adv_tested_model_disagree_count

  #Reshape arrays
  adv_tested_model_disagree_count = np.reshape(adv_tested_model_disagree_count, (np.shape(adv_tested_model_disagree_count)[0], 1))
  adv_de_disagree_count = np.reshape(adv_de_disagree_count, (np.shape(adv_de_disagree_count)[0], 1))

  if (metric == "Combined_Metric_1"):
    #Calculate Least Confidence values
    adv_tested_model_max_probs = np.reshape(adv_tested_model_max_probs, (np.shape(adv_tested_model_max_probs)[0], 1))
    adv_tested_model_max_probs = np.array([1.0 - x for x in adv_tested_model_max_probs])

    #Combine Least Confidence with DE Variation Score
    test_scores = np.hstack((adv_tested_model_max_probs, adv_tested_model_disagree_count))
    scaler_model_filename = filename_prefix + "_Scaler_DE-Dis_Conf.bin"
    logistic_model_filename = filename_prefix + "_Log_DE-Dis_Conf.pickle"

  if (metric == "Combined_Metric_2"):
    # Calculate margin values
    adv_prediction_sorted = np.sort(adv_tested_model_probs, axis=1)
    adv_margin_values = adv_prediction_sorted[:, -1] - adv_prediction_sorted[:, -2]
    adv_margin_values = np.reshape(adv_margin_values, (np.shape(adv_margin_values)[0], 1))

    # Combine Margin with DE Variation Score
    test_scores = np.hstack((adv_margin_values, adv_tested_model_disagree_count))
    scaler_model_filename = filename_prefix + "_Scaler_DE-Dis_Margin.bin"
    logistic_model_filename = filename_prefix + "_Log_DE-Dis_Margin.pickle"
  if (metric == "Combined_Metric_3"):
    #Calculate Entropy values
    adv_entropy_values = prob_entropy_all(adv_tested_model_probs)
    adv_entropy_values = np.reshape(adv_entropy_values, (np.shape(adv_entropy_values)[0], 1))
    # Combine Entropy with DE Variation Score
    test_scores = np.hstack((adv_entropy_values, adv_tested_model_disagree_count))
    scaler_model_filename = filename_prefix + "_Scaler_DE-Dis_Entropy.bin"
    logistic_model_filename = filename_prefix + "_Log_DE-Dis_Entropy.pickle"

  if metric in ["Combined_Metric_1","Combined_Metric_2","Combined_Metric_3"]:
    #Load scaler model and scale metric values for log regression
    cwd = os.path.dirname(os.path.abspath(__file__))

    sc_filename = os.path.join(cwd, scaler_model_filename)
    sc_x = load(sc_filename)
    test_scores = sc_x.transform(test_scores)
    log_filename = os.path.join(cwd, logistic_model_filename)
    log_model = sm.load(log_filename)
    test_scores = sm.add_constant(test_scores.astype(float))
    test_combined_scores = log_model.predict(test_scores)

  return test_combined_scores

def query_samples(x,
                     y,
                     model,
                     n_modelsamplingsize=1,
                     weight_files=None,
                    metric = "Entropy"):
  if weight_files is None:
    metric_values = [model.evaluate(x, y, verbose=0) for _ in range(n_modelsamplingsize)]
    ensemble_logits = [model(x).logits for _ in range(n_modelsamplingsize)]
  else:
    metric_values = []
    ensemble_logits = []
    for filename in weight_files:
      model.load_weights(filename)
      model_logits = tf.keras.Model(inputs=model.input, outputs=model.get_layer('logits').output)

      if (n_modelsamplingsize>1):
        ensemble_logits.extend([model(x).logits for _ in range(n_modelsamplingsize)])
      else:
        logits= model_logits.predict(x)
        ensemble_logits.extend([logits])

      for _ in range(n_modelsamplingsize):
        results = model.evaluate(x, y, verbose=2, batch_size=100)
        metric_values.extend([results])

  ensemble_logits = np.array(ensemble_logits)
  softmax_values = scipy.special.softmax(ensemble_logits, axis=2)
  probabilistic = np.mean(softmax_values, axis=0)

  if metric == 'Entropy':
    pb_entropy_values = prob_entropy_all(probabilistic)
    results = pb_entropy_values
  elif metric == 'Margin':
    pb_margin_values = margin_all(probabilistic)
    results = pb_margin_values
  elif metric == 'Confidence':
    ensemble_confidence_values = np.amax(probabilistic, axis=1)
    ensemble_confidence_values = np.array([1.0 - x for x in ensemble_confidence_values])
    results = ensemble_confidence_values

  return results

def query_samples_combinedmetrics(x,
                    y,
                    model,
                    n_modelsamplingsize=1,
                    weight_files=None,
                    metric = "Mahalanobis",
                    model_name="ResNet",
                    bayesian_type="DeepEnsemble"):

  if metric in ["Combined_Metric_1", "Combined_Metric_2", "Combined_Metric_3", "Prediction_Disagreement"]:
    results = get_combined_scores(metric, x, y, model, model_name, bayesian_type,
                                         n_modelsamplingsize, weight_files)

  return results

class MeanMetricWrapper(tf.keras.metrics.Mean):
  def __init__(self, fn, name=None, dtype=None, **kwargs):
    super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs
  def update_state(self, y_true, y_pred, sample_weight=None):
    matches = self._fn(y_true, y_pred, **self._fn_kwargs)
    return super(MeanMetricWrapper, self).update_state(
        matches, sample_weight=sample_weight)
  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = tf.keras.backend.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(MeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

