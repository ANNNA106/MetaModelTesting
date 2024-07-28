import numpy as np
from scipy.stats import pointbiserialr

def apfd(misclassification_list, number_of_test_cases, number_of_faults):
    sorted_list = np.arange(1, len(misclassification_list)+1)
    fault_index = 0
    test_case_index = 0
    sum_all = 0
    while (fault_index<number_of_faults):
        if (misclassification_list[test_case_index]==1):
            fault_index = fault_index + 1
            sum_all = sum_all + test_case_index + 1
        test_case_index = test_case_index + 1

    n = number_of_test_cases
    m = number_of_faults
    return 1-(float(sum_all)/(n*m))+(1./(2*n))


def faultpercentage(misclassification_list, number_of_test_cases, number_of_faults):
    number_of_tests_in_ten_percent = int(number_of_test_cases / 10)
    results = np.zeros(10)
    fault_count = 0
    for test_case_percentage in range(10):
        for index in range(number_of_tests_in_ten_percent):
            if (misclassification_list[index+(test_case_percentage*number_of_tests_in_ten_percent)] == 1):
                fault_count = fault_count + 1
        results[test_case_percentage] = float(fault_count / number_of_faults)
    return results


def computeCor(x,y):

    x = x[np.argwhere(np.logical_not(np.isnan(y)))]
    y = y[np.argwhere(np.logical_not(np.isnan(y)))]
    x = np.squeeze(x)
    y = np.squeeze(y)
    (biserialscore, p_biserial) = pointbiserialr(y, x)

    return biserialscore, p_biserial

def RAUC(number_of_faults, real_data_all, test_num):
    ideal_data = np.zeros(test_num)
    if (test_num>=number_of_faults):
        ideal_data[:number_of_faults] = 1
    else:
        ideal_data[:test_num] = 1

    ideal_area = curve_area(ideal_data)
    real_data = real_data_all[:test_num]
    real_area = curve_area(real_data)

    return real_area/ideal_area


def curve(data):
    data = np.array(data)
    for i in range(len(data)):
        idx = len(data) - i - 1
        data[idx] = np.sum(data[:idx + 1])
    return data


def curve_area(data):
    data = curve(data)
    return np.sum(data)
