#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:17 2019

@author: qq
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras import optimizers
import random
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import csv
from keras.utils import to_categorical
from keras.models import load_model
import pandas as pd
import argparse

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")
x_train = x_train / 255.0


def retrain(x_target, y_test, origin_acc, model, args, layer_names, selectsize=100, attack='fgsm', measure='lsa',
            datatype='mnist'):
    target_lst = []

    if measure == 'SRS':
        x_select, y_select = select_rondom(selectsize, x_target, x_target, y_test)
    if measure == 'MCP':
        x_select, y_select = select_my_optimize(model, selectsize, x_target, y_test)
    if measure == 'LSA':
        target_lst = fetch_lsa(model, x_train, x_target, attack, layer_names, args)
    if measure == 'DSA':
        target_lst = fetch_dsa(model, x_train, x_target, attack, layer_names, args)

    if measure == 'AAL':
        path = "./cifar_finalResults/cifar_" + attack + "_compound8_result.csv"
        csv_data = pd.read_csv(path, header=None)
        target_lst = []
        for i in range(len(csv_data.values.T)):
            target_lst.append(csv_data.values.T[i])
    if measure == 'CES':
        tmpfile = "./conditional/" + attack + "_cifar_" + str(selectsize) + ".npy"
        if os.path.exists(tmpfile):
            indexlst = list(np.load(tmpfile))
        else:
            indexlst = condition.conditional_sample(model, x_target, selectsize)
            np.save(tmpfile, np.array(indexlst))
        x_select, y_select = select_from_index(selectsize, x_target, indexlst, y_test)
    elif measure not in ['SRS', 'MCP']:
        x_select, y_select = select_from_large(selectsize, x_target, target_lst, y_test)

    y_select = to_categorical(y_select, 10)
    y_test = to_categorical(y_test, 10)

    model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])

    retrain_acc = 0

    model.fit(x_select, y_select, batch_size=100, epochs=5, shuffle=True, verbose=1, validation_data=(x_target, y_test))
    score = model.evaluate(x_target, y_test, verbose=0)
    retrain_acc = score[1]

    return retrain_acc


def find_second(act, ncl=10):
    max_ = 0
    second_max = 0
    sec_index = 0
    max_index = 0
    for i in range(ncl):
        if act[i] > max_:
            max_ = act[i]
            max_index = i

    for i in range(ncl):
        if i == max_index:
            continue
        if act[i] > second_max:
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_
    return max_index, sec_index, ratio


def select_my_optimize(model, selectsize, x_target, y_test):
    x = np.zeros((selectsize, 32, 32, 3))
    y = np.zeros((selectsize, 1))

    act_layers = model.predict(x_target)
    dicratio = [[] for i in range(100)]
    dicindex = [[] for i in range(100)]
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act)  # max_index
        dicratio[max_index * 10 + sec_index].append(ratio)
        dicindex[max_index * 10 + sec_index].append(i)

    selected_lst = select_from_firstsec_dic(selectsize, dicratio, dicindex)
    for i in range(selectsize):
        x[i] = x_target[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]

    return x, y


def select_from_firstsec_dic(selectsize, dicratio, dicindex, ncl=10):
    selected_lst = []
    selected_lst_values = []
    tmpsize = selectsize

    noempty = no_empty_number(dicratio)
    window_size = int(ncl * ncl)
    while (selectsize >= noempty) and (noempty > 0):
        for i in range(window_size):
            if len(dicratio[i]) != 0:
                tmp = max(dicratio[i])
                j = dicratio[i].index(tmp)
                selected_lst.append(dicindex[i][j])
                selected_lst_values.append(tmp)
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize = tmpsize - len(selected_lst)
        noempty = no_empty_number(dicratio)

    while len(selected_lst) != tmpsize:
        max_tmp = [0 for i in range(selectsize)]
        max_index_tmp = [0 for i in range(selectsize)]
        max_index_tmp_values = [0 for i in range(selectsize)]
        for i in range(window_size):
            if len(dicratio[i]) != 0:
                tmp_max = max(dicratio[i])
                if tmp_max > min(max_tmp):
                    index = max_tmp.index(min(max_tmp))
                    max_tmp[index] = tmp_max
                    max_index_tmp[index] = dicindex[i][dicratio[i].index(tmp_max)]
                    max_index_tmp_values[index] = tmp_max
        if len(max_index_tmp) == 0 and len(selected_lst) != tmpsize:
            print('wrong!!!!!!')
            break
        selected_lst = selected_lst + max_index_tmp
        selected_lst_values = selected_lst_values + max_index_tmp_values
    assert len(selected_lst) == tmpsize
    return selected_lst, selected_lst_values


def no_empty_number(dicratio):
    no_empty = 0
    for i in range(len(dicratio)):
        if len(dicratio[i]) != 0:
            no_empty += 1
    return no_empty


def select_from_large(select_amount, x_target, target_lsa, y_test):
    x = np.zeros((select_amount, 32, 32, 3))
    y = np.zeros((select_amount, 1))

    selected_lst, lsa_lst = order_output(target_lsa, select_amount)
    for i in range(select_amount):
        x[i] = x_target[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
    return x, y


def select_rondom(select_amount, x_target, target_lsa, y_test):
    x = np.zeros((select_amount, 32, 32, 3))
    y = np.zeros((select_amount, 1))

    selected_lst = np.random.choice(range(len(target_lsa)), replace=False, size=select_amount)
    # selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    for i in range(select_amount):
        x[i] = x_target[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
    return x, y


def select_from_index(select_amount, x_target, indexlst, y_test):
    x = np.zeros((select_amount, 32, 32, 3))
    y = np.zeros((select_amount, 1))
    # print(indexlst)
    for i in range(select_amount):
        x[i] = x_target[indexlst[i]]
        y[i] = y_test[indexlst[i]]
    return x, y


def find_index(target_lsa, selected_lst, max_lsa):
    for i in range(len(target_lsa)):
        if max_lsa == target_lsa[i] and i not in selected_lst:
            return i
    return 0


def order_output(target_lsa, select_amount):
    lsa_lst = []

    tmp_lsa_lst = target_lsa[:]
    selected_lst = []
    while len(selected_lst) < select_amount:
        max_lsa = max(tmp_lsa_lst)
        selected_lst.append(find_index(target_lsa, selected_lst, max_lsa))
        lsa_lst.append(max_lsa)
        tmp_lsa_lst.remove(max_lsa)
    return selected_lst, lsa_lst


def createdataset(attack, ratio=8):
    if attack in ['rotation', 'translation', 'shear', 'brightness', 'contrast', 'scale']:
        x_target = np.load('./imagetrans/cifar_' + attack + '.npy')
    else:
        x_target = np.load('./adv/data/cifar/Adv_cifar_' + attack + '.npy')
    if attack in ['rotation', 'translation', 'shear', 'brightness', 'contrast', 'scale']:
        x_target = x_target.astype("float32")
        x_target = (x_target / 255.0)

    model_path = './model/densenet_cifar10.h5df'
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_test = x_test.astype("float32")
    x_test = x_test / 255.0

    origin_lst = np.random.choice(range(10000), replace=False, size=ratio * 1000)
    mutated_lst = np.random.choice(range(10000), replace=False, size=10000 - ratio * 1000)

    x_dest = np.append(x_test[origin_lst], x_target[mutated_lst], axis=0)
    y_dest = np.append(y_test[origin_lst], y_test[mutated_lst])
    np.savez('./adv/data/cifar/cifar_' + attack + '_compound9.npz', x_test=x_dest, y_test=y_dest)

    y_dest = to_categorical(y_dest, 10)
    model = load_model(model_path)
    score = model.evaluate(x_dest, y_dest, verbose=0)
    # print('Test Loss: %.4f' % score[0])
    print('Before retrain, Test accuracy: %.4f' % score[1])
    return

def select_only(model, selectsize, x_target, ncl):
    act_layers = model.predict(x_target)
    dicratio = [[] for i in range(ncl*ncl)]
    dicindex = [[] for i in range(ncl*ncl)]
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act, ncl)  # max_index
        dicratio[max_index * ncl + sec_index].append(ratio)
        dicindex[max_index * ncl + sec_index].append(i)


    selected_lst,  selected_lst_values = select_from_firstsec_dic(selectsize, dicratio, dicindex, ncl)
    selected_idx = []
    selected_idx_values = []
    for i in range(selectsize):
        selected_idx.append(selected_lst[i])
        selected_idx_values.append(selected_lst_values[i])

    return selected_idx, selected_idx_values
