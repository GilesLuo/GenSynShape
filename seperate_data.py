# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 08:56:19 2021

@author: 40759
"""
import os
import json
import random


def seperate_data(obj_dir: str, object_name, output_dir=None):
    """
    :param output_dir:
    :param obj_dir: dir that stores the obj
    :param object_name: {"Chair", "Table", "Lamp", ...}
    """
    if output_dir is None:
        output_dir = obj_dir + '../'
    data = []
    for folder in os.listdir(obj_dir):
        # print(folder)
        data.append({'anno_id': '{}'.format(folder)})

    random_data = random.sample(data, len(data))

    train = []
    val = []
    test = []
    for file in random_data:
        # print(len(train),int(len(random_data)*7/10))
        if len(train) < int(len(random_data) * 7 / 10):
            train.append(file)
        elif len(val) < int(len(random_data) * 1 / 10):
            val.append(file)
        else:
            test.append(file)

    print("train set size: {}, \n val set size: {}, \n test set size: {}, \n".format(len(train), len(val), len(test)))

    with open('{}.train.json'.format(output_dir + object_name), 'w') as result_file:
        json.dump(train, result_file)

    with open('{}.val.json'.format(output_dir + object_name), 'w') as result_file:
        json.dump(val, result_file)

    with open('{}.test.json'.format(output_dir + object_name), 'w') as result_file:
        json.dump(test, result_file)
