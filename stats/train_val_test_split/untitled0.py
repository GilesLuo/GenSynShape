# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 15:51:26 2021

@author: 40759
"""

import json
import numpy as np

f = open('Chair.train.json')
train = json.load(f)
chair_train = []
for each in train:
    chair_train.append(each['anno_id'])
np.save('Chair.train.npy',chair_train)

f = open('Chair.val.json')
val = json.load(f)
chair_val = []
for each in val:
    chair_val.append(each['anno_id'])
np.save('Chair.val.npy',chair_val)

f = open('Chair.test.json')
test = json.load(f)
chair_test = []
for each in test:
    chair_test.append(each['anno_id'])
np.save('Chair.test.npy',chair_test)