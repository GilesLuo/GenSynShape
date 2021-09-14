# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 08:56:19 2021

@author: 40759
"""
import os 
import json
import random

object_name = 'Chair'
json_str = None
data = []
for folder in os.listdir('chair'):
    # print(folder)
    data.append({'anno_id': '{}'.format(folder)})

random_data = random.sample(data,len(data))

train = []
val = []
test = []
for file in random_data:
    # print(len(train),int(len(random_data)*7/10))
    if len(train) < int(len(random_data)*7/10):
        train.append(file)
    elif len(val) < int(len(random_data)*1/10):
        val.append(file)
    else:
        test.append(file)

print(len(train),len(val),len(test))

with open('{}.train.json'.format(object_name), 'w') as result_file:
    json.dump(train, result_file)
    
with open('{}.val.json'.format(object_name), 'w') as result_file:
    json.dump(val, result_file)
    
with open('{}.test.json'.format(object_name), 'w') as result_file:
    json.dump(test, result_file)