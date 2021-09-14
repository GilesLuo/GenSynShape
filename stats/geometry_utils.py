import numpy as np
import os
import json

def load(file):
    with open(file,'r') as f:
        data = json.load(f)
        return data

chair_train = load('train_val_test_split/Chair.train.json')
print(len(chair_train))

chair_valid = load('train_val_test_split/Chair.val.json')
print(len(chair_valid))

chair_test = load('train_val_test_split/Chair.test.json')
print(len(chair_test))