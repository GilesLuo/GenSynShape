import argparse
import os
import sys
from progressbar import ProgressBar
import random

from chair_grammar import *

from chair_back import *
from chair_seat import *
from chair_legs import *
from chair_armrests import *
from chair_stretchers import *

# out_dir = sys.argv[1]
# num_to_gen = int(sys.argv[2])
out_dir = 'gensyn/chair/'


def get_random(x, y):
    return random.random() * (y - x) + x


def get_random_bool(thres):
    return random.random() < thres


class Record(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        s = 'Namespace('
        for k in self.__dict__:
            s += str(k) + ': ' + str(self.__dict__[k]) + ', '
        s += ')'
        return s


def generate_chair(legWidth, legHeight, seatWidth, seatDepth,
                   seatHeight, backHeight, backDepth, counter):
    args = dict()
    args['out_dir'] = out_dir

    # Fix a set of lobal parameters
    args['legWidth'] = legWidth
    args['legHeight'] = legHeight
    args['seatWidth'] = seatWidth
    args['seatDepth'] = seatDepth
    args['seatHeight'] = seatHeight
    args['backHeight'] = backHeight
    args['backWidth'] = get_random(args['seatWidth'] - 0.1, args['seatWidth'])
    args['backDepth'] = backDepth

    # Make 4 variants for chair back
    back_args_set = []
    back_args_set.append(dict({'back_type': 'simple'}))
    back_args_set.append(dict({'hasBackTopBar'     : True, 'hasBackSideBar': True, 'back_type': 'H',
                               'backTopBarHeight'  : args['backHeight'] / 5,
                               'backSideBarWidth'  : args['backWidth'] / 10,
                               'backNumSurfaceBars': 5, 'backSurfaceBarWidth': 0.01, 'backSurfaceBarDepth': 0.01}))
    back_args_set.append(dict({'hasBackTopBar'     : False, 'hasBackSideBar': True, 'back_type': 'H',
                               'backTopBarHeight'  : args['backHeight'] / 5,
                               'backSideBarWidth'  : args['backWidth'] / 10,
                               'backNumSurfaceBars': 3, 'backSurfaceBarWidth': 0.02, 'backSurfaceBarDepth': 0.02}))
    back_args_set.append(dict({'hasBackTopBar'     : True, 'hasBackSideBar': False, 'back_type': 'T',
                               'backTopBarHeight'  : args['backHeight'] / 10,
                               'backSideBarWidth'  : args['backWidth'] / 10,
                               'backNumSurfaceBars': 4, 'backSurfaceBarWidth': 0.01, 'backSurfaceBarDepth': 0.01}))

    # Make 2 variants for chair legs
    leg_args_set = []
    leg_args_set.append(dict({'mode': 0}))
    leg_args_set.append(dict({'mode': 0.7}))

    # Make 4 variants for chair stretchers
    stretcher_args_set = []
    stretcher_args_set.append(dict({'hasStretchers': False}))
    stretcher_args_set.append(dict({'hasStretchers': True, 'stretcher_type': 'O'}))
    stretcher_args_set.append(dict({'hasStretchers': True, 'stretcher_type': 'H'}))
    stretcher_args_set.append(dict({'hasStretchers': True, 'stretcher_type': 'X'}))

    # Make 3 variants for chair arms
    arm_args_set = []
    arm_args_set.append(dict({'hasArmrests': False}))
    arm_args_set.append(dict({'hasArmrests': True, 'arm_type': 'T'}))
    arm_args_set.append(dict({'hasArmrests': True, 'arm_type': '7'}))

    # Make 4 x 2 x 4 x 3 = 96 variants for each chair
    for i1 in range(4):
        for i2 in range(2):
            for i3 in range(4):
                for i4 in range(3):
                    j = i1 * (2 * 4 * 3) + i2 * (4 * 3) + i3 * 3 + i4
                    args['out_dir'] = out_dir + ('%05d' % (counter * 96 + j))
                    cur_args = {**args, **back_args_set[i1], **leg_args_set[i2], **stretcher_args_set[i3],
                                **arm_args_set[i4]}
                    rec = Record(cur_args)
                    rec.seatDepth *= (1 - rec.mode * 0.4)
                    if not os.path.exists(args['out_dir']):
                        os.mkdir(args['out_dir'])
                        os.mkdir(args['out_dir'] + '/objs')
                    make_chair(Record(cur_args), counter * 96 + j)


def generate_chair_new(legWidth, legHeight, seatWidth, seatDepth,
                       seatHeight, backHeight, backDepth, counter):
    args = dict()
    args['out_dir'] = out_dir

    # Fix a set of lobal parameters
    args['legWidth'] = legWidth
    args['legHeight'] = legHeight
    args['seatWidth'] = seatWidth
    args['seatDepth'] = seatDepth
    args['seatHeight'] = seatHeight
    args['backHeight'] = backHeight
    args['backWidth'] = get_random(args['seatWidth'] - 0.1, args['seatWidth'])
    args['backDepth'] = backDepth

    # Make 4 variants for chair back
    backNumSurfaceBars = np.random.randint(low=3, high=5, size=3)
    backSurfaceBarDepth = 0.01 * np.random.random(6) + 0.01

    back_args_set = []
    back_args_set.append(dict({'back_type': 'simple'}))
    back_args_set.append(dict({'hasBackTopBar'     : True, 'hasBackSideBar': True, 'back_type': 'H',
                               'backTopBarHeight'  : args['backHeight'] / 5,
                               'backSideBarWidth'  : args['backWidth'] / 10,
                               'backNumSurfaceBars': backNumSurfaceBars[0], 'backSurfaceBarWidth': backSurfaceBarDepth[0], 'backSurfaceBarDepth': backSurfaceBarDepth[1]}))
    back_args_set.append(dict({'hasBackTopBar'     : False, 'hasBackSideBar': True, 'back_type': 'H',
                               'backTopBarHeight'  : args['backHeight'] / 5,
                               'backSideBarWidth'  : args['backWidth'] / 10,
                               'backNumSurfaceBars': backNumSurfaceBars[1], 'backSurfaceBarWidth': backSurfaceBarDepth[2], 'backSurfaceBarDepth': backSurfaceBarDepth[3]}))
    back_args_set.append(dict({'hasBackTopBar'     : True, 'hasBackSideBar': False, 'back_type': 'T',
                               'backTopBarHeight'  : args['backHeight'] / 10,
                               'backSideBarWidth'  : args['backWidth'] / 10,
                               'backNumSurfaceBars': backNumSurfaceBars[2], 'backSurfaceBarWidth': backSurfaceBarDepth[4], 'backSurfaceBarDepth': backSurfaceBarDepth[5]}))

    # Make 2 variants for chair legs
    leg_args_set = []
    leg_args_set.append(dict({'mode': 0}))
    leg_args_set.append(dict({'mode': 0.7}))

    # Make 4 variants for chair stretchers
    stretcher_args_set = []
    stretcher_args_set.append(dict({'hasStretchers': False}))
    stretcher_args_set.append(dict({'hasStretchers': True, 'stretcher_type': 'O'}))
    stretcher_args_set.append(dict({'hasStretchers': True, 'stretcher_type': 'H'}))
    stretcher_args_set.append(dict({'hasStretchers': True, 'stretcher_type': 'X'}))

    # Make 3 variants for chair arms
    arm_args_set = []
    arm_args_set.append(dict({'hasArmrests': False}))
    arm_args_set.append(dict({'hasArmrests': True, 'arm_type': 'T'}))
    arm_args_set.append(dict({'hasArmrests': True, 'arm_type': '7'}))

    # Make 4 x 2 x 4 x 3 = 96 variants for each chair
    for i1 in range(4):
        for i2 in range(2):
            for i3 in range(4):
                for i4 in range(3):
                    j = i1 * (2 * 4 * 3) + i2 * (4 * 3) + i3 * 3 + i4
                    args['out_dir'] = out_dir + ('%05d' % (counter * 96 + j))
                    cur_args = {**args, **back_args_set[i1], **leg_args_set[i2], **stretcher_args_set[i3],
                                **arm_args_set[i4]}
                    rec = Record(cur_args)
                    rec.seatDepth *= (1 - rec.mode * 0.4)
                    if not os.path.exists(args['out_dir']):
                        os.mkdir(args['out_dir'])
                        os.mkdir(args['out_dir'] + '/objs')
                    make_chair(Record(cur_args), counter * 96 + j)


if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm

    num_to_gen = 105
    print('Generating random chairs to folder %s ...' % (out_dir))
    # Fix a set of lobal parameters
    # args['legWidth'] = get_random(0.02, 0.07)
    # args['legHeight'] = get_random(0.1, 0.4)
    # args['seatWidth'] = get_random(0.4, 1.0)
    # args['seatDepth'] = get_random(0.4, 1.0)
    # args['seatHeight'] = get_random(0.02, 0.1)
    # args['backHeight'] = get_random(0.2, 0.5)
    # args['backDepth'] = get_random(0.02, 0.1)
    legWidth = np.linspace(0.02, 0.07, 3, endpoint=True)
    legHeight = np.linspace(0.1, 0.4, 2, endpoint=True)
    seatWidth = np.linspace(0.4, 1.0, 3, endpoint=True)
    seatDepth = np.linspace(0.4, 1.0, 3, endpoint=True)
    seatHeight = np.linspace(0.02, 0.1, 3, endpoint=True)
    backHeight = np.linspace(0.2, 0.5, 3, endpoint=True)
    backDepth = np.linspace(0.02, 0.1, 3, endpoint=True)
    counter = 0
    for i1 in tqdm(legWidth):
        for i2 in tqdm(legHeight, leave=False):
            for i3 in tqdm(seatWidth, leave=False):
                for i4 in tqdm(seatDepth, leave=False):
                    for i5 in tqdm(seatHeight, leave=False):
                        for i6 in tqdm(backHeight, leave=False):
                            for i7 in tqdm(backDepth, leave=False):
                                generate_chair(i1, i2, i3, i4,
                                               i5, i6, i7, counter)
                                counter += 1
