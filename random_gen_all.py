import argparse
import os
import sys
from progressbar import ProgressBar
import random
from random_gen_chair import get_random, Record
from chair_grammar import make_chair, make_sofa, make_stool

from chair_back import *
from chair_seat import *
from chair_legs import *
from chair_armrests import *
from chair_stretchers import *


def generate_furniture(cat_name, out_dir, legWidth, legHeight, seatWidth, seatDepth,
                       seatHeight, backHeight, backDepth, counter):
    if cat_name == "Chair":
        generate_furniture = make_chair
    elif cat_name == "Sofa":
        generate_furniture = make_sofa
    elif cat_name == "Stool":
        generate_furniture = make_stool
    else:
        raise ValueError('invalid cat_name')

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
    back_args_set = [dict({'back_type': 'simple'}),
                     dict({'hasBackTopBar'     : True, 'hasBackSideBar': True, 'back_type': 'H',
                           'backTopBarHeight'  : args['backHeight'] / 5,
                           'backSideBarWidth'  : args['backWidth'] / 10,
                           'backNumSurfaceBars': 5, 'backSurfaceBarWidth': 0.01, 'backSurfaceBarDepth': 0.01}),
                     dict({'hasBackTopBar'     : False, 'hasBackSideBar': True, 'back_type': 'H',
                           'backTopBarHeight'  : args['backHeight'] / 5,
                           'backSideBarWidth'  : args['backWidth'] / 10,
                           'backNumSurfaceBars': 3, 'backSurfaceBarWidth': 0.02, 'backSurfaceBarDepth': 0.02}),
                     dict({'hasBackTopBar'     : True, 'hasBackSideBar': False, 'back_type': 'T',
                           'backTopBarHeight'  : args['backHeight'] / 10,
                           'backSideBarWidth'  : args['backWidth'] / 10,
                           'backNumSurfaceBars': 4, 'backSurfaceBarWidth': 0.01, 'backSurfaceBarDepth': 0.01})]

    # Make 2 variants for chair legs
    leg_args_set = [dict({'mode': 0}), dict({'mode': 0.7})]

    # Make 4 variants for chair stretchers
    stretcher_args_set = [dict({'hasStretchers': False}), dict({'hasStretchers': True, 'stretcher_type': 'O'}),
                          dict({'hasStretchers': True, 'stretcher_type': 'H'}),
                          dict({'hasStretchers': True, 'stretcher_type': 'X'})]

    # Make 3 variants for chair arms
    arm_args_set = [dict({'hasArmrests': False}), dict({'hasArmrests': True, 'arm_type': 'T'}),
                    dict({'hasArmrests': True, 'arm_type': '7'})]

    # Make 4 x 2 x 4 x 3 = 96 variants for each chair
    for i1 in range(4):
        for i2 in range(2):
            for i3 in range(4):
                for i4 in range(3):
                    j = i1 * (2 * 4 * 3) + i2 * (4 * 3) + i3 * 3 + i4
                    args['out_dir'] = out_dir + str(counter * 96 + j)
                    cur_args = {**args, **back_args_set[i1], **leg_args_set[i2], **stretcher_args_set[i3],
                                **arm_args_set[i4]}
                    rec = Record(cur_args)
                    rec.seatDepth *= (1 - rec.mode * 0.4)
                    if not os.path.exists(args['out_dir']):
                        os.mkdir(args['out_dir'])
                        os.mkdir(args['out_dir'] + '/objs')
                    generate_furniture(Record(cur_args), counter * 96 + j)