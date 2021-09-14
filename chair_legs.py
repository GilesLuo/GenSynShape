from utils import *
import random


def make_left_front_leg(args):
    x_min = args.seatWidth - args.legWidth * (args.mode + 1)
    x_max = args.seatWidth
    y_min = -args.mode * args.legHeight
    y_max = args.legHeight
    z_min = args.seatDepth - args.legWidth * (args.mode + 1)
    z_max = args.seatDepth
    return [create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max)]


def make_left_back_leg(args):
    x_min = args.seatWidth - args.legWidth * (args.mode + 1)
    x_max = args.seatWidth
    y_min = -args.mode * args.legHeight
    y_max = args.legHeight
    z_min = 0
    z_max = args.legWidth * (args.mode + 1)
    return [create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max)]


def make_right_front_leg(args):
    x_min = 0
    x_max = args.legWidth * (args.mode + 1)
    y_min = -args.mode * args.legHeight
    y_max = args.legHeight
    z_min = args.seatDepth - args.legWidth * (args.mode + 1)
    z_max = args.seatDepth
    return [create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max)]


def make_right_back_leg(args):
    x_min = 0
    x_max = args.legWidth * (args.mode + 1)
    y_min = -args.mode * args.legHeight
    y_max = args.legHeight
    z_min = 0
    z_max = args.legWidth * (args.mode + 1)
    return [create_axis_aligned_setting(x_min, x_max, y_min, y_max, z_min, z_max)]


def make_legs(args, box_id):
    legs = []
    csg = {'name': 'chair_base', 'children': []}
    csg_reg_base = {'name': 'regular_leg_base', 'children': []}
    legs += make_left_front_leg(args)
    csg_reg_base['children'].append({'name': 'leg', 'objs': [str(box_id)]})
    legs += make_left_back_leg(args)
    csg_reg_base['children'].append({'name': 'leg', 'objs': [str(box_id + 1)]})
    legs += make_right_front_leg(args)
    csg_reg_base['children'].append({'name': 'leg', 'objs': [str(box_id + 2)]})
    legs += make_right_back_leg(args)
    csg_reg_base['children'].append({'name': 'leg', 'objs': [str(box_id + 3)]})
    csg['children'].append(csg_reg_base)
    return legs, csg, box_id + 4
