import numpy as np
from tqdm import tqdm
from random_gen_chair import get_random
from pyquaternion.quaternion import Quaternion
import prepare_shape
from random_gen_all import generate_furniture
import os
import json
import copy
import ipdb
import torch
import multiprocessing as mp
from pathos.multiprocessing import ProcessPool as Pool
import math
import random
from prepare_contact_points import qrot, get_pair_list, find_pts_ind


class GenShapes:
    def __init__(self, source_dir, num_core=16):
        self.source_dir = source_dir
        if not os.path.exists(self.source_dir):
            os.mkdir(self.source_dir)
        self.num_core = num_core
        self.parallel_pool = Pool(self.num_core)

    def gen_furniture_lin(self, obj_save_dir, cat_name, l1, l2, s1, s2, s3, b1, b2,
                          l1_l=0.02, l1_h=0.07,
                          l2_l=0.1, l2_h=0.4,
                          s1_l=0.4, s1_h=1.0,
                          s2_l=0.4, s2_h=1.0,
                          s3_l=0.02, s3_h=0.1,
                          b1_l=0.2, b1_h=0.5,
                          b2_l=0.02, b2_h=0.1):
        legWidth = np.linspace(l1_l, l1_h, l1, endpoint=True) + (l1_h - l1_l) * 0.1 * (2 * np.random.random(l1) - 1)
        legHeight = np.linspace(l2_l, l2_h, l2, endpoint=True) + (l2_h - l2_l) * 0.1 * (2 * np.random.random(l2) - 1)
        seatWidth = np.linspace(s1_l, s1_h, s1, endpoint=True) + (s1_l - s1_l) * 0.1 * (2 * np.random.random(l1) - 1)
        seatDepth = np.linspace(s2_l, s2_h, s2, endpoint=True) + (s2_l - s2_h) * 0.1 * (2 * np.random.random(l1) - 1)
        seatHeight = np.linspace(s3_l, s3_h, s3, endpoint=True) + (s3_l - s3_h) * 0.1 * (2 * np.random.random(l1) - 1)
        backHeight = np.linspace(b1_l, b1_h, b1, endpoint=True) + (b1_l - b1_h) * 0.1 * (2 * np.random.random(l1) - 1)
        backDepth = np.linspace(b2_l, b2_h, b2, endpoint=True) + (b2_l - b2_h) * 0.1 * (2 * np.random.random(l1) - 1)
        counter = 0

        allResults = []
        for i1 in legWidth:
            for i2 in legHeight:
                for i3 in seatWidth:
                    for i4 in seatDepth:
                        for i5 in seatHeight:
                            for i6 in backHeight:
                                for i7 in backDepth:
                                    result = self.parallel_pool.apipe(generate_furniture,
                                                                      cat_name, obj_save_dir, i1, i2, i3, i4,
                                                                      i5, i6, i7, counter)
                                    allResults.append([result, counter])
                                    counter += 1
        jobsCompleted = 0
        pbar = tqdm(total=len(allResults * 96))
        while len(allResults) > 0:
            for i in range(len(allResults)):
                task, j = allResults[i]
                if task.ready():
                    jobsCompleted += 1
                    pbar.desc = "        using {} cores, " \
                                "generating obj".format(self.num_core, j)
                    pbar.update(96)
                    task.get()
                    allResults.pop(i)
                    break

    def gen_furniture_rand(self, obj_save_dir, cat_name, pair_gen,
                           l1_l=0.02, l1_h=0.07,
                           l2_l=0.1, l2_h=0.4,
                           s1_l=0.4, s1_h=1.0,
                           s2_l=0.4, s2_h=1.0,
                           s3_l=0.02, s3_h=0.1,
                           b1_l=0.2, b1_h=0.5,
                           b2_l=0.02, b2_h=0.1, ):
        allResults = []
        for i in range(pair_gen):
            legWidth = get_random(l1_l, l1_h)
            legHeight = get_random(l2_l, l2_h)
            seatWidth = get_random(s1_l, s1_h)
            seatDepth = get_random(s2_l, s2_h)
            seatHeight = get_random(s3_l, s3_h)
            backHeight = get_random(b1_l, b1_h)
            backDepth = get_random(b2_l, b2_h)
            result = self.parallel_pool.apipe(generate_furniture,
                                              cat_name, obj_save_dir, legWidth, legHeight, seatWidth, seatDepth,
                                              seatHeight, backHeight, backDepth, i)
            allResults.append([result, i])

        jobsCompleted = 0
        pbar = tqdm(total=len(allResults * 96))
        while len(allResults) > 0:
            for i in range(len(allResults)):
                task, j = allResults[i]
                if task.ready():
                    jobsCompleted += 1
                    pbar.desc = "        using {} cores, " \
                                "generating obj".format(self.num_core, j)
                    pbar.update(96)
                    task.get()
                    allResults.pop(i)
                    break

    def run_pipeline(self, gen_info: dir, stat_path: str, method: str):
        """
        :param gen_info: is a dir in the format of {'cat_name': args}
                e.g., {'Chair': (1,1,1,1,1,1,1),
                       'Table': (1,1,1,1,1,1,1),
                      }
        :return:
        """

        for cat_name, args in gen_info.items():
            obj_save_dir = self.source_dir + cat_name + '_obj/'
            if not os.path.exists(obj_save_dir):
                os.mkdir(obj_save_dir)
            if os.listdir(obj_save_dir):
                print("chair obj data already exists in " + obj_save_dir + ", skipping...")
            else:
                if method == "random":
                    self.gen_furniture_rand(obj_save_dir, cat_name, np.prod(list(args)))
                elif method == "linspace":
                    self.gen_furniture_lin(obj_save_dir, cat_name, *args)
                else:
                    raise ValueError('only support random and linspace method')
            self.separate_data(obj_save_dir, object_name=cat_name, output_dir=self.source_dir)
            root_to_save_shape = self.source_dir + "shape_data/"

            if not os.path.exists(root_to_save_shape):
                os.mkdir(root_to_save_shape)
            self.prepare_data(obj_save_dir, root_to_save_shape, stat_path, cat_name, levels=[3])

            root_to_save_contact_points = self.source_dir + "contact_points/"
            if not os.path.exists(root_to_save_contact_points):
                os.mkdir(root_to_save_contact_points)
            self.prepare_contact_points(root_to_save_shape, root_to_save_contact_points,
                                        cat_name, levels=[3])

    @staticmethod
    def separate_data(obj_dir: str, object_name, output_dir=None):
        """
        :param output_dir:
        :param obj_dir: dir that stores the obj
        :param object_name: {"Chair", "Table", "Lamp", ...}
        """

        if output_dir is None:
            output_dir = obj_dir + '../'

        if os.path.exists("{}.test.json".format(output_dir + object_name)) and \
                os.path.exists("{}.train.json".format(output_dir + object_name)) and \
                os.path.exists("{}.val.json".format(output_dir + object_name)):
            print("dataset json already exists in " + output_dir + ", skipped...")
            return
        data = []
        for folder in os.listdir(obj_dir):
            # print(folder)
            data.append({'anno_id': '{}'.format(folder)})
        random_data = random.sample(data, len(data))

        train, val, test = [], [], []
        train_npy, val_npy, test_npy = [], [], []
        for file in tqdm(random_data, desc="separating data"):
            if len(train) < int(len(random_data) * 7 / 10):
                train.append(file)
                train_npy.append(int(file["anno_id"]))
            elif len(val) < int(len(random_data) * 1 / 10):
                val.append(file)
                val_npy.append(int(file["anno_id"]))
            else:
                test.append(file)
                test_npy.append(int(file["anno_id"]))
        print("Data separated. "
              "train set size: {}, val set size: {}, test set size: {}".format(len(train), len(val), len(test)))

        with open('{}.train.json'.format(output_dir + object_name), 'w') as result_file:
            json.dump(train, result_file)
        np.save('{}.train.npy'.format(output_dir + object_name), np.array(train_npy).squeeze())

        with open('{}.val.json'.format(output_dir + object_name), 'w') as result_file:
            json.dump(val, result_file)
        np.save('{}.val.npy'.format(output_dir + object_name), np.array(train_npy).squeeze())

        with open('{}.test.json'.format(output_dir + object_name), 'w') as result_file:
            json.dump(test, result_file)
        np.save('{}.test.npy'.format(output_dir + object_name), np.array(train_npy).squeeze())

    def prepare_data(self, obj_dir, root_to_save_file, stat_path,
                     cat_name, modes=None, levels=None):

        if levels is None:
            levels = [3, 1, 2]
        if modes is None:
            modes = ["train", "val", "test"]

        # import hier imformation
        fn_hier = stat_path + "after_merging_label_ids/" + cat_name + '-hier.txt'
        with open(fn_hier) as f:
            hier = f.readlines()
            hier = {'/' + s.split(' ')[1].replace('\n', ''): int(s.split(' ')[0]) for s in hier}

            # print(hier)
        # for each level
        for level in levels:

            # import level information
            fn_level = stat_path + "after_merging_label_ids/" + cat_name + '-level-' + str(level) + ".txt"
            lev = []
            with open(fn_level) as f:
                lev = f.readlines()
                lev = ['/' + s.split(' ')[1].replace('\n', '') for s in lev]

            allResults = []
            for mode in modes:
                object_json = json.load(open(self.source_dir + "/" + cat_name + "." + mode + ".json"))
                object_list = [object_json[i]['anno_id'] for i in range(len(object_json))]
                for i, idx in enumerate(object_list):
                    result = self.parallel_pool.apipe(self._gen_1_shape,
                                                      root_to_save_file, idx, lev, level, obj_dir, hier)
                    allResults.append([result, level, mode, idx])

            jobsCompleted = 0
            pbar = tqdm(total=len(allResults))
            while len(allResults) > 0:
                for i in range(len(allResults)):
                    task, level, mode, idx = allResults[i]
                    if task.ready():
                        jobsCompleted += 1
                        pbar.desc = "        using {} cores, " \
                                    "generating shape pc level{}-{}-{}".format(self.num_core, level, mode, idx)
                        pbar.update()
                        task.get()
                        allResults.pop(i)
                        break

    @staticmethod
    def _gen_1_shape(root_to_save_file, idx, lev, level, obj_dir, hier):
        if os.path.exists(root_to_save_file + str(idx) + "_level" + str(level) + ".npy"):
            return

        # get information in obj file
        obj_folder_name = obj_dir + str(idx)
        parts_pcs, Rs, ts, parts_names, sizes = prepare_shape.get_shape_info(obj_folder_name, lev)

        # get class index and geo class index
        parts_ids = [hier[name] for name in parts_names]
        geo_part_ids = prepare_shape.get_geo_part_ids(sizes, parts_ids)

        # gen sym_stick info
        sym = prepare_shape.get_sym(parts_pcs)
        # get part poses from R , T
        parts_poses = []
        for R, t in zip(Rs, ts):
            if np.linalg.det(R) < 0:
                R = -R

            q = Quaternion(matrix=R)

            q = np.array([q[i] for i in range(4)])
            parts_pose = np.concatenate((t, q), axis=0)
            parts_poses.append(parts_pose)

        parts_poses = np.array(parts_poses)
        new_dict = {v: k for k, v in hier.items()}
        dic_to_save = {"part_pcs"    : parts_pcs, "part_poses": parts_poses, "part_ids": parts_ids,
                       "geo_part_ids": geo_part_ids, "sym": sym}
        np.save(root_to_save_file + str(idx) + "_level" + str(level) + ".npy", dic_to_save)
        return

    def prepare_contact_points(self, shape_dir, root_to_save_file, cat_name, modes=None, levels=None):
        if levels is None:
            levels = [3, 1, 2]
        if modes is None:
            modes = ["train", "val", "test"]

        allResults = []
        for level in levels:
            for mode in modes:
                object_json = json.load(open(self.source_dir + cat_name + "." + mode + ".json"))
                object_list = [object_json[i]['anno_id'] for i in range(len(object_json))]
                for id in object_list:
                    result = self.parallel_pool.apipe(self._gen_1_contact_point,
                                                      root_to_save_file, level, shape_dir, id)
                    allResults.append([result, level, mode, id])

        jobsCompleted = 0
        pbar = tqdm(total=len(allResults))
        while len(allResults) > 0:
            for i in range(len(allResults)):
                task, level, mode, id = allResults[i]
                if task.ready():
                    jobsCompleted += 1
                    pbar.desc = "        using {} cores, " \
                                "generating contact point level{}-{}-{}".format(self.num_core, level, mode, id)
                    pbar.update()
                    task.get()
                    allResults.pop(i)
                    break
        self.parallel_pool.close()

    @staticmethod
    def _gen_1_contact_point(root_to_save_file, level, shape_dir, id):
        if not os.path.exists(root_to_save_file + 'pairs_with_contact_points_%s_level' % id + str(
                level) + '.npy'):
            # if os.path.isfile(root + "contact_points/" + 'pairs_with_contact_points_%s_level' % id +
            # str(level) + '.npy'):
            cur_data_fn = os.path.join(shape_dir, '%s_level' % id + str(level) + '.npy')

            cur_data = np.load(cur_data_fn, allow_pickle=True).item()
            cur_pts = cur_data['part_pcs']  # p x N x 3 (p is unknown number of parts for this shape)
            class_index = cur_data['part_ids']
            num_parts, num_point, _ = cur_pts.shape
            poses = cur_data['part_poses']
            quat = poses[:, 3:]
            center = poses[:, :3]
            gt_pts = copy.copy(cur_pts)
            for i in range(num_parts):
                gt_pts[i] = qrot(torch.from_numpy(quat[i]).unsqueeze(0).repeat(num_point, 1).unsqueeze(0),
                                 torch.from_numpy(cur_pts[i]).unsqueeze(0))
                gt_pts[i] = gt_pts[i] + center[i]

            oldfile = get_pair_list(gt_pts)
            newfile = oldfile
            for i in range(len(oldfile)):
                for j in range(len(oldfile[0])):
                    if i == j: continue
                    point = oldfile[i, j, 1:]
                    ind = find_pts_ind(gt_pts[i], point)
                    if ind == -1:
                        ipdb.set_trace()
                    else:
                        newfile[i, j, 1:] = cur_pts[i, ind]
            np.save(root_to_save_file + 'pairs_with_contact_points_%s_level' % id + str(
                level) + '.npy', newfile)
            return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='gensyn/')
    parser.add_argument('--gen_info', default={"Chair": (1, 1, 1, 1, 1, 2, 2)})
    parser.add_argument('--method', default="linspace", help="choose from random and linspace")
    parser.add_argument('--num_core', default=3, help="number of core used for multi-processing")
    args = parser.parse_args()

    shape_generator = GenShapes(source_dir=args.source_dir, num_core=args.num_core)
    shape_generator.run_pipeline(gen_info=args.gen_info,
                                 stat_path="./stats/", method=args.method)
