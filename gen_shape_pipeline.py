import numpy as np
from tqdm import tqdm
from random_gen_chair import generate_chair_new
from pyquaternion import Quaternion
import prepare_shape
import os
import json
import random


class GenShapes:
    def __init__(self, source_dir):
        self.source_dir = source_dir

    def gen_chair(self, obj_save_dir, l1=3, l2=3, s1=3, s2=3, s3=3, b1=3, b2=3):
        legWidth = np.linspace(0.02, 0.07, l1, endpoint=True)
        legHeight = np.linspace(0.1, 0.4, l2, endpoint=True)
        seatWidth = np.linspace(0.4, 1.0, s1, endpoint=True)
        seatDepth = np.linspace(0.4, 1.0, s2, endpoint=True)
        seatHeight = np.linspace(0.02, 0.1, s3, endpoint=True)
        backHeight = np.linspace(0.2, 0.5, b1, endpoint=True)
        backDepth = np.linspace(0.02, 0.1, b2, endpoint=True)
        counter = 0

        pbar = tqdm(desc='Generating random chairs to folder %s' % obj_save_dir,
                    total=len(legWidth) * len(legHeight) * len(seatWidth) * len(
                        seatDepth) * len(seatHeight) * len(backHeight) * len(backDepth) * 96)
        for i1 in legWidth:
            for i2 in legHeight:
                for i3 in seatWidth:
                    for i4 in seatDepth:
                        for i5 in seatHeight:
                            for i6 in backHeight:
                                for i7 in backDepth:
                                    generate_chair_new(obj_save_dir, i1, i2, i3, i4,
                                                       i5, i6, i7, counter)
                                    counter += 1
                                    pbar.update(96)

    def pipeline(self, gen_info: dir, stat_path:str, output_dir:str):
        """
        :param gen_info: is a dir in the format of {'cat_name': args}
                e.g., {'Chair': (1,1,1,1,1,1,1),
                       'Table': (1,1,1,1,1,1,1),
                      }
        :return:
        """
        for cat_name, args in gen_info.items():
            obj_save_dir = self.source_dir + cat_name + '/'
            if not os.path.exists(obj_save_dir):
                os.mkdir(obj_save_dir)
            if os.listdir(obj_save_dir):
                print("chair obj data already exists in " + obj_save_dir + ", skipping...")
            else:
                self.gen_chair(obj_save_dir, *args)
            self.separate_data(obj_save_dir, object_name=cat_name, output_dir=self.source_dir)
            self.prepare_dataset(obj_save_dir, self.source_dir, stat_path, cat_name)

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
            print("dataset json already exists in " + output_dir + ", skipping...")
            return
        print("separating data into " + obj_dir)
        data = []
        for folder in os.listdir(obj_dir):
            # print(folder)
            data.append({'anno_id': '{}'.format(folder)})
        random_data = random.sample(data, len(data))

        train, val, test = [], [], []
        for file in random_data:
            if len(train) < int(len(random_data) * 7 / 10):
                train.append(file)
            elif len(val) < int(len(random_data) * 1 / 10):
                val.append(file)
            else:
                test.append(file)
        print("Separation done. \n"
              "train set size: {},\n"
              "val   set size: {},\n "
              "test  set size: {}".format(len(train), len(val), len(test)))

        with open('{}.train.json'.format(output_dir + object_name), 'w') as result_file:
            json.dump(train, result_file)

        with open('{}.val.json'.format(output_dir + object_name), 'w') as result_file:
            json.dump(val, result_file)

        with open('{}.test.json'.format(output_dir + object_name), 'w') as result_file:
            json.dump(test, result_file)

    def prepare_dataset(self, obj_dir, root_to_save_file, stat_path,
                        cat_name, modes=None, levels=None):
        """
        :param root_to_save_file: output dir
        :param cat_name: select one from {"Cabinet", "Table", "Chair", 'Lamp'}
        :return:
        """
        if levels is None:
            levels = [3, 1, 2]
        if modes is None:
            modes = ["train", "val", "test"]

        # import hier imformation
        fn_hier = stat_path + "after_merging_label_ids/" + cat_name + '-hier.txt'
        with open(fn_hier) as f:
            hier = f.readlines()
            hier = {'/' + s.split(' ')[1].replace('\n', ''): int(s.split(' ')[0]) for s in hier}

            print(hier)
        # for each level
        for level in levels:

            # import level information
            fn_level = stat_path + "after_merging_label_ids/" + cat_name + '-level-' + str(level) + ".txt"
            lev = []
            with open(fn_level) as f:
                lev = f.readlines()
                lev = ['/' + s.split(' ')[1].replace('\n', '') for s in lev]

            # for each mode 
            num = 0
            for mode in modes:

                # get the object list to deal with
                # object_json =json.load(open(self.source_dir + "/train_val_test_split/" + cat_name +"." + mode + ".json"))
                object_json = json.load(open(self.source_dir + "/" + cat_name + "." + mode + ".json"))
                object_list = [object_json[i]['anno_id'] for i in range(len(object_json))]
                # print(ob)
                # for each object:
                for i, idx in enumerate(object_list):
                    print("level ", level, " mode ", mode, " ", idx, " is start to convert!", i, "/", len(object_list))

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
                    np.save(root_to_save_file + str(fn) + "_level" + str(level) + ".npy", dic_to_save)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='gensyn/')
    parser.add_argument('--gen_info', default={"Chair": (1, 1, 1, 1, 1, 1, 2)})
    args = parser.parse_args()

    shape_generator = GenShapes(source_dir=args.source_dir)
    shape_generator.pipeline(gen_info=args.gen_info,
                             stat_path="./stats/", output_dir=args.source_dir)
