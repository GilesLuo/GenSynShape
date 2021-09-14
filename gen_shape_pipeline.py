import numpy as np
from tqdm import tqdm
from random_gen_chair import generate_chair_new
from seperate_data import seperate_data


def gen_chair(out_dir, l1=3, l2=3, s1=3, s2=3, s3=3, b1=3, b2=3):
    legWidth = np.linspace(0.02, 0.07, l1, endpoint=True)
    legHeight = np.linspace(0.1, 0.4, l2, endpoint=True)
    seatWidth = np.linspace(0.4, 1.0, s1, endpoint=True)
    seatDepth = np.linspace(0.4, 1.0, s2, endpoint=True)
    seatHeight = np.linspace(0.02, 0.1, s3, endpoint=True)
    backHeight = np.linspace(0.2, 0.5, b1, endpoint=True)
    backDepth = np.linspace(0.02, 0.1, b2, endpoint=True)
    counter = 0

    pbar = tqdm(desc='Generating random chairs to folder %s ...' % (out_dir),
                total=len(legWidth) * len(legHeight) * len(seatWidth) * len(
                    seatDepth) * len(seatHeight) * len(backHeight) * len(backDepth) * 96)
    for i1 in legWidth:
        for i2 in legHeight:
            for i3 in seatWidth:
                for i4 in seatDepth:
                    for i5 in seatHeight:
                        for i6 in backHeight:
                            for i7 in backDepth:
                                generate_chair_new(out_dir, i1, i2, i3, i4,
                                                   i5, i6, i7, counter)
                                counter += 1
                                pbar.update(96)
    seperate_data(out_dir, object_name='Chair')
    print("data has been seperated into " + out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='gensyn/chair/')
    parser.add_argument('--gen_nums', default=(3, 3, 3, 3, 3, 3, 3))
    parser.add_argument('--category', type=str, help='model def file')
    parser.add_argument('--model_dir', type=str, help='the path of the model')
    args = parser.parse_args()
    gen_chair(args.out_dir, *args.gen_nums)
