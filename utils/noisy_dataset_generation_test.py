import argparse
import cv2
import os
import random
import shutil

from logger.logger import Logger


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/',
                        help='Source dir of dataset.')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.05,
                        help='The ratio of data to be deteriorated with noisy.')
    parser.add_argument('--beta',
                        type=float,
                        default=0.2,
                        help='The degree of label to be deteriorated.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/',
                        help='Destination dir of dataset.')
    parser.add_argument('--label_class_name_list',
                        type=list,
                        default=['clavicle', 'heart', 'lung'],
                        help='Destination dir of dataset.')

    args = parser.parse_args()

    assert os.path.exists(args.src_data_root_dir), \
        'Source data root dir does not exist: {}.'.format(args.src_data_root_dir)

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    shutil.copytree(args.src_data_root_dir, args.dst_data_root_dir)

    return args


def TestNoisyDatasetGeneration(args):
    # set up logger
    logger = Logger(args.dst_data_root_dir)

    training_image_dir = os.path.join(args.dst_data_root_dir, 'training', 'images')
    filename_list = os.listdir(training_image_dir)
    noisy_data_num = int(len(filename_list) * args.alpha)

    for label_class_name in args.label_class_name_list:
        random.shuffle(filename_list)
        noisy_filename_list = filename_list[:noisy_data_num]

        for noisy_filename in noisy_filename_list:
            src_label_path = os.path.join(args.dst_data_root_dir, 'training', label_class_name, noisy_filename)
            src_label_np = cv2.imread(src_label_path, cv2.IMREAD_GRAYSCALE)

            dst_label_np = add_noisy(src_label_np, args.beta)

            cv2.imwrite(src_label_path, dst_label_np)

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestNoisyDatasetGeneration(args)
