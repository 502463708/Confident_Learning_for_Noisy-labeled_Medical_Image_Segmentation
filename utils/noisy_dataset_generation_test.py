import argparse
import cv2
import os
import random
import shutil

from utils.noisy_dataset_generation import add_noise


def ParseArguments():
    parser = argparse.ArgumentParser()

    alpha = 0  # 0.3 0.5 0.7
    class_name = 'clavicle'  # 'clavicle', 'heart', 'lung'
    beta = 0

    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/clean-data',
                        help='Source dir of dataset.')
    parser.add_argument('--alpha',
                        type=float,
                        default=alpha,
                        help='The ratio of data to be deteriorated with noisy.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/noisy-data-alpha-{}-{}-{}'.format(alpha, class_name, beta),
                        help='Destination dir of dataset.')
    parser.add_argument('--label_class_name_list',
                        type=list,
                        default=['clavicle', 'heart', 'lung'],
                        help='Class name list.')
    parser.add_argument('--label_class_parameter_list',
                        type=list,
                        default=[[beta, beta], [beta, beta], [beta, beta]],
                        help='Class name list.')

    args = parser.parse_args()

    assert os.path.exists(args.src_data_root_dir), \
        'Source data root dir does not exist: {}.'.format(args.src_data_root_dir)

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)

    all_root_dir = os.path.join(args.dst_data_root_dir, 'all')
    sub_data_1_root_dir = os.path.join(args.dst_data_root_dir, 'sub-1')
    sub_data_2_root_dir = os.path.join(args.dst_data_root_dir, 'sub-2')

    shutil.copytree(args.src_data_root_dir, all_root_dir)

    args.all_root_dir = all_root_dir
    args.sub_data_1_root_dir = sub_data_1_root_dir
    args.sub_data_2_root_dir = sub_data_2_root_dir

    return args


def TestNoisyDatasetGeneration(args):
    training_image_dir = os.path.join(args.all_root_dir, 'training', 'images')
    filename_list = os.listdir(training_image_dir)
    noisy_data_num = int(len(filename_list) * args.alpha)

    # add noise into the training data
    for label_class_name, label_class_parameter in zip(args.label_class_name_list, args.label_class_parameter_list):
        random.shuffle(filename_list)
        noisy_filename_list = filename_list[:noisy_data_num]

        for noisy_filename in noisy_filename_list:
            src_label_path = os.path.join(args.all_root_dir, 'training', label_class_name, noisy_filename)
            src_label_np = cv2.imread(src_label_path, cv2.IMREAD_GRAYSCALE)

            if label_class_parameter[1] > 0:
                dst_label_np, noise_type = add_noise(src_label_np, label_class_parameter[0], label_class_parameter[1])
            else:
                dst_label_np = src_label_np
                noise_type = 'No noise'
            print(label_class_name, noisy_filename, noise_type)

            cv2.imwrite(src_label_path, dst_label_np)

    #
    shutil.copytree(args.all_root_dir, args.sub_data_1_root_dir)
    shutil.copytree(args.all_root_dir, args.sub_data_2_root_dir)

    random.shuffle(filename_list)
    half_data_num = int(len(filename_list) * 0.5)
    sub_1_delete_filename_list = filename_list[:half_data_num]
    sub_2_delete_filename_list = filename_list[half_data_num:]

    dir_name_list = args.label_class_name_list
    dir_name_list.append('images')

    for class_name in dir_name_list:
        for sub_1_delete_filename in sub_1_delete_filename_list:
            delete_file_path = os.path.join(args.sub_data_1_root_dir, 'training', class_name, sub_1_delete_filename)

            assert os.path.exists(delete_file_path), '{} do not exist'.format(delete_file_path)
            os.remove(delete_file_path)
            print('Delete file in sub dataset 1: {}'.format(delete_file_path))
            assert not os.path.exists(delete_file_path), '{} still exist'.format(delete_file_path)

    for class_name in dir_name_list:
        for sub_2_delete_filename in sub_2_delete_filename_list:
            delete_file_path = os.path.join(args.sub_data_2_root_dir, 'training', class_name, sub_2_delete_filename)

            assert os.path.exists(delete_file_path), '{} do not exist'.format(delete_file_path)
            os.remove(delete_file_path)
            print('Delete file in sub dataset 2: {}'.format(delete_file_path))
            assert not os.path.exists(delete_file_path), '{} still exist'.format(delete_file_path)

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestNoisyDatasetGeneration(args)
