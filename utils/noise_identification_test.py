import argparse
import cv2
import numpy as np
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from logger.logger import Logger
from time import time

'''
noisy_data_root_dir
label_class_name: 'clavicle', 'heart', 'lung'
CL_type: 'prune_by_class' or 'prune_by_noise_rate'
'''

def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/clean-data/',
                        help='Clean data root dir.')
    parser.add_argument('--noisy_data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/noisy-data-alpha-0.5-lung-20/all/',
                        help='Noisy data root dir.')
    parser.add_argument('--label_class_name',
                        type=str,
                        default='lung',  # 'clavicle', 'heart', 'lung'
                        help='The label class name.')
    parser.add_argument('--CL_type',
                        type=str,
                        default='intersection',  # 'Cij', 'Qij', 'intersection', 'union', 'prune_by_class', 'prune_by_noise_rate', 'both'
                        help='The implement of Confident Learning.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='training',
                        help='The type of dataset to be evaluated (training, validation).')
    parser.add_argument('--dst_saving_dir',
                        type=str,
                        default='/data1/minqing/results/',
                        help='Destination dir for saving results.')

    args = parser.parse_args()

    return args


def TestNoiseIdentification(args):
    assert args.CL_type in ['Cij', 'Qij', 'intersection', 'union', 'prune_by_class', 'prune_by_noise_rate', 'both']

    image_dir = os.path.join(args.clean_data_root_dir, args.dataset_type, 'images')
    clean_mask_dir = os.path.join(args.clean_data_root_dir, args.dataset_type, args.label_class_name)
    noisy_mask_dir = os.path.join(args.noisy_data_root_dir, args.dataset_type, args.label_class_name)
    confident_map_dir = os.path.join(args.noisy_data_root_dir, args.dataset_type,
                                     '{}-confident-maps'.format(args.label_class_name))
    if args.CL_type != 'both':
        confident_map_dir = confident_map_dir.replace('confident-maps', 'confident-maps-' + args.CL_type.replace('_', '-'))

    assert os.path.exists(clean_mask_dir)
    assert os.path.exists(noisy_mask_dir)
    assert os.path.exists(confident_map_dir)
    assert os.path.exists(args.dst_saving_dir)

    dst_saving_dir = os.path.join(args.dst_saving_dir, args.noisy_data_root_dir.split('/')[-3] + '-' +
                                  args.dataset_type + '-' + args.label_class_name)
    if args.CL_type != 'both':
        dst_saving_dir += '-' + args.CL_type.replace('_', '-')
    if os.path.exists(dst_saving_dir):
        shutil.rmtree(dst_saving_dir)
    os.mkdir(dst_saving_dir)

    logger = Logger(dst_saving_dir, 'logger.txt')
    logger.write_and_print('Clean mask dir: {}'.format(clean_mask_dir))
    logger.write_and_print('Noisy mask dir: {}'.format(noisy_mask_dir))
    logger.write_and_print('Confident map dir: {}'.format(confident_map_dir))

    filename_list = os.listdir(clean_mask_dir)

    noise_num_dataset_level = 0
    positive_num_dataset_level = 0
    recall_num_dataset_level = 0

    for filename in filename_list:
        logger.write_and_print('--------------------------------------------------------------------------------------')
        logger.write_and_print('    Evaluating: {}'.format(filename))

        # start time of this batch
        start_time_for_batch = time()

        src_image_path = os.path.join(image_dir, filename)
        src_clean_mask_path = os.path.join(clean_mask_dir, filename)
        src_noisy_mask_path = os.path.join(noisy_mask_dir, filename)
        src_confident_map_path = os.path.join(confident_map_dir, filename)

        clean_mask_np = (cv2.imread(src_clean_mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float) / 255.0).astype(np.uint8)
        noisy_mask_np = (cv2.imread(src_noisy_mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float) / 255.0).astype(np.uint8)
        confident_map_np = cv2.imread(src_confident_map_path, cv2.IMREAD_GRAYSCALE).astype(np.float) / 255.0

        noise_np = clean_mask_np ^ noisy_mask_np

        # calculating noise identification metric
        recall_num_image_level = (noise_np * confident_map_np).sum()
        noise_num_image_level = noise_np.sum()
        positive_num_image_level = confident_map_np.sum()
        positive_num_dataset_level += positive_num_image_level
        recall_num_dataset_level += recall_num_image_level
        noise_num_dataset_level += noise_num_image_level

        dst_image_path = os.path.join(dst_saving_dir, filename.replace('.png', '_image.png'))
        dst_clean_mask_path = os.path.join(dst_saving_dir, filename.replace('.png', '_clean_mask.png'))
        dst_noisy_mask_path = os.path.join(dst_saving_dir, filename.replace('.png', '_noisy_mask.png'))
        dst_noise_path = os.path.join(dst_saving_dir, filename.replace('.png', '_noise.png'))
        dst_confident_map_path = os.path.join(dst_saving_dir, filename.replace('.png', '_confident_map.png'))

        shutil.copyfile(src_image_path, dst_image_path)
        shutil.copyfile(src_clean_mask_path, dst_clean_mask_path)
        shutil.copyfile(src_noisy_mask_path, dst_noisy_mask_path)
        shutil.copyfile(src_confident_map_path, dst_confident_map_path)
        cv2.imwrite(dst_noise_path, (noise_np * 255).astype(np.uint8))

        logger.write_and_print('    Noise pixel number = {}'.format(noise_num_image_level))
        logger.write_and_print('    Positive pixel number = {}'.format(positive_num_image_level))
        logger.write_and_print('    Recalled pixel number = {}'.format(recall_num_image_level))
        logger.write_and_print('    Finished evaluating, consuming time = {:.4f}s'.format(time() - start_time_for_batch))
        logger.write_and_print('--------------------------------------------------------------------------------------')

        logger.flush()

    mean_recall_rate = recall_num_dataset_level / noise_num_dataset_level
    mean_precision_rate = recall_num_dataset_level / positive_num_dataset_level
    f1_score = 2 * mean_recall_rate * mean_precision_rate / (mean_recall_rate + mean_precision_rate)

    logger.write_and_print('--------------------------------------------------------------------------------------')
    logger.write_and_print('Mean recall rate = {:.2f}%'.format(mean_recall_rate * 100))
    logger.write_and_print('Mean precision rate = {:.2f}%'.format(mean_precision_rate * 100))
    logger.write_and_print('F1 score = {:.2f}%'.format(f1_score * 100))
    print('{:.2f}%\t{:.2f}%\t{:.2f}%'.format(mean_recall_rate * 100, mean_precision_rate * 100, f1_score * 100))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestNoiseIdentification(args)
