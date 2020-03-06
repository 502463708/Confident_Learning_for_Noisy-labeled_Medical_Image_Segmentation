import argparse
import cv2
import numpy as np
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from logger.logger import Logger
from time import time


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/clean-data/',
                        help='Clean data root dir.')
    parser.add_argument('--noisy_data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/noisy-data-alpha-0.5-beta1-7-beta2-12/all/',
                        help='Noisy data root dir.')
    parser.add_argument('--label_class_name',
                        type=str,
                        default='clavicle',  # 'clavicle', 'heart', 'lung'
                        help='The label class name.')
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
    image_dir = os.path.join(args.clean_data_root_dir, args.dataset_type, 'images')
    clean_mask_dir = os.path.join(args.clean_data_root_dir, args.dataset_type, args.label_class_name)
    noisy_mask_dir = os.path.join(args.noisy_data_root_dir, args.dataset_type, args.label_class_name)
    confident_map_dir = os.path.join(args.noisy_data_root_dir, args.dataset_type,
                                     '{}-confident-maps'.format(args.label_class_name))

    assert os.path.exists(clean_mask_dir)
    assert os.path.exists(noisy_mask_dir)
    assert os.path.exists(confident_map_dir)
    assert os.path.exists(args.dst_saving_dir)

    dst_saving_dir = os.path.join(args.dst_saving_dir, args.noisy_data_root_dir.split('/')[-3] + '-' +
                                  args.dataset_type + '-' + args.label_class_name)
    if os.path.exists(dst_saving_dir):
        shutil.rmtree(dst_saving_dir)
    os.mkdir(dst_saving_dir)

    logger = Logger(dst_saving_dir, 'logger.txt')
    logger.write_and_print('Clean mask dir: {}'.format(clean_mask_dir))
    logger.write_and_print('Noisy mask dir: {}'.format(noisy_mask_dir))
    logger.write_and_print('Confident map dir: {}'.format(confident_map_dir))

    filename_list = os.listdir(clean_mask_dir)

    dice_list = list()
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

        # calculating noise identification dice
        intersection = (noise_np * confident_map_np).sum()
        union = noise_np.sum() + confident_map_np.sum()

        dice_score = 2. * intersection / (union + 1e-4)

        dice_list.append(dice_score)

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

        logger.write_and_print('    Dice score = {:.4f}s'.format(dice_score))
        logger.write_and_print('    Finished evaluating, consuming time = {:.4f}s'.format(time() - start_time_for_batch))
        logger.write_and_print('--------------------------------------------------------------------------------------')

        logger.flush()

    mean_dice_score = np.array(dice_list).mean()

    logger.write_and_print('--------------------------------------------------------------------------------------')
    logger.write_and_print('Mean dice score = {:.4f}s'.format(mean_dice_score))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestNoiseIdentification(args)
