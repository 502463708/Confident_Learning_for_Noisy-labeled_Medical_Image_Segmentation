import cv2
import numpy as np
import os

from sklearn.model_selection import train_test_split


def image_filename_list_split(image_dir, training_ratio, validation_ratio, test_ratio, random_seed, logger=None):
    image_list = os.listdir(image_dir)
    assert len(image_list) > 0

    image_list_training_and_val, image_list_test, _, _ = train_test_split(image_list, image_list,
                                                                          test_size=test_ratio,
                                                                          random_state=random_seed if random_seed >= 0 else None)

    image_list_training, image_list_val, _, _ = train_test_split(image_list_training_and_val,
                                                                 image_list_training_and_val,
                                                                 test_size=validation_ratio / (
                                                                         validation_ratio + training_ratio),
                                                                 random_state=random_seed if random_seed >= 0 else None)

    if logger is None:
        print('*******************************************************************************************************')
        print('Training set contains {} images.'.format(len(image_list_training)))
        print('Validation set contains {} images.'.format(len(image_list_val)))
        print('Test set contains {} images.'.format(len(image_list_test)))
        print('*******************************************************************************************************')
    else:
        logger.write_and_print('**************************************************************************************')
        logger.write_and_print('Training set contains {} images.'.format(len(image_list_training)))
        logger.write_and_print('Validation set contains {} images.'.format(len(image_list_val)))
        logger.write_and_print('Test set contains {} images.'.format(len(image_list_test)))
        logger.write_and_print('**************************************************************************************')

    return image_list_training, image_list_val, image_list_test


def crop_image_and_label(image_np, label_np, logger=None, intensity_threshold=2000):
    assert image_np.shape == label_np.shape
    assert len(image_np.shape) == 2

    height, width = image_np.shape

    # calculate start_crop_column_idx and end_crop_column_idx
    image_np_summed_along_with_column = image_np.sum(0)
    column_idx_list = np.where(image_np_summed_along_with_column > intensity_threshold)[0]
    start_crop_column_idx = 0
    end_crop_column_idx = width - 1
    if len(column_idx_list) > 0:
        start_crop_column_idx = column_idx_list[0]
        end_crop_column_idx = column_idx_list[-1]

    # calculate start_crop_row_idx and end_crop_row_idx
    image_np_summed_along_with_row = image_np.sum(1)
    row_idx_list = np.where(image_np_summed_along_with_row > intensity_threshold)[0]
    start_crop_row_idx = 0
    end_crop_row_idx = width - 1
    if len(row_idx_list) > 0:
        start_crop_row_idx = row_idx_list[0]
        end_crop_row_idx = row_idx_list[-1]

    if logger is None:
        print('  column idx: {} - {} -> {} - {}'.format(0, width - 1, start_crop_column_idx, end_crop_column_idx))
        print('  row idx: {} - {} -> {} - {}'.format(0, height - 1, start_crop_row_idx, end_crop_row_idx))
    else:
        logger.write_and_print(
            '  column idx: {} - {} -> {} - {}'.format(0, width - 1, start_crop_column_idx, end_crop_column_idx))
        logger.write_and_print(
            '  row idx: {} - {} -> {} - {}'.format(0, height - 1, start_crop_row_idx, end_crop_row_idx))

    image_np = image_np[start_crop_row_idx: end_crop_row_idx, start_crop_column_idx:end_crop_column_idx]
    label_np = label_np[start_crop_row_idx: end_crop_row_idx, start_crop_column_idx:end_crop_column_idx]

    assert image_np.shape == label_np.shape

    return image_np, label_np


def crop_and_save_data(filename_list, image_dir, label_dir, save_path, dataset_type, intensity_threshold, logger=None):
    assert len(filename_list) > 0

    current_idx = 0
    for filename in filename_list:
        current_idx += 1
        if logger is None:
            print('---------------------------------------------------------------------------------------------------')
            print(
                'Processing {} out of {}: {} in {} set'.format(current_idx, len(filename_list), filename, dataset_type))
        else:
            logger.write_and_print('----------------------------------------------------------------------------------')
            logger.write_and_print(
                'Processing {} out of {}: {} in {} set'.format(current_idx, len(filename_list), filename, dataset_type))

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)

        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if intensity_threshold > 0:
            image_np, label_np = crop_image_and_label(image_np, label_np, logger, intensity_threshold)
        else:
            image_np, label_np = crop_image_and_label(image_np, label_np, logger)

        assert image_np.shape == label_np.shape

        dst_data_dir = os.path.join(save_path, dataset_type)

        cv2.imwrite(os.path.join(dst_data_dir, 'images', filename), image_np)
        cv2.imwrite(os.path.join(dst_data_dir, 'labels', filename), label_np)

    return
