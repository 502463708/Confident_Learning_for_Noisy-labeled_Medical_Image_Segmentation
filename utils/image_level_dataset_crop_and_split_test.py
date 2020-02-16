import argparse
import os
import shutil

from logger.logger import Logger
from utils.image_level_dataset_crop_and_split import image_filename_list_split, crop_and_save_data


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-20191105/Inbreast-radiograph-level-raw-images-with-pixel-level-labels-dataset/',
                        help='Source data root dir.')

    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-20191105/Inbreast-radiograph-level-roi-extracted-data-split-dataset/',
                        help='Destination data root dir.')

    parser.add_argument('--random_seed',
                        type=int,
                        default=0,
                        help='Set random seed for reduplicating the results.'
                             '-1 -> do not set random seed.')

    parser.add_argument('--training_ratio',
                        type=float,
                        default=0.6,
                        help='The ratio of training set over all.')

    parser.add_argument('--validation_ratio',
                        type=float,
                        default=0.2,
                        help='The ratio of validation set over all.')

    parser.add_argument('--test_ratio',
                        type=float,
                        default=0.2,
                        help='The ratio of test set over all.')

    parser.add_argument('--intensity_threshold',
                        type=int,
                        default=-1,
                        help='The intensity threshold for cropping: row (column) with summed intensity less than '
                             'intensity_threshold will be cropped out.'
                             '-1 -> the default intensity_threshold 2000.')

    args = parser.parse_args()

    assert os.path.exists(args.src_data_root_dir), 'Source data root dir does not exist.'

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)

    for dataset_type in ['training', 'validation', 'test']:
        dataset_type_dir = os.path.join(args.dst_data_root_dir, dataset_type)
        os.mkdir(dataset_type_dir)
        os.mkdir(os.path.join(dataset_type_dir, 'images'))
        os.mkdir(os.path.join(dataset_type_dir, 'labels'))

    return args


def TestImageLevelDatasetCropAndSplit(args):
    image_data_root_dir = os.path.join(args.src_data_root_dir, 'images')
    label_data_root_dir = os.path.join(args.src_data_root_dir, 'labels')

    assert os.path.exists(image_data_root_dir), 'Source image data root dir does not exist.'
    assert os.path.exists(label_data_root_dir), 'Source label data root dir does not exist.'

    # set up logger
    logger = Logger(args.dst_data_root_dir)

    image_list_training, \
    image_list_val, \
    image_list_test = image_filename_list_split(image_data_root_dir,
                                                training_ratio=args.training_ratio,
                                                validation_ratio=args.validation_ratio,
                                                test_ratio=args.test_ratio,
                                                random_seed=args.random_seed,
                                                logger=logger)

    crop_and_save_data(filename_list=image_list_training, image_dir=image_data_root_dir, label_dir=label_data_root_dir,
                       save_path=args.dst_data_root_dir, dataset_type='training',
                       intensity_threshold=args.intensity_threshold, logger=logger)
    crop_and_save_data(filename_list=image_list_val, image_dir=image_data_root_dir, label_dir=label_data_root_dir,
                       save_path=args.dst_data_root_dir, dataset_type='validation',
                       intensity_threshold=args.intensity_threshold, logger=logger)
    crop_and_save_data(filename_list=image_list_test, image_dir=image_data_root_dir, label_dir=label_data_root_dir,
                       save_path=args.dst_data_root_dir, dataset_type='test',
                       intensity_threshold=args.intensity_threshold, logger=logger)

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestImageLevelDatasetCropAndSplit(args)
