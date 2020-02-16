import argparse
import os
import shutil

from logger.logger import Logger
from utils.patch_level_dataset_negative_patch_removal import remove_negative_patches


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-sub-datasets/sub-dataset-2/',
                        help='Source data root dir.')

    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-sub-datasets/sub-dataset-2/',
                        help='Destination data root dir.')

    parser.add_argument('--random_seed',
                        type=int,
                        default=0,
                        help='Set random seed for reduplicating the results.'
                             '-1 -> do not set random seed.')

    args = parser.parse_args()

    assert os.path.exists(args.src_data_root_dir), 'Source data root dir does not exist.'

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)

    for patch_type in ['positive_patches', 'negative_patches']:
        patch_type_dir = os.path.join(args.dst_data_root_dir, patch_type)
        os.mkdir(patch_type_dir)

        for dataset_type in ['training', 'validation', 'test']:
            dataset_type_dir = os.path.join(patch_type_dir, dataset_type)
            os.mkdir(dataset_type_dir)
            os.mkdir(os.path.join(dataset_type_dir, 'images'))
            os.mkdir(os.path.join(dataset_type_dir, 'labels'))

    return args


def TestPatchLevelDatasetNegativePatchRemoval(args):
    # set up logger
    logger = Logger(args.dst_data_root_dir)

    for dataset_type in ['training', 'validation', 'test']:
        logger.write_and_print('--------------------------------------------------------------------------------------')
        logger.write_and_print('Processing {} set'.format(dataset_type))
        positive_image_count, negative_image_count, sampled_negative_image_count = remove_negative_patches(
            args.src_data_root_dir, args.dst_data_root_dir, dataset_type, args.random_seed, logger)

        logger.write_and_print('Finished processing {} set.'.format(dataset_type))
        logger.write_and_print('This dataset has {} positive patches.'.format(positive_image_count))
        logger.write_and_print('This dataset originally had {} negative patches.'.format(negative_image_count))
        logger.write_and_print('This dataset now has {} negative patches.'.format(sampled_negative_image_count))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestPatchLevelDatasetNegativePatchRemoval(args)
