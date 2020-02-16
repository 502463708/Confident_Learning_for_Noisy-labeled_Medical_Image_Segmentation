import argparse
import os
import shutil

from logger.logger import Logger
from utils.patch_level_sub_dataset_generation import get_sub_dataset_filename_list, copy_data_from_src_2_dst


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_patch_level_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/',
                        help='Source dir of patch-level pos-2-neg ratio 1:1 dataset.')
    parser.add_argument('--src_radiograph_level_sub_datasets_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-radiograph-level-roi-extracted-data-split-sub-datasets/',
                        help='Source dir of radiograph-level sub datasets.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-sub-datasets/',
                        help='Destination data dir.')

    args = parser.parse_args()

    assert os.path.exists(args.src_patch_level_data_root_dir), \
        'Source data root dir does not exist: {}.'.format(args.src_patch_level_data_root_dir)
    assert os.path.exists(args.src_radiograph_level_sub_datasets_root_dir), \
        'Source data root dir does not exist: {}.'.format(args.src_radiograph_level_sub_datasets_root_dir)

    if os.path.exists(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)
    os.mkdir(args.dst_data_root_dir)

    # calculate sub_dataset_number based on src_radiograph_level_sub_datasets_root_dir
    sub_dataset_number = len([lists for lists in os.listdir(args.src_radiograph_level_sub_datasets_root_dir) if
                              os.path.isdir(os.path.join(args.src_radiograph_level_sub_datasets_root_dir, lists))])

    for sub_dataset_idx in range(sub_dataset_number):
        sub_dataset_name = 'sub-dataset-{}'.format(sub_dataset_idx + 1)
        sub_dataset_dir = os.path.join(args.dst_data_root_dir, sub_dataset_name)
        os.mkdir(sub_dataset_dir)

        for patch_type in ['positive_patches', 'negative_patches']:
            patch_type_dir = os.path.join(sub_dataset_dir, patch_type)
            os.mkdir(patch_type_dir)

            for dataset_type in ['training', 'validation', 'test']:
                dataset_type_dir = os.path.join(patch_type_dir, dataset_type)
                os.mkdir(dataset_type_dir)
                os.mkdir(os.path.join(dataset_type_dir, 'images'))
                os.mkdir(os.path.join(dataset_type_dir, 'labels'))

    return args


def TestPatchLevelSubDatasetGeneration(args):
    # set up logger
    logger = Logger(args.dst_data_root_dir)

    # calculate sub_dataset_number based on src_radiograph_level_sub_datasets_root_dir
    sub_dataset_number = len([lists for lists in os.listdir(args.src_radiograph_level_sub_datasets_root_dir) if
                              os.path.isdir(os.path.join(args.src_radiograph_level_sub_datasets_root_dir, lists))])

    for patch_type in ['positive_patches', 'negative_patches']:

        for dataset_type in ['training', 'validation', 'test']:

            for sub_dataset_idx in range(sub_dataset_number):
                sub_dataset_name = 'sub-dataset-{}'.format(sub_dataset_idx + 1)

                src_patch_level_dataset_type_dir = os.path.join(args.src_patch_level_data_root_dir, patch_type,
                                                                dataset_type)
                src_radiograph_level_dataset_type_dir = os.path.join(args.src_radiograph_level_sub_datasets_root_dir,
                                                                     sub_dataset_name, dataset_type)
                dst_patch_level_dataset_type_dir = os.path.join(args.dst_data_root_dir, sub_dataset_name, patch_type,
                                                                dataset_type)

                sub_patch_level_filename_list = get_sub_dataset_filename_list(src_patch_level_dataset_type_dir,
                                                                              src_radiograph_level_dataset_type_dir,
                                                                              patch_type, dataset_type, sub_dataset_idx,
                                                                              logger=None)

                copy_data_from_src_2_dst(src_patch_level_dataset_type_dir, dst_patch_level_dataset_type_dir,
                                         sub_patch_level_filename_list, sub_dataset_idx,
                                         dataset_type, patch_type, logger=logger)
    return


if __name__ == '__main__':
    args = ParseArguments()

    TestPatchLevelSubDatasetGeneration(args)
