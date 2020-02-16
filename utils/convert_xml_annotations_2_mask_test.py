import argparse
import os
import shutil

from logger.logger import Logger
from utils.convert_xml_annotations_2_mask import image_with_xml2image_with_mask


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-radiograph-level-raw-images-with-XML-annotations-dataset/',
                        help='Source data root dir.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-20191105/Inbreast-radiograph-level-raw-images-with-pixel-level-labels-dataset/',
                        help='Destination data root dir.')
    parser.add_argument('--calcification_list',
                        type=list,
                        default=['Calcification', 'Calcifications', 'Unnamed', 'Point 1', 'Point 3'],
                        help='The included elements are gonna be labeled as calcifications.')
    parser.add_argument('--other_lesion_list',
                        type=list,
                        default=['Cluster', 'Espiculated Region', 'Assymetry', 'Spiculated Region', 'Distortion',
                                 'Spiculated region', 'Mass', 'Asymmetry'],
                        # default=[],
                        help='The included elements are gonna be labeled as other lesion.')
    parser.add_argument('--diameter_threshold',
                        type=float,
                        default=14,
                        help='The calcifications whose diameter >= diameter_threshold will be discarded.')
    parser.add_argument('--distance_threshold',
                        type=int,
                        default=21,
                        # default=-1,
                        help='The distance_threshold for picking up micro calcifications nearby the other lesion area.')

    args = parser.parse_args()

    # the source data root dir must exist
    assert os.path.exists(args.src_data_root_dir), 'Source data root dir does not exist.'

    # remove the destination data root dir if it already exists
    parent_dir = os.path.dirname(args.dst_data_root_dir) if args.dst_data_root_dir[-1] != '/' else os.path.dirname(
        args.dst_data_root_dir[:-1])
    if os.path.exists(parent_dir):
        shutil.rmtree(parent_dir)

    # create brand new destination data root dir
    os.mkdir(parent_dir)
    os.mkdir(args.dst_data_root_dir)
    os.mkdir(os.path.join(args.dst_data_root_dir, 'images'))
    os.mkdir(os.path.join(args.dst_data_root_dir, 'labels'))
    os.mkdir(os.path.join(args.dst_data_root_dir, 'stacked_data_in_nii_format'))

    return args


def TestConvertXml2Mask(args):
    src_image_dir = os.path.join(args.src_data_root_dir, 'images')
    src_xml_dir = os.path.join(args.src_data_root_dir, 'xml_annotations')

    # the source data root dir must contain images and labels
    assert os.path.exists(src_image_dir)
    assert os.path.exists(src_xml_dir)

    # set up logger
    logger = Logger(args.dst_data_root_dir)

    image_filename_list = os.listdir(src_image_dir)

    # for statistical purpose
    calcification_count_dataset_level = 0
    large_calcification_count_dataset_level = 0
    neighborhood_calcification_count_dataset_level = 0
    covered_calcification_count_dataset_level = 0
    micro_calcification_count_dataset_level = 0
    other_lesion_count_dataset_level = 0

    current_idx = 0
    for image_filename in image_filename_list:
        current_idx += 1
        logger.write_and_print('--------------------------------------------------------------------------------------')
        logger.write_and_print(
            'Processing {} out of {}, filename: {}'.format(current_idx, len(image_filename_list), image_filename))

        calcification_count_image_level, large_calcification_count_image_level, \
        neighborhood_calcification_count_image_level, covered_calcification_count_image_level, \
        micro_calcification_count_image_level, other_lesion_count_image_level = image_with_xml2image_with_mask(
            args.src_data_root_dir,
            args.dst_data_root_dir,
            image_filename,
            args.calcification_list,
            args.other_lesion_list,
            args.diameter_threshold,
            args.distance_threshold,
            logger=logger)

        calcification_count_dataset_level += calcification_count_image_level
        large_calcification_count_dataset_level += large_calcification_count_image_level
        neighborhood_calcification_count_dataset_level += neighborhood_calcification_count_image_level
        covered_calcification_count_dataset_level += covered_calcification_count_image_level
        micro_calcification_count_dataset_level += micro_calcification_count_image_level
        other_lesion_count_dataset_level += other_lesion_count_image_level

    logger.write_and_print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    logger.write_and_print('This dataset contains {} calcifications.'.format(calcification_count_dataset_level))
    logger.write_and_print(
        'This dataset contains {} large calcifications.'.format(large_calcification_count_dataset_level))
    logger.write_and_print(
        'This dataset contains {} neighborhood calcifications.'.format(neighborhood_calcification_count_dataset_level))
    logger.write_and_print(
        'This dataset contains {} covered calcifications.'.format(covered_calcification_count_dataset_level))
    logger.write_and_print(
        'This dataset contains {} micro calcifications.'.format(micro_calcification_count_dataset_level))
    logger.write_and_print('This dataset contains {} other lesions.'.format(other_lesion_count_dataset_level))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestConvertXml2Mask(args)
