import argparse
import os
import shutil

from utils.xml_annotations_analysis import generate_labels_according_to_xml


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-raw-data-with-XML-annotations/',
                        help='Source data root dir.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='/data/lars/results/Inbreast-raw-data-with-XML-annotations_analysis/',
                        help='Output dir.')
    parser.add_argument('--annotation_name_list',
                        type=list,
                        default=['Cluster', 'Calcification', 'Point 3', 'Espiculated Region', 'Assymetry', 'Unnamed',
                                 'Spiculated Region', 'Distortion', 'Spiculated region', 'Mass', 'Asymmetry',
                                 'Calcifications', 'Point 1'],
                        help='The name list of the annotations to be analysed.')

    args = parser.parse_args()

    # the source data root dir must exist
    assert os.path.exists(args.src_data_root_dir), 'Source data root dir does not exist.'

    # remove the output data root dir if it already exists
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    # create brand new output data root dir
    os.mkdir(args.output_dir)

    for annotation_name in args.annotation_name_list:
        os.mkdir(os.path.join(args.output_dir, annotation_name))

    return args


def TestXmlAnnotationsAnalysis(args):
    src_image_dir = os.path.join(args.src_data_root_dir, 'images')
    src_xml_dir = os.path.join(args.src_data_root_dir, 'xml_annotations')

    # the source data root dir must contain images and labels
    assert os.path.exists(src_image_dir)
    assert os.path.exists(src_xml_dir)

    xml_filename_list = os.listdir(src_xml_dir)

    current_idx = 0
    for xml_filename in xml_filename_list:
        current_idx += 1
        print('-------------------------------------------------------------------------------------------------------')
        print('Processing {} out of {}, filename: {}'.format(current_idx, len(xml_filename_list), xml_filename))

        image_filename = xml_filename.replace('xml', 'png')

        absolute_src_image_path = os.path.join(src_image_dir, image_filename)
        absolute_src_xml_path = os.path.join(src_xml_dir, xml_filename)

        generate_labels_according_to_xml(absolute_src_image_path, absolute_src_xml_path, args.output_dir,
                                         args.annotation_name_list)

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestXmlAnnotationsAnalysis(args)
