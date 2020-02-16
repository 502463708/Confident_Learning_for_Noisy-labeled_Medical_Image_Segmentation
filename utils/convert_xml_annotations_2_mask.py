import cv2
import numpy as np
import os
import shutil
import SimpleITK as sitk
import xml.etree.ElementTree as ET

from common.utils import get_min_distance
from skimage.draw import polygon
from skimage import measure


class Annotation(object):
    def __init__(self, xml_node):
        children_nodes = xml_node.getchildren()
        self.name = children_nodes[15].text
        self.area = float(children_nodes[1].text)
        self.coordinate_list = self.get_coordinate_list(children_nodes[21])
        self.pixel_number = len(self.coordinate_list)

        return

    def print_details(self):
        print('  name: {}'.format(self.name))
        print('  area: {}'.format(self.area))
        print('  pixel_number: {}'.format(self.pixel_number))
        print('  *****************************************************************************************************')

        return

    def get_coordinate_list(self, coordinate_root_node):
        coordinate_list = list()
        coordinate_child_nodes = coordinate_root_node.getchildren()
        for coordinate_child_node in coordinate_child_nodes:
            pixel_coordinate_text = coordinate_child_node.text
            pixel_coordinate_text = pixel_coordinate_text[1: -1]
            pixel_coordinates_text_list = pixel_coordinate_text.split(',')
            x = round(float(pixel_coordinates_text_list[0]))
            y = round(float(pixel_coordinates_text_list[1]))
            coordinate_list.append([x, y])

        return coordinate_list


class ImageLevelAnnotationCollection(object):
    def __init__(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        animNode = root.find('dict')
        animNode = animNode.find('array')
        animNode = animNode.find('dict')
        annotation_root_node = animNode.find('array')
        annotation_child_nodes = annotation_root_node.findall('dict')
        self.annotation_list = list()
        for annotation_child_node in annotation_child_nodes:
            annotation_obj = Annotation(annotation_child_node)
            self.annotation_list.append(annotation_obj)

        return


def generate_label_from_image(src_data_root_dir, dst_data_root_dir, image_filename):
    absolute_src_image_path = os.path.join(src_data_root_dir, 'images', image_filename)
    absolute_dst_label_path = os.path.join(dst_data_root_dir, 'labels', image_filename)
    absolute_dst_stacked_data_path = os.path.join(dst_data_root_dir, 'stacked_data_in_nii_format',
                                                  image_filename.replace('png', 'nii'))

    assert os.path.exists(absolute_src_image_path)

    image_np = cv2.imread(absolute_src_image_path, cv2.IMREAD_GRAYSCALE)
    label_np = np.zeros_like(image_np, dtype='uint8')

    # background labelling
    label_np[image_np == 0] = 85

    # saving label
    cv2.imwrite(absolute_dst_label_path, label_np)

    # saving stacked data for the debug purpose
    stacked_np = np.concatenate((np.expand_dims(image_np, axis=0), np.expand_dims(label_np, axis=0)), axis=0)
    stacked_image = sitk.GetImageFromArray(stacked_np)
    sitk.WriteImage(stacked_image, absolute_dst_stacked_data_path)

    return


def generate_label_from_xml(src_data_root_dir, dst_data_root_dir, image_filename, calcification_list,
                            other_lesion_list, logger=None, diameter_threshold=14, distance_threshold=112):
    absolute_src_image_path = os.path.join(src_data_root_dir, 'images', image_filename)
    xml_filename = image_filename.replace('png', 'xml')
    absolute_src_xml_path = os.path.join(src_data_root_dir, 'xml_annotations', xml_filename)
    absolute_dst_label_path = os.path.join(dst_data_root_dir, 'labels', image_filename)
    absolute_dst_stacked_data_path = os.path.join(dst_data_root_dir, 'stacked_data_in_nii_format',
                                                  image_filename.replace('png', 'nii'))

    assert os.path.exists(absolute_src_image_path)
    assert os.path.exists(absolute_src_xml_path)

    image_np = cv2.imread(absolute_src_image_path, cv2.IMREAD_GRAYSCALE)
    xml_obj = ImageLevelAnnotationCollection(absolute_src_xml_path)

    label_np = np.zeros_like(image_np)  # column, row
    background_mask_np = np.zeros_like(image_np)  # column, row
    calcification_mask_np = np.zeros_like(image_np)  # column, row
    other_lesion_mask_np = np.zeros_like(image_np)  # column, row

    # background labelling
    background_mask_np[image_np == 0] = 1

    # for statistical purpose
    calcification_count_image_level = 0
    large_calcification_count_image_level = 0
    neighborhood_calcification_count_image_level = 0
    covered_calcification_count_image_level = 0
    micro_calcification_count_image_level = 0
    other_lesion_count_image_level = 0

    # generate calcification_mask_np and other_lesion_mask_np
    for annotation in xml_obj.annotation_list:
        # convert the outline annotation into area annotation
        coordinate_list = np.array(annotation.coordinate_list) - 1
        column_indexes, row_indexes = polygon(coordinate_list[:, 0], coordinate_list[:, 1])

        # for the calcification annotations
        if annotation.name in calcification_list:
            calcification_count_image_level += 1
            # in case that only one pixel is annotated
            if len(row_indexes) == 0:
                for coordinate in coordinate_list:
                    calcification_mask_np[coordinate[1], coordinate[0]] = 1
            else:
                # to avoid the situation that coordinate indexes get out of range
                height, width = image_np.shape
                row_indexes = np.clip(row_indexes, 0, height - 1)
                column_indexes = np.clip(column_indexes, 0, width - 1)
                calcification_mask_np[row_indexes, column_indexes] = 1  # row ,column

        # for the other lesion annotations
        elif annotation.name in other_lesion_list:
            other_lesion_count_image_level += 1
            # in case that only one pixel is annotated
            if len(row_indexes) == 0:
                for coordinate in coordinate_list:
                    other_lesion_mask_np[coordinate[1], coordinate[0]] = 1
            else:
                # to avoid the situation that coordinate indexes get out of range
                height, width = image_np.shape
                row_indexes = np.clip(row_indexes, 0, height - 1)
                column_indexes = np.clip(column_indexes, 0, width - 1)
                other_lesion_mask_np[row_indexes, column_indexes] = 1  # row ,column

    # analyse the connected components for calcification_mask_np
    calcification_connected_components = measure.label(input=calcification_mask_np, connectivity=2)
    calcification_connected_component_props = measure.regionprops(calcification_connected_components)

    for prop in calcification_connected_component_props:
        # a large calcification is gonna be picked up as an outlier calcification
        if prop.major_axis_length >= diameter_threshold:
            large_calcification_count_image_level += 1
            coordinates = prop.coords
            for coordinate in coordinates:
                row_idx = coordinate[0]
                column_idx = coordinate[1]
                calcification_mask_np[row_idx][column_idx] = 0
                other_lesion_mask_np[row_idx][column_idx] = 1

        # a tiny calcification is considered as a qualified calcification
        else:
            micro_calcification_count_image_level += 1

    # a micro calcification which is near by the other lesion area is gonna be picked up as an outlier calcification
    if other_lesion_mask_np.max() > 0:
        # analyse the connected components for calcification_mask_np
        calcification_connected_components = measure.label(input=calcification_mask_np, connectivity=2)
        calcification_connected_component_props = measure.regionprops(calcification_connected_components)

        for prop in calcification_connected_component_props:
            min_distance = get_min_distance(other_lesion_mask_np, prop.centroid)
            # a micro calcification which is near by the other lesion
            # area is gonna be picked up as an outlier calcification
            if min_distance < distance_threshold:
                micro_calcification_count_image_level -= 1
                if min_distance > 0:
                    neighborhood_calcification_count_image_level += 1
                else:
                    covered_calcification_count_image_level += 1

                coordinates = prop.coords
                for coordinate in coordinates:
                    row_idx = coordinate[0]
                    column_idx = coordinate[1]
                    calcification_mask_np[row_idx][column_idx] = 0
                    other_lesion_mask_np[row_idx][column_idx] = 1

    # generate the final label
    label_np[calcification_mask_np == 1] = 255
    label_np[other_lesion_mask_np == 1] = 165
    label_np[background_mask_np == 1] = 85

    # saving label
    cv2.imwrite(absolute_dst_label_path, label_np)

    # saving stacked data for the debug purpose
    stacked_np = np.concatenate((np.expand_dims(image_np, axis=0), np.expand_dims(label_np, axis=0)), axis=0)
    stacked_image = sitk.GetImageFromArray(stacked_np)
    sitk.WriteImage(stacked_image, absolute_dst_stacked_data_path)

    if logger is None:
        print('  This image contains {} calcifications.'.format(calcification_count_image_level))
        print('  This image contains {} large calcifications.'.format(large_calcification_count_image_level))
        print(
            '  This image contains {} neighborhood calcifications.'.format(
                neighborhood_calcification_count_image_level))
        print('  This image contains {} micro calcifications.'.format(micro_calcification_count_image_level))
        print('  This image contains {} other lesions.'.format(other_lesion_count_image_level))
    else:
        logger.write_and_print('  This image contains {} calcifications.'.format(calcification_count_image_level))
        logger.write_and_print(
            '  This image contains {} large calcifications.'.format(large_calcification_count_image_level))
        logger.write_and_print(
            '  This image contains {} neighborhood calcifications.'.format(
                neighborhood_calcification_count_image_level))
        logger.write_and_print(
            '  This image contains {} covered calcifications.'.format(covered_calcification_count_image_level))
        logger.write_and_print(
            '  This image contains {} micro calcifications.'.format(micro_calcification_count_image_level))
        logger.write_and_print('  This image contains {} other lesions.'.format(other_lesion_count_image_level))

    return calcification_count_image_level, large_calcification_count_image_level, \
           neighborhood_calcification_count_image_level, covered_calcification_count_image_level, \
           micro_calcification_count_image_level, other_lesion_count_image_level


def image_with_xml2image_with_mask(src_data_root_dir, dst_data_root_dir, image_filename, calcification_list,
                                   other_lesion_list, diameter_threshold, distance_threshold, logger=None):
    absolute_src_image_path = os.path.join(src_data_root_dir, 'images', image_filename)
    xml_filename = image_filename.replace('png', 'xml')
    absolute_src_xml_path = os.path.join(src_data_root_dir, 'xml_annotations', xml_filename)
    absolute_dst_image_path = os.path.join(dst_data_root_dir, 'images', image_filename)

    # the source image must exist
    assert os.path.exists(absolute_src_image_path)

    # copy the image file into the destination image folder
    shutil.copyfile(absolute_src_image_path, absolute_dst_image_path)

    # for statistical purpose
    calcification_count_image_level = 0
    large_calcification_count_image_level = 0
    neighborhood_calcification_count_image_level = 0
    covered_calcification_count_image_level = 0
    micro_calcification_count_image_level = 0
    other_lesion_count_image_level = 0

    # if this image does not have its corresponding xml file -> generate a mask which is completely filled with 0
    if not os.path.exists(absolute_src_xml_path):
        if logger is None:
            print('This image does not have xml annotation.')
        else:
            logger.write_and_print('This image does not have xml annotation.')

        generate_label_from_image(src_data_root_dir, dst_data_root_dir, image_filename)
    # if this image has its corresponding xml file -> generate a mask according to its xml file
    else:
        if logger is None:
            print('This image has xml annotation.')
        else:
            logger.write_and_print('This image has xml annotation.')

        calcification_count_image_level, large_calcification_count_image_level, \
        neighborhood_calcification_count_image_level, covered_calcification_count_image_level, \
        micro_calcification_count_image_level, other_lesion_count_image_level = generate_label_from_xml(
            src_data_root_dir,
            dst_data_root_dir,
            image_filename,
            calcification_list,
            other_lesion_list,
            logger,
            diameter_threshold,
            distance_threshold)

    return calcification_count_image_level, large_calcification_count_image_level, \
           neighborhood_calcification_count_image_level, covered_calcification_count_image_level, \
           micro_calcification_count_image_level, other_lesion_count_image_level
