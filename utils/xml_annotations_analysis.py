import cv2
import numpy as np
import os
import shutil

from skimage.draw import polygon
from utils.convert_xml_annotations_2_mask import ImageLevelAnnotationCollection


def generate_labels_according_to_xml(absolute_src_image_path, absolute_src_xml_path, output_dir, annotation_name_list):
    image_np = cv2.imread(absolute_src_image_path, cv2.IMREAD_GRAYSCALE)
    xml_obj = ImageLevelAnnotationCollection(absolute_src_xml_path)

    # for saving the annotation name specified mask
    annotation_name_2_mask_dict = dict()

    # generate mask for each annotation genre
    for annotation in xml_obj.annotation_list:
        # only analyse the annotations specified in the annotation_name_list
        if annotation.name in annotation_name_list:
            annotation.print_details()

            # create a new key-value pair in the dict
            existing_annotation_name_list = list(annotation_name_2_mask_dict.keys())
            if annotation.name not in existing_annotation_name_list:
                annotation_name_2_mask_dict[annotation.name] = np.zeros_like(image_np, dtype=np.uint8)  # column, row
            mask_for_this_annotation_np = annotation_name_2_mask_dict[annotation.name]

            # convert the outline annotation into area annotation
            coordinate_list = np.array(annotation.coordinate_list) - 1
            cc, rr = polygon(coordinate_list[:, 0], coordinate_list[:, 1])

            # in case that only one pixel is annotated
            if len(rr) == 0:
                for coordinate in coordinate_list:
                    mask_for_this_annotation_np[coordinate[1], coordinate[0]] = 255
            else:
                # to avoid the situation that coordinate indexes get out of range
                height, width = image_np.shape
                rr = np.clip(rr, 0, height - 1)
                cc = np.clip(cc, 0, width - 1)

                mask_for_this_annotation_np[rr, cc] = 255  # row ,column

    existing_annotation_name_list = list(annotation_name_2_mask_dict.keys())
    _, image_filename = os.path.split(absolute_src_image_path)
    label_filename = image_filename.replace('.png', '_mask.png')
    for existing_annotation_name in existing_annotation_name_list:
        mask_for_this_annotation_np = annotation_name_2_mask_dict[existing_annotation_name]

        annotation_name_specified_output_dir = os.path.join(output_dir, existing_annotation_name)

        absolute_dst_image_path = os.path.join(annotation_name_specified_output_dir, image_filename)
        shutil.copyfile(absolute_src_image_path, absolute_dst_image_path)

        absolute_dst_label_path = os.path.join(annotation_name_specified_output_dir, label_filename)
        cv2.imwrite(absolute_dst_label_path, mask_for_this_annotation_np)

    return
