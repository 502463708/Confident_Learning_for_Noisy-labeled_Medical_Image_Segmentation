import copy
import cv2
import numpy as np
import os
import SimpleITK as sitk

from common.utils import generate_uncertainty_maps
from skimage import measure


def generate_coordinate_list(residue_or_mask_np, mode='detected'):
    # mode must be either 'detected' or 'annotated'
    assert mode in ['detected', 'annotated']

    if mode == 'detected':
        # mode: detected -> iterate each connected component on processed_residue_radiograph_np
        mask_np = copy.copy(residue_or_mask_np)
        mask_np[residue_or_mask_np > 0] = 1
    else:
        # mode: annotated -> iterate each connected component on pixel_level_label_np
        mask_np = copy.copy(residue_or_mask_np)
        # remain micro calcifications and normal tissue label only
        mask_np[mask_np > 1] = 0

    # generate information of each connected component
    connected_components = measure.label(mask_np)
    props = measure.regionprops(connected_components)

    # created for saving the coordinates and the detected score for this connected component
    coordinate_list = list()

    if len(props) > 0:
        for prop in props:
            # record the centroid of this connected component
            coordinate_list.append(np.array(prop.centroid))

    return coordinate_list


def merge_coord_list(pred_coord_list, label_coord_list):
    if len(pred_coord_list) == 0 and len(label_coord_list) == 0:
        coord_list = list()
    elif len(pred_coord_list) == 0:
        coord_list = label_coord_list
    elif len(label_coord_list) == 0:
        coord_list = pred_coord_list
    else:
        coord_list = list()
        merged_coord_list = pred_coord_list + label_coord_list

        for coord_1 in merged_coord_list:
            near_num = 0
            for coord_2 in merged_coord_list:
                if np.linalg.norm(coord_1 - coord_2) < 2:
                    near_num += 1
            if near_num < 2:
                coord_list.append(coord_1)

    return coord_list


def generate_legal_indexes(coord, patch_size, height, width):
    # generate legal start and end idx for row and column
    centroid_row_idx = coord[0]
    centroid_column_idx = coord[1]
    #
    centroid_row_idx = np.clip(centroid_row_idx, patch_size[0] / 2, height - patch_size[0] / 2)
    centroid_column_idx = np.clip(centroid_column_idx, patch_size[1] / 2, width - patch_size[1] / 2)
    #
    start_row_idx = int(centroid_row_idx - patch_size[0] / 2)
    end_row_idx = int(centroid_row_idx + patch_size[0] / 2)
    start_column_idx = int(centroid_column_idx - patch_size[1] / 2)
    end_column_idx = int(centroid_column_idx + patch_size[1] / 2)

    assert end_row_idx - start_row_idx == patch_size[0]
    assert end_column_idx - start_column_idx == patch_size[1]

    crop_indexes = [start_row_idx, end_row_idx, start_column_idx, end_column_idx]

    return crop_indexes


def save_images_labels_uncertainty_maps(coord_list, image_tensor, image_np, pixel_level_label_np, net_list, filename,
                                        positive_dataset_type_dir, negative_dataset_type_dir, reconstruction_patch_size,
                                        saving_patch_size):
    height, width = image_np.shape

    positive_patch_idx = 0
    negative_patch_idx = 0

    for coord in coord_list:
        # generate legal start and end idx for row and column
        saving_crop_indexes = generate_legal_indexes(coord, saving_patch_size, height, width)
        reconstruction_crop_indexes = generate_legal_indexes(coord, reconstruction_patch_size, height, width)

        # crop this patch from image and label
        image_patch_np = copy.copy(image_np[saving_crop_indexes[0]:saving_crop_indexes[1],
                                   saving_crop_indexes[2]:saving_crop_indexes[3]])
        pixel_level_label_patch_np = copy.copy(pixel_level_label_np[saving_crop_indexes[0]:saving_crop_indexes[1],
                                               saving_crop_indexes[2]:saving_crop_indexes[3]])
        image_level_label_patch_bool = True if (pixel_level_label_patch_np == 1).sum() > 0 else False

        # MC dropout for uncertainty map
        image_patch_tensor = copy.copy(image_tensor[:, :, reconstruction_crop_indexes[0]:reconstruction_crop_indexes[1],
                                       reconstruction_crop_indexes[2]:reconstruction_crop_indexes[3]])
        uncertainty_map_np = generate_uncertainty_maps(net_list, image_patch_tensor)
        uncertainty_map_np = uncertainty_map_np.squeeze()
        #
        # uncertainty map size 112*112 -> 56*56
        center_coord = [int(reconstruction_patch_size[0] / 2), int(reconstruction_patch_size[1] / 2)]
        center_crop_indexes = generate_legal_indexes(center_coord, saving_patch_size, reconstruction_patch_size[0],
                                                     reconstruction_patch_size[1])
        uncertainty_map_np = uncertainty_map_np[center_crop_indexes[0]: center_crop_indexes[1], center_crop_indexes[2]:
                                                center_crop_indexes[3]]
        uncertainty_map_image = sitk.GetImageFromArray(uncertainty_map_np)

        # transformed into png format
        image_patch_np *= 255
        #
        pixel_level_label_patch_np[pixel_level_label_patch_np == 1] = 255
        pixel_level_label_patch_np[pixel_level_label_patch_np == 2] = 165
        pixel_level_label_patch_np[pixel_level_label_patch_np == 3] = 85
        #
        image_patch_np = image_patch_np.astype(np.uint8)
        pixel_level_label_patch_np = pixel_level_label_patch_np.astype(np.uint8)

        if image_level_label_patch_bool:
            positive_patch_idx += 1
            absolute_image_saving_path = os.path.join(positive_dataset_type_dir,
                                                      'positive_' + filename.split('.')[0] + '_{}.png'.format(
                                                          positive_patch_idx))
            absolute_label_saving_path = absolute_image_saving_path.replace('images', 'labels')
            absolute_uncertainty_map_saving_path = absolute_image_saving_path.replace('images', 'uncertainty-maps')
            absolute_uncertainty_map_saving_path = absolute_uncertainty_map_saving_path.replace('.png', '.nii')
        else:
            negative_patch_idx += 1
            absolute_image_saving_path = os.path.join(negative_dataset_type_dir,
                                                      'negative_' + filename.split('.')[0] + '_{}.png'.format(
                                                          negative_patch_idx))
            absolute_label_saving_path = absolute_image_saving_path.replace('images', 'labels')

            absolute_uncertainty_map_saving_path = absolute_image_saving_path.replace('images', 'uncertainty-maps')
            absolute_uncertainty_map_saving_path = absolute_uncertainty_map_saving_path.replace('.png', '.nii')

        # saving
        cv2.imwrite(absolute_image_saving_path, image_patch_np)
        cv2.imwrite(absolute_label_saving_path, pixel_level_label_patch_np)
        sitk.WriteImage(uncertainty_map_image, absolute_uncertainty_map_saving_path)

    return positive_patch_idx, negative_patch_idx
