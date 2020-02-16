import copy
import cv2
import numpy as np
import os
import torch

from skimage import measure


def dilate_image_level_label(image_level_label, dilation_radius):
    """
    This function implements the dilation for mask
    :param image_level_label:
    :param dilation_radius:
    :return:
    """
    assert dilation_radius > 0

    dilation_diameter = 2 * dilation_radius + 1
    kernel = np.zeros((dilation_diameter, dilation_diameter), np.uint8)

    for row_idx in range(dilation_diameter):
        for column_idx in range(dilation_diameter):
            if np.linalg.norm(np.array([row_idx, column_idx]) - np.array(
                    [dilation_radius, dilation_radius])) <= dilation_radius:
                kernel[row_idx, column_idx] = 1

    dilated_image_level_label = cv2.dilate(image_level_label, kernel, iterations=1)

    assert dilated_image_level_label.shape == image_level_label.shape

    return dilated_image_level_label


def post_process_residue(image_np, prob_threshold, area_threshold):
    """
    This function implements the post-process for the residue, including
    the following 2 steps:
        1. transform the residue into binary mask based on prob_threshold
        2. discard connected components whose area < area_threshold
    :param image_np: residue
    :param prob_threshold:
    :param area_threshold:
    :return:
    """
    assert len(image_np.shape) == 2

    image_np[image_np <= prob_threshold] = 0
    image_np[image_np > prob_threshold] = 1

    connected_components = measure.label(image_np, connectivity=2)

    props = measure.regionprops(connected_components)

    connected_component_num = len(props)

    if connected_component_num > 0:
        for connected_component_idx in range(connected_component_num):
            if props[connected_component_idx].area < area_threshold:
                connected_components[connected_components == connected_component_idx + 1] = 0

    post_processed_image_np = np.zeros_like(image_np)
    post_processed_image_np[connected_components != 0] = 1

    return post_processed_image_np


def get_ckpt_path(model_saving_dir, epoch_idx=-1):
    """
    Given a dir (where the model is saved) and an index (which ckpt is specified),
    This function returns the absolute ckpt path
    :param model_saving_dir:
    :param epoch_idx:
        default mode: epoch_idx = -1 -> return the best ckpt
        specified mode: epoch_idx >= 0 -> return the specified ckpt
    :return: absolute ckpt path
    """
    assert os.path.exists(model_saving_dir)
    assert epoch_idx >= -1

    ckpt_dir = os.path.join(model_saving_dir, 'ckpt')
    # specified mode: epoch_idx is specified -> load the specified ckpt
    if epoch_idx >= 0:
        ckpt_path = os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(epoch_idx))
    # default mode: epoch_idx is not specified -> load the best ckpt
    else:
        saved_ckpt_list = os.listdir(ckpt_dir)
        best_ckpt_filename = [best_ckpt_filename for best_ckpt_filename in saved_ckpt_list if
                              'net_best_on_validation_set' in best_ckpt_filename][0]
        ckpt_path = os.path.join(ckpt_dir, best_ckpt_filename)

    return ckpt_path


def save_best_ckpt(metrics, net, ckpt_dir, epoch_idx):
    """
    This function can discriminatively save this ckpt in case that it is the currently best ckpt on validation set
    :param metrics:
    :param net:
    :param ckpt_dir:
    :param epoch_idx:
    :return:
    """

    is_best_ckpt = metrics.determine_saving_metric_on_validation_list[-1] == max(
        metrics.determine_saving_metric_on_validation_list)
    if is_best_ckpt:
        # firstly remove the last saved best ckpt
        saved_ckpt_list = os.listdir(ckpt_dir)
        for saved_ckpt_filename in saved_ckpt_list:
            if 'net_best_on_validation_set' in saved_ckpt_filename:
                os.remove(os.path.join(ckpt_dir, saved_ckpt_filename))

        # save the current best ckpt
        torch.save(net.state_dict(),
                   os.path.join(ckpt_dir, 'net_best_on_validation_set_epoch_{}.pth'.format(epoch_idx)))

    return


def extract_classification_preds_channel(preds, channel_idx, use_softmax=True, keep_dim=True):
    """

    :param preds:
    :param channel_idx:
    :param use_softmax:
    :param keep_dim:
    :return:
    """
    assert torch.is_tensor(preds)
    assert len(preds.shape) == 4
    assert channel_idx >= 0

    if use_softmax:
        preds = torch.softmax(preds, dim=1)

    extracted_preds = preds[:, channel_idx, :, :]

    if keep_dim:
        extracted_preds = extracted_preds.unsqueeze(dim=1)
        assert len(preds.shape) == len(extracted_preds.shape)

    return extracted_preds


def BatchImageToNumber(Batch_tensor):
    Batch_np = Batch_tensor.numpy()
    print(Batch_tensor.shape)
    assert len(Batch_tensor.shape) == 3  # B*H*W
    Batch_out_list=[]
    for i in range(Batch_np.shape[0]):
        img = Batch_np[i, :, :]
        label_connected_components = measure.label(img, connectivity=2)
        label_props = measure.regionprops(label_connected_components)
        label_num = list(len(label_props))
        Batch_out_list.append(label_num)
    Batch_out_np= np.array(Batch_out_list)
    Batch_out_tensor=torch.from_numpy(Batch_out_np)
    return Batch_out_tensor


def get_min_distance(mask, coordinate):
    # mask must be a 2D binary mask
    assert np.where(mask == 0)[0].shape[0] + np.where(mask == 1)[0].shape[0] == mask.size
    assert len(mask.shape) == 2

    # coordinate must be 2 dimensions
    assert len(coordinate) == 2
    assert isinstance(coordinate, np.ndarray) or isinstance(coordinate, tuple)
    if isinstance(coordinate, tuple):
        coordinate = np.array(coordinate)

    # return None when there is no foreground existing in the source mask
    min_distance = None

    # calculate the minimum distance between coordinate and the pixels marked 1 on mask
    if np.where(mask == 1)[0].shape[0] > 0:
        # generate two matrix for row and column indexes respectively
        idx_indicated_matrix = np.indices(mask.shape)
        row_idx_indicated_matrix = idx_indicated_matrix[0]
        column_idx_indicated_matrix = idx_indicated_matrix[1]

        # generate distance matrix
        distance_matrix = np.sqrt(
            np.square(row_idx_indicated_matrix - coordinate[0]) + np.square(
                column_idx_indicated_matrix - coordinate[1]))

        # get the min distance
        min_distance = distance_matrix[mask == 1].min()

    return min_distance


def get_net_list(network, ckpt_dir, mc_epoch_indexes, logger):
    assert len(mc_epoch_indexes) > 0

    net_list = list()

    for mc_epoch_idx in mc_epoch_indexes:
        net = copy.deepcopy(network)
        ckpt_path = os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(mc_epoch_idx))
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load(ckpt_path))
        net = net.eval()
        net_list.append(net)

        logger.write_and_print('Load ckpt: {0} for MC dropout...'.format(ckpt_path))

    return net_list


def generate_uncertainty_maps(net_list, images_tensor):
    accumulated_residues_tensor = None

    for net in net_list:
        _, prediction_residues_tensor = net(images_tensor)
        if accumulated_residues_tensor is None:
            accumulated_residues_tensor = prediction_residues_tensor
        else:
            accumulated_residues_tensor = torch.cat((accumulated_residues_tensor, prediction_residues_tensor), dim=1)

    uncertainty_maps = accumulated_residues_tensor.var(dim=1)
    uncertainty_maps_np = uncertainty_maps.cpu().detach().numpy()
    uncertainty_maps_np = 1 - np.exp(-uncertainty_maps_np)

    return uncertainty_maps_np


def generate_radiograph_level_reconstructed_and_residue_result(images_tensor, reconstruction_net, pixel_level_label_np,
                                                               patch_size, stride):
    if images_tensor.device.type == 'cpu':
        images_tensor = images_tensor.cuda()

    assert stride > 0, 'The variable stride must be a positive integer.'

    height = images_tensor.shape[2]
    width = images_tensor.shape[3]

    reconstructed_radiograph = np.zeros((height, width))
    residue_radiograph = np.zeros((height, width))
    counting_mask = np.zeros((height, width))

    start_row_idx = -1
    end_row_idx = -1
    while end_row_idx < height:
        if start_row_idx == -1:
            start_row_idx = 0
            end_row_idx = start_row_idx + patch_size[0]
        else:
            start_row_idx += stride
            end_row_idx += stride
        if end_row_idx > height:
            gap_row = end_row_idx - height
            end_row_idx -= gap_row
            start_row_idx -= gap_row

        start_column_idx = -1
        end_column_idx = -1
        while end_column_idx < width:
            if start_column_idx == -1:
                start_column_idx = 0
                end_column_idx = start_column_idx + patch_size[1]
            else:
                start_column_idx += stride
                end_column_idx += stride
            if end_column_idx > width:
                gap_column = end_column_idx - width
                end_column_idx -= gap_column
                start_column_idx -= gap_column

            # debug only
            # print(
            #     'row idx range: {} - {} (height = {}), column idx range: {} - {} (width = {})'.format(start_row_idx,
            #                                                                                           end_row_idx,
            #                                                                                           height,
            #                                                                                           start_column_idx,
            #                                                                                           end_column_idx,
            #                                                                                           width))

            # generate the current patch label
            patch_label_np = pixel_level_label_np[start_row_idx:end_row_idx, start_column_idx:end_column_idx]

            # this patch is totally occupied with outlier calcification -> skip
            if np.where(patch_label_np == 2)[0].shape[0] == patch_label_np.size:
                continue
            # this patch is totally occupied with background -> skip
            if np.where(patch_label_np == 3)[0].shape[0] == patch_label_np.size:
                continue
            # this patch is totally occupied with outlier calcification and background -> skip
            if np.where(patch_label_np == 2)[0].shape[0] + np.where(patch_label_np == 3)[0].shape[0] \
                    == patch_label_np.size:
                continue

            # crop this patch for model inference
            patch_image_tensor = images_tensor[:, :, start_row_idx:end_row_idx, start_column_idx:end_column_idx]

            # generate the current reconstructed patch and residue patch
            reconstructed_patch_tensor, residue_patch_tensor = reconstruction_net(patch_image_tensor)
            reconstructed_current_patch = reconstructed_patch_tensor.cpu().detach().numpy().squeeze()
            residue_current_patch = residue_patch_tensor.cpu().detach().numpy().squeeze()

            # get the existing reconstructed patch and residue patch
            reconstructed_existing_patch = reconstructed_radiograph[start_row_idx:end_row_idx,
                                           start_column_idx:end_column_idx]
            residue_existing_patch = residue_radiograph[start_row_idx:end_row_idx,
                                     start_column_idx:end_column_idx]

            # get the counting patch and weight matrix
            counting_mask_patch = counting_mask[start_row_idx:end_row_idx, start_column_idx:end_column_idx]
            current_weight_matrix = 1 / (counting_mask_patch + 1)
            existing_weight_matrix = 1 - current_weight_matrix

            # update the existing reconstructed radiopgraph and residue radiopgraph
            reconstructed_radiograph[start_row_idx:end_row_idx, start_column_idx:end_column_idx] = \
                reconstructed_existing_patch * existing_weight_matrix + \
                reconstructed_current_patch * current_weight_matrix
            residue_radiograph[start_row_idx:end_row_idx, start_column_idx:end_column_idx] = \
                residue_existing_patch * existing_weight_matrix + \
                residue_current_patch * current_weight_matrix

            # update the counting mask
            counting_mask_patch += 1
            counting_mask[start_row_idx:end_row_idx, start_column_idx:end_column_idx] = counting_mask_patch

    assert reconstructed_radiograph.shape == residue_radiograph.shape

    return reconstructed_radiograph, residue_radiograph


def post_process_residue_radiograph(raw_residue_radiograph_np, pixel_level_label_np, prob_threshold, area_threshold,
                                    remove_overlapped_connected_component=True):
    assert raw_residue_radiograph_np.shape == pixel_level_label_np.shape

    # created for labelling connected components
    residue_mask_radiograph_np = np.zeros_like(raw_residue_radiograph_np)

    # created for saving the post processed residue
    processed_residue_radiograph_np = copy.copy(raw_residue_radiograph_np)

    # only pixels with residue value >= prob_threshold can be remained
    processed_residue_radiograph_np[processed_residue_radiograph_np < prob_threshold] = 0

    # generate information for each connected component on processed_residue_radiograph_np
    residue_mask_radiograph_np[processed_residue_radiograph_np > 0] = 1
    connected_components = measure.label(residue_mask_radiograph_np)
    props = measure.regionprops(connected_components)

    # iterate each connected component
    connected_idx = 0
    for prop in props:
        connected_idx += 1
        indexes = connected_components == connected_idx
        # remove the detected results with area smaller than area_threshold
        if prop.area < area_threshold:
            processed_residue_radiograph_np[indexes] = 0
        if remove_overlapped_connected_component:
            # remove the detected results overlapped with background or other lesion
            if pixel_level_label_np[indexes].max() > 1:
                processed_residue_radiograph_np[indexes] = 0

    processed_residue_radiograph_np[pixel_level_label_np == 2] = 0
    processed_residue_radiograph_np[pixel_level_label_np == 3] = 0

    return processed_residue_radiograph_np
