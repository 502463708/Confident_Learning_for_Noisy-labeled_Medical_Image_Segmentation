import argparse
import copy
import cv2
import numpy as np
import os
import shutil
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn

from common.utils import get_net_list, generate_uncertainty_maps
from config.config_micro_calcification_patch_level_reconstruction import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from logger.logger import Logger
from metrics.metrics_patch_level_reconstruction import MetricsReconstruction
from net.vnet2d_v2 import VNet2d
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/',
                        help='Source data dir.')
    parser.add_argument('--model_saving_dir',
                        type=str,
                        default='/data/lars/models/20191108_5764_uCs_patch_level_reconstruction_ttestlossv3_default_dilation_radius_7/',
                        help='Model saved dir.')
    parser.add_argument('--epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--mc_epoch_indexes',
                        type=int,
                        default=[410, 420, 430, 440, 450, 460, 470, 480, 490, 500],
                        help='The epoch ckpt index list for generating uncertainty maps'
                             'set null list [] to switch off.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='test',
                        help='The type of dataset to be evaluated (training, validation, test).')
    parser.add_argument('--dilation_radius',
                        type=int,
                        default=7,
                        help='The specified dilation_radius when training.')
    parser.add_argument('--prob_threshold',
                        type=float,
                        default=0.1,
                        help='residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1')
    parser.add_argument('--area_threshold',
                        type=float,
                        default=3.14 * 7 * 7 / 4,
                        help='Connected components whose area < area_threshold will be discarded.')
    parser.add_argument('--distance_threshold',
                        type=int,
                        default=14,
                        help='Candidates whose distance between calcification < distance_threshold is a recalled one.')
    parser.add_argument('--slack_for_recall',
                        type=bool,
                        default=True,
                        help='The bool variable for slacking recall metric standard.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=48,
                        help='Batch size for evaluation.')
    parser.add_argument('--save_nii',
                        type=bool,
                        default=False,
                        help='A bool variable indicating whether nii format data is gonna be saved.')

    args = parser.parse_args()

    return args


def save_tensor_in_png_and_nii_format(images_tensor, reconstructed_images_tensor, prediction_residues_tensor,
                                      post_process_preds_np, pixel_level_labels_tensor,
                                      pixel_level_labels_dilated_tensor, uncertainty_maps_np, filenames,
                                      result_flag_list, prediction_saving_dir, save_nii=False):
    # convert tensor into numpy and squeeze channel dimension
    images_np = images_tensor.cpu().detach().numpy().squeeze(axis=1)
    reconstructed_images_np = reconstructed_images_tensor.cpu().detach().numpy().squeeze(axis=1)
    prediction_residues_np = prediction_residues_tensor.cpu().detach().numpy().squeeze(axis=1)
    pixel_level_labels_np = pixel_level_labels_tensor.numpy()
    pixel_level_labels_dilated_np = pixel_level_labels_dilated_tensor.numpy()

    save_uncertainty_maps = False if uncertainty_maps_np is None else True

    # iterating each image of this batch
    for idx in range(images_np.shape[0]):
        image_np = images_np[idx, :, :]
        reconstructed_image_np = reconstructed_images_np[idx, :, :]
        prediction_residue_np = prediction_residues_np[idx, :, :]
        post_process_pred_np = post_process_preds_np[idx, :, :]
        pixel_level_label_np = pixel_level_labels_np[idx, :, :]
        pixel_level_label_dilated_np = pixel_level_labels_dilated_np[idx, :, :]
        if save_uncertainty_maps:
            uncertainty_map_np = uncertainty_maps_np[idx, :, :]

        filename = filenames[idx]

        if save_uncertainty_maps:
            stacked_np = np.concatenate((np.expand_dims(image_np, axis=0),
                                         np.expand_dims(reconstructed_image_np, axis=0),
                                         np.expand_dims(prediction_residue_np, axis=0),
                                         np.expand_dims(post_process_pred_np, axis=0),
                                         np.expand_dims(pixel_level_label_np, axis=0),
                                         np.expand_dims(pixel_level_label_dilated_np, axis=0),
                                         np.expand_dims(uncertainty_map_np, axis=0)), axis=0)
        else:
            stacked_np = np.concatenate((np.expand_dims(image_np, axis=0),
                                         np.expand_dims(reconstructed_image_np, axis=0),
                                         np.expand_dims(prediction_residue_np, axis=0),
                                         np.expand_dims(post_process_pred_np, axis=0),
                                         np.expand_dims(pixel_level_label_np, axis=0),
                                         np.expand_dims(pixel_level_label_dilated_np, axis=0)), axis=0)

        stacked_image = sitk.GetImageFromArray(stacked_np)

        image_np *= 255
        reconstructed_image_np *= 255
        prediction_residue_np *= 255
        post_process_pred_np *= 255
        pixel_level_label_np *= 255
        pixel_level_label_dilated_np *= 255
        if save_uncertainty_maps:
            uncertainty_map_np *= 4 * 255

        image_np = image_np.astype(np.uint8)
        reconstructed_image_np = reconstructed_image_np.astype(np.uint8)
        prediction_residue_np = prediction_residue_np.astype(np.uint8)
        post_process_pred_np = post_process_pred_np.astype(np.uint8)
        pixel_level_label_np = pixel_level_label_np.astype(np.uint8)
        pixel_level_label_dilated_np = pixel_level_label_dilated_np.astype(np.uint8)
        if save_uncertainty_maps:
            uncertainty_map_np = uncertainty_map_np.astype(np.uint8)
            uncertainty_map_np = cv2.applyColorMap(uncertainty_map_np, cv2.COLORMAP_JET)

        result_flag_2_class_mapping = {0: 'TPs_only', 1: 'FPs_only', 2: 'FNs_only', 3: 'FPs_FNs_both', }
        saving_class = result_flag_2_class_mapping[result_flag_list[idx]]

        if save_nii:
            sitk.WriteImage(stacked_image, os.path.join(prediction_saving_dir, saving_class,
                                                        filename.replace('png', 'nii')))

        cv2.imwrite(os.path.join(prediction_saving_dir, saving_class,
                                 filename.replace('.png', '_image.png')), image_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, saving_class,
                                 filename.replace('.png', '_reconstructed.png')), reconstructed_image_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, saving_class,
                                 filename.replace('.png', '_residue.png')), prediction_residue_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, saving_class,
                                 filename.replace('.png', '_post_processed_residue.png')), post_process_pred_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, saving_class,
                                 filename.replace('.png', '_pixel_level_label.png')), pixel_level_label_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, saving_class,
                                 filename.replace('.png', '_pixel_level_dilated_label.png')),
                    pixel_level_label_dilated_np)
        if save_uncertainty_maps:
            cv2.imwrite(os.path.join(prediction_saving_dir, saving_class,
                                     filename.replace('.png', '_uncertainty_map.png')), uncertainty_map_np)

    return


def TestMicroCalcificationReconstruction(args):
    prediction_saving_dir = os.path.join(args.model_saving_dir,
                                         'reconstruction_results_dataset_{}_epoch_{}'.format(args.dataset_type,
                                                                                             args.epoch_idx))
    visualization_saving_dir = os.path.join(prediction_saving_dir, 'qualitative_results')
    visualization_TP_saving_dir = os.path.join(visualization_saving_dir, 'TPs_only')
    visualization_FP_saving_dir = os.path.join(visualization_saving_dir, 'FPs_only')
    visualization_FN_saving_dir = os.path.join(visualization_saving_dir, 'FNs_only')
    visualization_FP_FN_saving_dir = os.path.join(visualization_saving_dir, 'FPs_FNs_both')

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(prediction_saving_dir):
        shutil.rmtree(prediction_saving_dir)
    os.mkdir(prediction_saving_dir)
    os.mkdir(visualization_saving_dir)
    os.mkdir(visualization_TP_saving_dir)
    os.mkdir(visualization_FP_saving_dir)
    os.mkdir(visualization_FN_saving_dir)
    os.mkdir(visualization_FP_FN_saving_dir)

    # initialize logger
    logger = Logger(prediction_saving_dir, 'quantitative_results.txt')
    logger.write_and_print('Dataset: {}'.format(args.data_root_dir))
    logger.write_and_print('Dataset type: {}'.format(args.dataset_type))

    # define the network
    network = VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)

    # load the specified ckpt
    ckpt_dir = os.path.join(args.model_saving_dir, 'ckpt')
    # epoch_idx is specified -> load the specified ckpt
    if args.epoch_idx >= 0:
        ckpt_path = os.path.join(ckpt_dir, 'net_epoch_{}.pth'.format(args.epoch_idx))
    # epoch_idx is not specified -> load the best ckpt
    else:
        saved_ckpt_list = os.listdir(ckpt_dir)
        best_ckpt_filename = [best_ckpt_filename for best_ckpt_filename in saved_ckpt_list if
                              'net_best_on_validation_set' in best_ckpt_filename][0]
        ckpt_path = os.path.join(ckpt_dir, best_ckpt_filename)

    # transfer net into gpu devices
    net = copy.deepcopy(network)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(ckpt_path))
    net = net.eval()

    logger.write_and_print('Load ckpt: {0} for evaluating...'.format(ckpt_path))

    # get calculate_uncertainty global variance
    calculate_uncertainty = True if len(args.mc_epoch_indexes) > 0 else False

    # get net list for imitating MC dropout process
    net_list = None
    if calculate_uncertainty:
        net_list = get_net_list(network, ckpt_dir, args.mc_epoch_indexes, logger)

    # create dataset
    dataset = MicroCalcificationDataset(data_root_dir=args.data_root_dir,
                                        mode=args.dataset_type,
                                        enable_random_sampling=False,
                                        pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                        image_channels=cfg.dataset.image_channels,
                                        cropping_size=cfg.dataset.cropping_size,
                                        dilation_radius=args.dilation_radius,
                                        load_uncertainty_map=False,
                                        calculate_micro_calcification_number=cfg.dataset.calculate_micro_calcification_number,
                                        enable_data_augmentation=False)

    # create data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=cfg.train.num_threads)

    metrics = MetricsReconstruction(args.prob_threshold, args.area_threshold, args.distance_threshold,
                                    args.slack_for_recall)

    calcification_num = 0
    recall_num = 0
    FP_num = 0

    for batch_idx, (images_tensor, pixel_level_labels_tensor, pixel_level_labels_dilated_tensor, _,
                    image_level_labels_tensor, _, filenames) in enumerate(data_loader):
        logger.write_and_print('Evaluating batch: {}'.format(batch_idx))

        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # network forward
        reconstructed_images_tensor, prediction_residues_tensor = net(images_tensor)

        # MC dropout
        uncertainty_maps_np = generate_uncertainty_maps(net_list, images_tensor) if calculate_uncertainty else None

        # evaluation
        post_process_preds_np, calcification_num_batch_level, recall_num_batch_level, FP_num_batch_level, \
        result_flag_list = metrics.metric_batch_level(prediction_residues_tensor, pixel_level_labels_tensor)

        calcification_num += calcification_num_batch_level
        recall_num += recall_num_batch_level
        FP_num += FP_num_batch_level

        # print logging information
        logger.write_and_print(
            'The number of the annotated calcifications of this batch = {}'.format(calcification_num_batch_level))
        logger.write_and_print(
            'The number of the recalled calcifications of this batch = {}'.format(recall_num_batch_level))
        logger.write_and_print(
            'The number of the false positive calcifications of this batch = {}'.format(FP_num_batch_level))
        logger.write_and_print('Consuming time: {:.4f}s'.format(time() - start_time_for_batch))
        logger.write_and_print('--------------------------------------------------------------------------------------')

        save_tensor_in_png_and_nii_format(images_tensor, reconstructed_images_tensor, prediction_residues_tensor,
                                          post_process_preds_np, pixel_level_labels_tensor,
                                          pixel_level_labels_dilated_tensor, uncertainty_maps_np, filenames,
                                          result_flag_list, visualization_saving_dir, save_nii=args.save_nii)

        logger.flush()

    logger.write_and_print('The number of the annotated calcifications of this dataset = {}'.format(calcification_num))
    logger.write_and_print('The number of the recalled calcifications of this dataset = {}'.format(recall_num))
    logger.write_and_print('The number of the false positive calcifications of this dataset = {}'.format(FP_num))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationReconstruction(args)
