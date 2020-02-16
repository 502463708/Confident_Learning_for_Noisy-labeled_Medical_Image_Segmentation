import argparse
import cv2
import numpy as np
import os
import shutil
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn

from common.utils import get_ckpt_path
from config.config_micro_calcification_patch_level_classification import cfg as c_cfg
from config.config_micro_calcification_patch_level_reconstruction import cfg as r_cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from logger.logger import Logger
from metrics.metrics_patch_level_reconstruction import MetricsReconstruction
from net.resnet18 import ResNet18
from net.vnet2d_v2 import VNet2d
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-dataset-cropped-pathches/',
                        help='The source data dir.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='test',
                        help='The type of dataset to be evaluated (training, validation, test).')
    parser.add_argument('--prediction_saving_dir',
                        type=str,
                        default='/data/lars/results/Micro_calcification_detection/',
                        help='The predicted results saving dir.')
    parser.add_argument('--reconstruction_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20190925_uCs_reconstruction_ttestlossv3_default_dilation_radius_7',
                        help='The reconstruction model saved dir.')
    parser.add_argument('--reconstruction_epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--classification_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20190926_uCs_image_level_classification_CE_default/',
                        help='The classification model saved dir.')
    parser.add_argument('--classification_epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--dilation_radius',
                        type=int,
                        default=7,
                        help='The specified dilation_radius when training.')
    parser.add_argument('--prob_threshold',
                        type=float,
                        default=0.2,
                        help='residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1')
    parser.add_argument('--area_threshold',
                        type=float,
                        default=3.14 * 12 * 12 / 3,
                        help='Connected components whose area < area_threshold will be discarded.')
    parser.add_argument('--distance_threshold',
                        type=int,
                        default=14,
                        help='Candidates whose distance between calcification < distance_threshold is a recalled one.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=48,
                        help='Batch size for evaluation.')

    args = parser.parse_args()

    return args


def micro_calcification_detection_batch_level(residues_tensor, classification_preds_tensor, detection_threshold=-1.0):
    assert len(residues_tensor.shape) == 4
    assert len(classification_preds_tensor.shape) == 2
    assert residues_tensor.shape[0] == classification_preds_tensor.shape[0]

    batch_size = residues_tensor.shape[0]

    # transfer the tensor into cpu device
    if residues_tensor.device.type != 'cpu':
        residues_tensor = residues_tensor.cpu().detach()
    # transform the tensor into ndarray format
    residues_np = residues_tensor.numpy()

    # transfer the tensor into cpu device
    if classification_preds_tensor.device.type != 'cpu':
        classification_preds_tensor = classification_preds_tensor.cpu().detach()
    # transform the tensor into ndarray format
    classification_preds_np = classification_preds_tensor.numpy()

    detection_results_np_list = list()

    for patch_idx in range(batch_size):
        residue_np = residues_np[patch_idx, :, :, :]
        classification_pred_np = classification_preds_np[patch_idx, :]
        if classification_pred_np.max() == classification_pred_np[0]:
            detection_result_np = np.zeros_like(residue_np)
        else:
            detection_result_np = residue_np
        detection_results_np_list.append(detection_result_np)

    detection_results_np = np.array(detection_results_np_list)

    assert residues_np.shape == detection_results_np.shape

    return detection_results_np


def save_tensor_in_png_and_nii_format(images_tensor, reconstructed_images_tensor, prediction_residues_tensor,
                                      post_process_preds_np, pixel_level_labels_tensor,
                                      pixel_level_labels_dilated_tensor, filenames, prediction_saving_dir):
    # convert tensor into numpy and squeeze channel dimension
    images_np = images_tensor.cpu().detach().numpy().squeeze(axis=1)
    reconstructed_images_np = reconstructed_images_tensor.cpu().detach().numpy().squeeze(axis=1)
    prediction_residues_np = prediction_residues_tensor.cpu().detach().numpy().squeeze(axis=1)
    pixel_level_labels_np = pixel_level_labels_tensor.numpy()
    pixel_level_labels_dilated_np = pixel_level_labels_dilated_tensor.numpy()

    # iterating each image of this batch
    for idx in range(images_np.shape[0]):
        image_np = images_np[idx, :, :]
        reconstructed_image_np = reconstructed_images_np[idx, :, :]
        prediction_residue_np = prediction_residues_np[idx, :, :]
        post_process_pred_np = post_process_preds_np[idx, :, :]
        pixel_level_label_np = pixel_level_labels_np[idx, :, :]
        pixel_level_label_dilated_np = pixel_level_labels_dilated_np[idx, :, :]
        filename = filenames[idx]

        stacked_np = np.concatenate((np.expand_dims(image_np, axis=0),
                                     np.expand_dims(reconstructed_image_np, axis=0),
                                     np.expand_dims(prediction_residue_np, axis=0),
                                     np.expand_dims(post_process_pred_np, axis=0),
                                     np.expand_dims(pixel_level_label_np, axis=0),
                                     np.expand_dims(pixel_level_label_dilated_np, axis=0)), axis=0)

        stacked_image = sitk.GetImageFromArray(stacked_np)
        sitk.WriteImage(stacked_image, os.path.join(prediction_saving_dir, filename.replace('png', 'nii')))

        image_np *= 255
        reconstructed_image_np *= 255
        prediction_residue_np *= 255
        post_process_pred_np *= 255
        pixel_level_label_np *= 255
        pixel_level_label_dilated_np *= 255

        image_np = image_np.astype(np.uint8)
        reconstructed_image_np = reconstructed_image_np.astype(np.uint8)
        prediction_residue_np = prediction_residue_np.astype(np.uint8)
        post_process_pred_np = post_process_pred_np.astype(np.uint8)
        pixel_level_label_np = pixel_level_label_np.astype(np.uint8)
        pixel_level_label_dilated_np = pixel_level_label_dilated_np.astype(np.uint8)

        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_image.png')),
                    image_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_reconstructed.png')),
                    reconstructed_image_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_residue.png')),
                    prediction_residue_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_post_processed_residue.png')),
                    post_process_pred_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_pixel_level_label.png')),
                    pixel_level_label_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_pixel_level_dilated_label.png')),
                    pixel_level_label_dilated_np)

    return


def TestMicroCalcificationDetectionPatchLevel(args):
    visualization_saving_dir = os.path.join(args.prediction_saving_dir, 'qualitative_results')

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(args.prediction_saving_dir):
        shutil.rmtree(args.prediction_saving_dir)
    os.mkdir(args.prediction_saving_dir)
    os.mkdir(visualization_saving_dir)

    # initialize logger
    logger = Logger(args.prediction_saving_dir, 'quantitative_results.txt')
    logger.write_and_print('Dataset: {}'.format(args.data_root_dir))
    logger.write_and_print('Dataset type: {}'.format(args.dataset_type))
    logger.write_and_print('Reconstruction model saving dir: {}'.format(args.reconstruction_model_saving_dir))
    logger.write_and_print('Reconstruction ckpt index: {}'.format(args.reconstruction_epoch_idx))
    logger.write_and_print('Classification model saving dir: {}'.format(args.classification_model_saving_dir))
    logger.write_and_print('Classification ckpt index: {}'.format(args.classification_epoch_idx))

    # define the reconstruction network
    reconstruction_net = VNet2d(num_in_channels=r_cfg.net.in_channels, num_out_channels=r_cfg.net.out_channels)
    #
    # get the reconstruction absolute ckpt path
    reconstruction_ckpt_path = get_ckpt_path(args.reconstruction_model_saving_dir, args.reconstruction_epoch_idx)
    #
    # load ckpt and transfer net into gpu devices
    reconstruction_net = torch.nn.DataParallel(reconstruction_net).cuda()
    reconstruction_net.load_state_dict(torch.load(reconstruction_ckpt_path))
    reconstruction_net = reconstruction_net.eval()
    #
    logger.write_and_print('Load ckpt: {0}...'.format(reconstruction_ckpt_path))

    # define the classification network
    classification_net = ResNet18(in_channels=c_cfg.net.in_channels, num_classes=c_cfg.net.num_classes)
    #
    # get the classification absolute ckpt path
    classification_ckpt_path = get_ckpt_path(args.classification_model_saving_dir, args.classification_epoch_idx)
    #
    # load ckpt and transfer net into gpu devices
    classification_net = torch.nn.DataParallel(classification_net).cuda()
    classification_net.load_state_dict(torch.load(classification_ckpt_path))
    classification_net = classification_net.eval()
    #
    logger.write_and_print('Load ckpt: {0}...'.format(classification_ckpt_path))

    # create dataset and data loader
    dataset = MicroCalcificationDataset(data_root_dir=args.data_root_dir,
                                        mode=args.dataset_type,
                                        enable_random_sampling=False,
                                        pos_to_neg_ratio=r_cfg.dataset.pos_to_neg_ratio,
                                        image_channels=r_cfg.dataset.image_channels,
                                        cropping_size=r_cfg.dataset.cropping_size,
                                        dilation_radius=args.dilation_radius,
                                        load_uncertainty_map=False,
                                        calculate_micro_calcification_number=r_cfg.dataset.calculate_micro_calcification_number,
                                        enable_data_augmentation=False)
    #
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=r_cfg.train.num_threads)

    metrics = MetricsReconstruction(args.prob_threshold, args.area_threshold, args.distance_threshold)

    calcification_num = 0
    recall_num = 0
    FP_num = 0

    for batch_idx, (images_tensor, pixel_level_labels_tensor, pixel_level_labels_dilated_tensor, _,
                    image_level_labels_tensor, _, filenames) in enumerate(data_loader):
        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # reconstruction network forward
        reconstructed_images_tensor, prediction_residues_tensor = reconstruction_net(images_tensor)

        # classification network forward
        classification_preds_tensor = classification_net(images_tensor)

        # merge the reconstruction and the classification results
        detection_results_np = micro_calcification_detection_batch_level(prediction_residues_tensor,
                                                                         classification_preds_tensor)

        # evaluation
        post_process_preds_np, calcification_num_batch_level, recall_num_batch_level, FP_num_batch_level = \
            metrics.metric_batch_level(detection_results_np, pixel_level_labels_tensor)

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
        logger.write_and_print('batch: {}, consuming time: {:.4f}s'.format(batch_idx, time() - start_time_for_batch))
        logger.write_and_print('--------------------------------------------------------------------------------------')

        save_tensor_in_png_and_nii_format(images_tensor, reconstructed_images_tensor, prediction_residues_tensor,
                                          post_process_preds_np, pixel_level_labels_tensor,
                                          pixel_level_labels_dilated_tensor, filenames, visualization_saving_dir)

        logger.flush()

    logger.write_and_print('The number of the annotated calcifications of this dataset = {}'.format(calcification_num))
    logger.write_and_print('The number of the recalled calcifications of this dataset = {}'.format(recall_num))
    logger.write_and_print('The number of the false positive calcifications of this dataset = {}'.format(FP_num))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationDetectionPatchLevel(args)
