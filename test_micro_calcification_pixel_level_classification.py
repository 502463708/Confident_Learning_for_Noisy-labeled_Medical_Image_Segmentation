import argparse
import cv2
import numpy as np
import os
import shutil
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn

from common.utils import extract_classification_preds_channel
from config.config_micro_calcification_pixel_level_classification import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from logger.logger import Logger
from metrics.metrics_pixel_level_classification import MetricsPixelLevelClassification
from net.vnet2d_v3 import VNet2d
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-dataset-cropped-pathches/',
                        help='Source data dir.')
    parser.add_argument('--model_saving_dir',
                        type=str,
                        default='/data/lars/models/20190926_uCs_pixel_level_classification_CE_default_dilation_radius_7/',
                        help='Model saved dir.')
    parser.add_argument('--epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
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
                        default=0.5,
                        help='residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1.')
    parser.add_argument('--area_threshold',
                        type=float,
                        default=3.14 * 7 * 7 / 3,
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


def save_tensor_in_png_and_nii_format(images_tensor, predictions_tensor, post_process_preds_np,
                                      pixel_level_labels_tensor, pixel_level_labels_dilated_tensor, filenames,
                                      prediction_saving_dir):
    # convert tensor into numpy and squeeze channel dimension
    images_np = images_tensor.cpu().detach().numpy().squeeze(axis=1)
    predictions_np = predictions_tensor.cpu().detach().numpy().squeeze(axis=1)
    pixel_level_labels_np = pixel_level_labels_tensor.numpy()
    pixel_level_labels_dilated_np = pixel_level_labels_dilated_tensor.numpy()

    # iterating each image of this batch
    for idx in range(images_np.shape[0]):
        image_np = images_np[idx, :, :]
        prediction_np = predictions_np[idx, :, :]
        post_process_pred_np = post_process_preds_np[idx, :, :]
        pixel_level_label_np = pixel_level_labels_np[idx, :, :]
        pixel_level_label_dilated_np = pixel_level_labels_dilated_np[idx, :, :]
        filename = filenames[idx]

        stacked_np = np.concatenate((np.expand_dims(image_np, axis=0),
                                     np.expand_dims(prediction_np, axis=0),
                                     np.expand_dims(post_process_pred_np, axis=0),
                                     np.expand_dims(pixel_level_label_np, axis=0),
                                     np.expand_dims(pixel_level_label_dilated_np, axis=0)), axis=0)

        stacked_image = sitk.GetImageFromArray(stacked_np)
        sitk.WriteImage(stacked_image, os.path.join(prediction_saving_dir, filename.replace('png', 'nii')))

        image_np *= 255
        prediction_np *= 255
        post_process_pred_np *= 255
        pixel_level_label_np *= 255
        pixel_level_label_dilated_np *= 255

        image_np = image_np.astype(np.uint8)
        prediction_np = prediction_np.astype(np.uint8)
        post_process_pred_np = post_process_pred_np.astype(np.uint8)
        pixel_level_label_np = pixel_level_label_np.astype(np.uint8)
        pixel_level_label_dilated_np = pixel_level_label_dilated_np.astype(np.uint8)

        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_image.png')),
                    image_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_pred.png')),
                    prediction_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_post_processed_pred.png')),
                    post_process_pred_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_pixel_level_label.png')),
                    pixel_level_label_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_pixel_level_dilated_label.png')),
                    pixel_level_label_dilated_np)

    return


def TestMicroCalcificationReconstruction(args):
    prediction_saving_dir = os.path.join(args.model_saving_dir,
                                         'pixel_level_classification_results_dataset_{}_epoch_{}'.format(
                                             args.dataset_type, args.epoch_idx))
    visualization_saving_dir = os.path.join(prediction_saving_dir, 'qualitative_results')

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(prediction_saving_dir):
        shutil.rmtree(prediction_saving_dir)
    os.mkdir(prediction_saving_dir)
    os.mkdir(visualization_saving_dir)

    # initialize logger
    logger = Logger(prediction_saving_dir, 'quantitative_results.txt')
    logger.write_and_print('Dataset: {}'.format(args.data_root_dir))
    logger.write_and_print('Dataset type: {}'.format(args.dataset_type))

    # define the network
    net = VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)

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
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(ckpt_path))
    net = net.eval()

    logger.write_and_print('Load ckpt: {0}...'.format(ckpt_path))

    # create dataset and data loader
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
    #
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=cfg.train.num_threads)

    metrics = MetricsPixelLevelClassification(args.prob_threshold, args.area_threshold, args.distance_threshold)

    calcification_num = 0
    recall_num = 0
    FP_num = 0

    for batch_idx, (images_tensor, pixel_level_labels_tensor, pixel_level_labels_dilated_tensor, _,
                    image_level_labels_tensor, _, filenames) in enumerate(data_loader):
        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # network forward
        predictions_tensor = net(images_tensor)

        # extract the 1-st channel from classification results
        predictions_tensor = extract_classification_preds_channel(predictions_tensor, 1)

        # evaluation
        post_process_preds_np, calcification_num_batch_level, recall_num_batch_level, FP_num_batch_level = \
            metrics.metric_batch_level(predictions_tensor, pixel_level_labels_tensor)

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

        save_tensor_in_png_and_nii_format(images_tensor, predictions_tensor, post_process_preds_np,
                                          pixel_level_labels_tensor, pixel_level_labels_dilated_tensor, filenames,
                                          visualization_saving_dir)

        logger.flush()

    logger.write_and_print('The number of the annotated calcifications of this dataset = {}'.format(calcification_num))
    logger.write_and_print('The number of the recalled calcifications of this dataset = {}'.format(recall_num))
    logger.write_and_print('The number of the false positive calcifications of this dataset = {}'.format(FP_num))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationReconstruction(args)
