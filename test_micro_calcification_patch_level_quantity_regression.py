import argparse
import cv2
import numpy as np
import os
import shutil
import torch
import torch.backends.cudnn as cudnn

from config.config_micro_calcification_patch_level_quantity_regression import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from logger.logger import Logger
from metrics.metrics_patch_level_quantity_regression import MetricsImageLEvelQuantityRegression
from net.resnet18 import ResNet18
from time import time
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/home/groupprofzli/data1/dwz/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/',
                        help='Source data dir.')
    parser.add_argument('--model_saving_dir',
                        type=str,
                        default='/home/groupprofzli/data1/dwz/data/models/20191027_uCs_quantity_regression_L1/',
                        help='Model saved dir.')
    parser.add_argument('--epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='test',
                        help='The type of dataset to be evaluated (training, validation, test).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=48,
                        help='Batch size for evaluation.')

    args = parser.parse_args()

    return args


def TestMicroCalcificationPatchLevelQuantityRegression(args):
    start_time_for_epoch = time()

    prediction_saving_dir = os.path.join(args.model_saving_dir,
                                         'patch_level_quantity_regression_results_dataset_{}_epoch_{}'.format(
                                             args.dataset_type, args.epoch_idx))
    visualization_saving_dir = os.path.join(prediction_saving_dir, 'qualitative_results')

    over_preds_dir = os.path.join(visualization_saving_dir, 'over_preds')
    correct_preds_dir = os.path.join(visualization_saving_dir, 'correct_preds')
    under_preds_dir = os.path.join(visualization_saving_dir, 'under_preds')

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(prediction_saving_dir):
        shutil.rmtree(prediction_saving_dir)
    os.mkdir(prediction_saving_dir)
    os.mkdir(visualization_saving_dir)
    os.mkdir(over_preds_dir)
    os.mkdir(correct_preds_dir)
    os.mkdir(under_preds_dir)

    # initialize logger
    logger = Logger(prediction_saving_dir, 'quantitative_results.txt')
    logger.write_and_print('Dataset: {}'.format(args.data_root_dir))
    logger.write_and_print('Dataset type: {}'.format(args.dataset_type))

    # define the network
    net = ResNet18(in_channels=cfg.net.in_channels, num_classes=cfg.net.num_classes)

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
                                        dilation_radius=cfg.dataset.dilation_radius,
                                        load_uncertainty_map=False,
                                        calculate_micro_calcification_number=cfg.dataset.calculate_micro_calcification_number,
                                        enable_data_augmentation=False)

    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=cfg.train.num_threads)

    metrics = MetricsImageLEvelQuantityRegression(cfg.dataset.cropping_size)

    pred_num_epoch_level = 0
    distance_epoch_level = 0
    over_pred_epoch_level = 0
    correct_pred_epoch_level = 0
    under_pred_epoch_level = 0

    for batch_idx, (
            images_tensor, pixel_level_labels_tensor, _, _, image_level_labels_tensor,
            micro_calcification_number_label_tensor,
            filenames) in enumerate(
        data_loader):
        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()
        pixel_level_labels_tensor = pixel_level_labels_tensor.cuda()
        micro_calcification_number_label_tensor = micro_calcification_number_label_tensor.type(torch.FloatTensor)
        micro_calcification_number_label_tensor = micro_calcification_number_label_tensor.cuda()

        # network forward
        preds_tensor = net(images_tensor)  # the shape of preds_tensor: [B*1]

        # metrics
        classification_flag_np, visual_preds_np, visual_labels_np, distance_batch_level, over_preds_batch_level, \
        correct_preds_batch_level, under_preds_batch_level = \
            metrics.metric_batch_level(preds_tensor, micro_calcification_number_label_tensor)
        pred_num_epoch_level += preds_tensor.shape[0]
        distance_epoch_level += distance_batch_level
        over_pred_epoch_level += over_preds_batch_level
        correct_pred_epoch_level += correct_preds_batch_level
        under_pred_epoch_level += under_preds_batch_level

        # print logging information
        logger.write_and_print(
            'The number of the over predicted patches of this batch = {}'.format(over_preds_batch_level))
        logger.write_and_print(
            'The number of the correct predicted patches of this batch = {}'.format(correct_preds_batch_level))
        logger.write_and_print(
            'The number of the under predicted patches of this batch = {}'.format(under_preds_batch_level))
        logger.write_and_print('The value of the MSE of this batch = {}'.format(distance_batch_level))
        logger.write_and_print('batch: {}, batch_size: {}, consuming time: {:.4f}s'.format(batch_idx, args.batch_size,
                                                                                           time() - start_time_for_batch))
        logger.write_and_print('--------------------------------------------------------------------------------------')

        images_np = images_tensor.cpu().numpy()
        pixel_level_labels_np = pixel_level_labels_tensor.cpu().numpy()
        for patch_idx in range(images_tensor.shape[0]):
            image_np = images_np[patch_idx, 0, :, :]
            visual_pred_np = visual_preds_np[patch_idx, :, :]
            visual_label_np = visual_labels_np[patch_idx, :, :]
            pixel_level_label_np = pixel_level_labels_np[patch_idx, :, :]
            filename = filenames[patch_idx]
            classification_flag = classification_flag_np[patch_idx]

            assert image_np.shape == pixel_level_label_np.shape
            assert len(image_np.shape) == 2

            image_np *= 255
            image_np = image_np.astype(np.uint8)

            pixel_level_label_np *= 255
            pixel_level_label_np = pixel_level_label_np.astype(np.uint8)

            flag_2_dir_mapping = {0: 'over_preds', 1: 'correct_preds', 2: 'under_preds'}
            saving_dir_of_this_patch = os.path.join(visualization_saving_dir, flag_2_dir_mapping[classification_flag])
            cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_image.png')), image_np)
            cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_pixel_level_label.png')),
                        pixel_level_label_np)
            cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_mask_num.png')),
                        visual_label_np)
            cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_pred_num.png')),
                        visual_pred_np)

    # print logging information
    logger.write_and_print('##########################################################################################')
    logger.write_and_print('The number of the patches of this dataset = {}'.format(pred_num_epoch_level))
    logger.write_and_print(
        'The number of the over predicted patches of this dataset = {}'.format(over_pred_epoch_level))
    logger.write_and_print(
        'The number of the correct predicted patches of this dataset = {}'.format(correct_pred_epoch_level))
    logger.write_and_print(
        'The number of the under predicted patches of this dataset = {}'.format(under_pred_epoch_level))
    logger.write_and_print(
        'The value of the MSE of this dataset = {}'.format(distance_epoch_level / pred_num_epoch_level))
    logger.write_and_print('consuming time: {:.4f}s'.format(time() - start_time_for_epoch))
    logger.write_and_print('##########################################################################################')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationPatchLevelQuantityRegression(args)
