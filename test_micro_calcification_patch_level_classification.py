import argparse
import os
import shutil
import torch
import torch.backends.cudnn as cudnn

from cam.cam import *
from config.config_micro_calcification_patch_level_classification import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from logger.logger import Logger
from metrics.metrics_patch_level_classification import MetricsImageLevelClassification
from net.resnet18 import ResNet18
from time import time
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-dataset-cropped-pathches-connected-component-1/',
                        help='Source data dir.')
    parser.add_argument('--model_saving_dir',
                        type=str,
                        default='/data/lars/models/20190923_uCs_image_level_classification_connected_1_CE_default/',
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
    parser.add_argument('--enable_CAM',
                        type=bool,
                        default=True,
                        help='Whether CAM results can be saved.')

    args = parser.parse_args()

    return args


def TestMicroCalcificationImageLevelClassification(args):
    start_time_for_epoch = time()

    prediction_saving_dir = os.path.join(args.model_saving_dir,
                                         'image_level_classification_results_dataset_{}_epoch_{}'.format(
                                             args.dataset_type, args.epoch_idx))
    visualization_saving_dir = os.path.join(prediction_saving_dir, 'qualitative_results')

    TPs_saving_dir = os.path.join(visualization_saving_dir, 'TPs')
    TNs_saving_dir = os.path.join(visualization_saving_dir, 'TNs')
    FPs_saving_dir = os.path.join(visualization_saving_dir, 'FPs')
    FNs_saving_dir = os.path.join(visualization_saving_dir, 'FNs')

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(prediction_saving_dir):
        shutil.rmtree(prediction_saving_dir)
    os.mkdir(prediction_saving_dir)
    os.mkdir(visualization_saving_dir)
    os.mkdir(TPs_saving_dir)
    os.mkdir(TNs_saving_dir)
    os.mkdir(FPs_saving_dir)
    os.mkdir(FNs_saving_dir)

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

    metrics = MetricsImageLevelClassification(cfg.dataset.cropping_size)

    TPs_epoch_level = 0
    TNs_epoch_level = 0
    FPs_epoch_level = 0
    FNs_epoch_level = 0

    for batch_idx, (
    images_tensor, pixel_level_labels_tensor, _, _, image_level_labels_tensor, _, filenames) in enumerate(
            data_loader):
        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # reshape the label to meet the requirement of CrossEntropy
        image_level_labels_tensor = image_level_labels_tensor.view(-1)  # [B, C] -> [B]

        # network forward
        preds_tensor = net(images_tensor)

        # evaluation
        _, classification_flag_np, TPs_batch_level, TNs_batch_level, FPs_batch_level, FNs_batch_level = \
            metrics.metric_batch_level(preds_tensor, image_level_labels_tensor)

        TPs_epoch_level += TPs_batch_level
        TNs_epoch_level += TNs_batch_level
        FPs_epoch_level += FPs_batch_level
        FNs_epoch_level += FNs_batch_level

        # print logging information
        logger.write_and_print('The number of the TPs of this batch = {}'.format(TPs_batch_level))
        logger.write_and_print('The number of the TNs of this batch = {}'.format(TNs_batch_level))
        logger.write_and_print('The number of the FPs of this batch = {}'.format(FPs_batch_level))
        logger.write_and_print('The number of the FNs of this batch = {}'.format(FNs_batch_level))
        logger.write_and_print('batch: {}, batch_size: {}, consuming time: {:.4f}s'.format(batch_idx, args.batch_size,
                                                                                           time() - start_time_for_batch))
        logger.write_and_print('--------------------------------------------------------------------------------------')

        images_np = images_tensor.cpu().numpy()
        pixel_level_labels_np = pixel_level_labels_tensor.numpy()
        for patch_idx in range(images_tensor.shape[0]):
            image_np = images_np[patch_idx, 0, :, :]
            pixel_level_label_np = pixel_level_labels_np[patch_idx, :, :]
            filename = filenames[patch_idx]
            classification_flag = classification_flag_np[patch_idx]

            assert image_np.shape == pixel_level_label_np.shape
            assert len(image_np.shape) == 2

            image_np *= 255
            image_np = image_np.astype(np.uint8)

            pixel_level_label_np *= 255
            pixel_level_label_np = pixel_level_label_np.astype(np.uint8)

            flag_2_dir_mapping = {0: 'TPs', 1: 'TNs', 2: 'FPs', 3: 'FNs'}
            saving_dir_of_this_patch = os.path.join(visualization_saving_dir, flag_2_dir_mapping[classification_flag])

            cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_image.png')), image_np)
            cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_pixel_level_label.png')),
                        pixel_level_label_np)

            if args.enable_CAM:
                result = generateCAM(net, image_np, "layer3")
                cv2.imwrite(os.path.join(saving_dir_of_this_patch, filename.replace('.png', '_cam.png')),
                            result)

    # print logging information
    logger.write_and_print('##########################################################################################')
    logger.write_and_print('The number of the TPs of this dataset = {}'.format(TPs_epoch_level))
    logger.write_and_print('The number of the TNs of this dataset = {}'.format(TNs_epoch_level))
    logger.write_and_print('The number of the FPs of this dataset = {}'.format(FPs_epoch_level))
    logger.write_and_print('The number of the FNs of this dataset = {}'.format(FNs_epoch_level))
    logger.write_and_print('consuming time: {:.4f}s'.format(time() - start_time_for_epoch))
    logger.write_and_print('##########################################################################################')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationImageLevelClassification(args)
