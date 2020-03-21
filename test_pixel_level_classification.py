import argparse
import cv2
import numpy as np
import os
import shutil
import torch
import torch.backends.cudnn as cudnn

from config.config_confident_learning_pixel_level_classification import cfg
from dataset.dataset_confident_learning_2d import ConfidentLearningDataset2d
from net.pick_and_learn import PLNet2d
from net.vnet2d_v3 import VNet2d
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/clean-data',
                        help='Source data dir.')
    parser.add_argument('--model_saving_dir',
                        type=str,
                        default='/data1/minqing/models/20200317_JRST_dataset_noisy_alpha-0.3_lung_30_all_segmentation_lung_pick_and_learn/',
                        help='Model saved dir.')
    parser.add_argument('--network_filename',
                        type=str,
                        default='pick_and_learn',
                        help='The name of the file containing the implemented network.')
    parser.add_argument('--epoch_idx',
                        type=int,
                        default=200,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--label_class_name',
                        type=str,
                        default='lung',  # 'clavicle', 'heart', 'lung'
                        help='The label class name.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='validation',
                        help='The type of dataset to be evaluated (training, validation).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Batch size for evaluation.')

    args = parser.parse_args()

    return args


def save_tensor_in_png(images_tensor, predictions_tensor, pixel_level_labels_tensor, filenames, prediction_saving_dir):
    # convert tensor into numpy and squeeze channel dimension
    images_np = images_tensor.cpu().detach().numpy().squeeze(axis=1)
    predictions_np = predictions_tensor.cpu().detach().numpy()
    pixel_level_labels_np = pixel_level_labels_tensor.numpy()

    # iterating each image of this batch
    for idx in range(images_np.shape[0]):
        image_np = images_np[idx, :, :]
        prediction_np = predictions_np[idx, :, :]
        pixel_level_label_np = pixel_level_labels_np[idx, :, :]
        filename = filenames[idx]

        image_np *= 255
        prediction_np *= 255
        pixel_level_label_np *= 255

        image_np = image_np.astype(np.uint8)
        prediction_np = prediction_np.astype(np.uint8)
        pixel_level_label_np = pixel_level_label_np.astype(np.uint8)

        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_image.png')),
                    image_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_pred.png')),
                    prediction_np)
        cv2.imwrite(os.path.join(prediction_saving_dir, filename.replace('.png', '_mask.png')),
                    pixel_level_label_np)

    return


def TestPixelLevelClassification(args):
    visualization_saving_dir = os.path.join(args.model_saving_dir, 'results-{}-epoch-{}'.format(
        args.model_saving_dir.split('/')[-2].replace('_', '-'), args.epoch_idx))

    # remove existing dir which has the same name and create clean dir
    if os.path.exists(visualization_saving_dir):
        shutil.rmtree(visualization_saving_dir)
    os.mkdir(visualization_saving_dir)

    # define the network
    assert args.network_filename in ['vnet3d_v3', 'pick_and_learn']
    if args.network_filename in ['vnet3d_v3']:
        net = VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)
    elif args.network_filename in ['pick_and_learn']:
        net = PLNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)

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

    # create dataset and data loader
    # data_root_dir, mode, class_name, enable_random_sampling, image_channels, cropping_size,
    #                  load_confident_map, enable_data_augmentation
    dataset = ConfidentLearningDataset2d(data_root_dir=args.data_root_dir,
                                         mode=args.dataset_type,
                                         class_name=args.label_class_name,
                                         enable_random_sampling=False,
                                         image_channels=cfg.dataset.image_channels,
                                         cropping_size=cfg.dataset.cropping_size,
                                         load_confident_map=False,
                                         enable_data_augmentation=False)
    #
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=cfg.train.num_threads)

    for batch_idx, (images_tensor, pixel_level_labels_tensor, _, filenames) in enumerate(data_loader):
        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # network forward
        if net.module.get_name() == 'VNet2d':
            predictions_tensor = net(images_tensor)
        elif net.module.get_name() == 'PLNet2d':
            predictions_tensor, _ = net(images_tensor, pixel_level_labels_tensor)

        _, post_process_preds = torch.max(predictions_tensor, dim=1)

        print('batch: {}, consuming time: {:.4f}s'.format(batch_idx, time() - start_time_for_batch))
        print('-------------------------------------------------------------------------------------------------------')

        save_tensor_in_png(images_tensor, post_process_preds, pixel_level_labels_tensor, filenames,
                           visualization_saving_dir)

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestPixelLevelClassification(args)
