import argparse
import cleanlab
import cv2
import numpy as np
import os
import sys
import torch
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.getcwd()))

from config.config_confident_learning_pixel_level_classification import cfg
from dataset.dataset_confident_learning_2d import ConfidentLearningDataset2d
from net.vnet2d_v3 import VNet2d
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/noisy-data-alpha-0.5-beta1-7-beta2-12/sub-2/',
                        help='Source data dir.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/noisy-data-alpha-0.5-beta1-7-beta2-12/all/',
                        help='Destination data dir.')
    parser.add_argument('--model_saving_dir',
                        type=str,
                        default='/data1/minqing/models/20200306_JRST_dataset_noisy_alpha-0.5_beta1_7_beta2_12_sub_1_segmentation_clavicle_CE_default/',
                        help='Model saved dir.')
    parser.add_argument('--label_class_name',
                        type=str,
                        default='lung',  # 'clavicle', 'heart', 'lung'
                        help='The label class name.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='training',
                        help='The type of dataset to be evaluated (training, validation).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Batch size for evaluation.')
    parser.add_argument('--epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')

    args = parser.parse_args()

    return args


def TestConfidentMapGeneration(args):
    class_cm_dir = os.path.join(args.dst_data_root_dir, args.dataset_type,
                                '{}-confident-maps'.format(args.label_class_name))
    # create dir when it does not exist
    if not os.path.exists(class_cm_dir):
        os.mkdir(class_cm_dir)

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

    # create dataset
    dataset = ConfidentLearningDataset2d(data_root_dir=args.src_data_root_dir,
                                         mode=args.dataset_type,
                                         enable_random_sampling=False,
                                         class_name=args.label_class_name,
                                         image_channels=cfg.dataset.image_channels,
                                         cropping_size=cfg.dataset.cropping_size,
                                         load_confident_map=False,
                                         enable_data_augmentation=False)

    # create data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=cfg.train.num_threads)

    preds_np_accumulated = 0
    masks_np_accumulated = 0
    filename_list = list()

    num_channels = cfg.net.out_channels
    height = cfg.dataset.cropping_size[0]
    width = cfg.dataset.cropping_size[1]

    for batch_idx, (images_tensor, masks_tensor, _, filenames) in enumerate(data_loader):
        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        preds_tensor = net(images_tensor, use_softmax=True)
        preds_np = preds_tensor.cpu().detach().numpy()
        masks_np = masks_tensor.cpu().numpy()
        filename_list += list(filenames)

        if batch_idx == 0:
            preds_np_accumulated = preds_np
            masks_np_accumulated = masks_np
        else:
            preds_np_accumulated = np.concatenate((preds_np_accumulated, preds_np), axis=0)
            masks_np_accumulated = np.concatenate((masks_np_accumulated, masks_np), axis=0)

        print('Finished evaluating, consuming time = {:.4f}s'.format(time() - start_time_for_batch))
        print('--------------------------------------------------------------------------------------')

    preds_np_accumulated = np.swapaxes(preds_np_accumulated, 1, 2)
    preds_np_accumulated = np.swapaxes(preds_np_accumulated, 2, 3)
    preds_np_accumulated = preds_np_accumulated.reshape(-1, num_channels)
    preds_np_accumulated = np.ascontiguousarray(preds_np_accumulated)

    masks_np_accumulated = masks_np_accumulated.reshape(-1).astype(np.uint8)

    assert preds_np_accumulated.shape[0] == masks_np_accumulated.shape[0]

    noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_np_accumulated, prune_method='both', n_jobs=1)
    confident_maps_np = noise.reshape(-1, height, width).astype(np.uint8) * 255

    for idx in range(len(filename_list)):
        filename = filename_list[idx]
        confident_map_np = confident_maps_np[idx]

        dst_path = os.path.join(class_cm_dir, filename)

        cv2.imwrite(dst_path, confident_map_np)

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestConfidentMapGeneration(args)
