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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cudnn.benchmark = True

'''
data_root_dir : alpha=0.3,0.7; beta=clavicle(5,10),heart(14,20),lung(20,30)
model_sub_1_saving_dir
label_class_name: 'clavicle', 'heart', 'lung'
CL_type: 'prune_by_class' or 'prune_by_noise_rate'
'''

def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data1/minqing/data/JRST/noisy-data-alpha-0.7-clavicle-10/',
                        help='Source data dir.')
    parser.add_argument('--model_sub_1_saving_dir',
                        type=str,
                        default='/data1/minqing/models/20200312_JRST_dataset_noisy_alpha-0.7_clavicle_10_sub_1_segmentation_clavicle_CE_default/',
                        help='Model saved dir.')
    parser.add_argument('--label_class_name',
                        type=str,
                        default='clavicle',  # 'clavicle', 'heart', 'lung'
                        help='The label class name.')
    parser.add_argument('--CL_type',
                        type=str,
                        default='prune_by_class',  # 'prune_by_class', 'prune_by_noise_rate', 'both'
                        help='The implement of Confident Learning.')
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


def TestConfidentMapCrossValidation(args, model_idx):
    assert model_idx in [1, 2]

    assert args.CL_type in ['prune_by_class', 'prune_by_noise_rate', 'both']

    src_data_root_dir = os.path.join(args.data_root_dir, 'sub-2')
    model_saving_dir = args.model_sub_1_saving_dir
    if model_idx == 2:
        model_saving_dir = model_saving_dir.replace('sub_1', 'sub_2')
        src_data_root_dir = src_data_root_dir.replace('sub-2', 'sub-1')

    class_cm_dir = os.path.join(args.data_root_dir, 'all', args.dataset_type,
                                '{}-confident-maps'.format(args.label_class_name))
    if args.CL_type != 'both':
        class_cm_dir = class_cm_dir.replace('confident-maps', 'confident-maps' + args.CL_type.replace('_', '-'))

    # create dir when it does not exist
    if not os.path.exists(class_cm_dir):
        os.mkdir(class_cm_dir)

    # define the network
    net = VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)

    # load the specified ckpt
    ckpt_dir = os.path.join(model_saving_dir, 'ckpt')
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
    dataset = ConfidentLearningDataset2d(data_root_dir=src_data_root_dir,
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

    noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_np_accumulated, prune_method='both',
                                               n_jobs=1)
    confident_maps_np = noise.reshape(-1, height, width).astype(np.uint8) * 255

    for idx in range(len(filename_list)):
        filename = filename_list[idx]
        confident_map_np = confident_maps_np[idx]

        dst_path = os.path.join(class_cm_dir, filename)

        cv2.imwrite(dst_path, confident_map_np)

    return


def TestConfidentMapGeneration(args):
    TestConfidentMapCrossValidation(args, 1)
    TestConfidentMapCrossValidation(args, 2)

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestConfidentMapGeneration(args)
