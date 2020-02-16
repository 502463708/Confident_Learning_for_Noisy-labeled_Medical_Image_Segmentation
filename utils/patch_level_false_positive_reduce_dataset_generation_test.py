import argparse
import os
import sys
import torch
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.getcwd()))

from common.utils import get_ckpt_path, generate_radiograph_level_reconstructed_and_residue_result, \
    post_process_residue_radiograph, get_net_list
from config.config_micro_calcification_patch_level_reconstruction import cfg
from dataset.dataset_micro_calcification_radiograph_level import MicroCalcificationRadiographLevelDataset
from logger.logger import Logger
from net.vnet2d_v2 import VNet2d
from torch.utils.data import DataLoader
from time import time
from utils.patch_level_false_positive_reduce_dataset_generation import generate_coordinate_list, merge_coord_list, \
    save_images_labels_uncertainty_maps

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-radiograph-level-roi-extracted-data-split-dataset/',
                        help='The source data dir.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-false-positive-classification-datasets/dataset_debug',
                        help='The destination data dir.')
    parser.add_argument('--reconstruction_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20191108_5764_uCs_patch_level_reconstruction_ttestlossv3_default_dilation_radius_7/',
                        help='The reconstruction model saved dir.')
    parser.add_argument('--reconstruction_epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--uncertainty_model_saving_dir',
                        type=str,
                        default='/data/lars/models/20191108_5764_uCs_patch_level_reconstruction_ttestlossv3_default_dilation_radius_7/',
                        help='The reconstruction model for uncertainty saved dir.')
    parser.add_argument('--mc_epoch_indexes',
                        type=int,
                        default=[410, 420, 430, 440, 450, 460, 470, 480, 490, 500],
                        help='The epoch ckpt indexes for generating uncertainty maps.')
    parser.add_argument('--reconstruction_patch_size',
                        type=tuple,
                        default=(112, 112),
                        help='The patch size for reconstruction.')
    parser.add_argument('--reconstruction_patch_stride',
                        type=int,
                        default=56,
                        help='The patch moving stride from one patch to another.')
    parser.add_argument('--prob_threshold',
                        type=float,
                        default=0.1,
                        help='residue[residue <= prob_threshold] = 0; residue[residue > prob_threshold] = 1')
    parser.add_argument('--area_threshold',
                        type=float,
                        default=3.14 * 7 * 7 * 0.2,
                        help='Connected components whose area < area_threshold will be discarded.')
    parser.add_argument('--patch_size',
                        type=tuple,
                        default=(56, 56),
                        help='The patch size for saving.')
    args = parser.parse_args()

    return args


def TestMicroCalcificationRadiographLevelDetection(args):
    # start time of this dataset
    start_time_for_dataset = time()

    # create clean dir
    if not os.path.exists(args.dst_data_root_dir):
        os.mkdir(args.dst_data_root_dir)
        for patch_type in ['positive_patches', 'negative_patches']:
            os.mkdir(os.path.join(args.dst_data_root_dir, patch_type))
            for dataset_type in ['training', 'validation', 'test']:
                os.mkdir(os.path.join(args.dst_data_root_dir, patch_type, dataset_type))
                for image_type in ['images', 'labels', 'uncertainty-maps']:
                    os.mkdir(os.path.join(args.dst_data_root_dir, patch_type, dataset_type, image_type))

    # initialize logger
    logger = Logger(args.dst_data_root_dir)
    logger.write_and_print('Dataset: {}'.format(args.src_data_root_dir))
    logger.write_and_print('Reconstruction model saving dir: {}'.format(args.reconstruction_model_saving_dir))
    logger.write_and_print('Reconstruction ckpt index: {}'.format(args.reconstruction_epoch_idx))

    # get net list for imitating MC dropout process
    net_for_mc = VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)
    uncertainty_model_ckpt_dir = os.path.join(args.uncertainty_model_saving_dir, 'ckpt')
    net_list = get_net_list(net_for_mc, uncertainty_model_ckpt_dir, args.mc_epoch_indexes, logger)

    # define the reconstruction network
    reconstruction_net = VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)
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

    for dataset_type in ['training', 'validation', 'test']:
        positive_dataset_type_dir = os.path.join(args.dst_data_root_dir, 'positive_patches', dataset_type, 'images')
        negative_dataset_type_dir = os.path.join(args.dst_data_root_dir, 'negative_patches', dataset_type, 'images')

        # create dataset
        dataset = MicroCalcificationRadiographLevelDataset(data_root_dir=args.src_data_root_dir, mode=dataset_type)

        # create data loader
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_threads)

        positive_patch_num_dataset_type_level = 0
        negative_patch_num_dataset_type_level = 0

        logger.write_and_print('  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        logger.write_and_print('  Start evaluating {} set...'.format(dataset_type))

        for radiograph_idx, (images_tensor, pixel_level_labels_tensor, _, filenames) in enumerate(data_loader):
            filename = filenames[0]

            # logging
            logger.write_and_print('    Start evaluating {} set radiograph {} out of {}: {}...'.format(dataset_type,
                                                                                                       radiograph_idx + 1,
                                                                                                       dataset.__len__(),
                                                                                                       filename))

            # start time of this radiograph
            start_time_for_radiograph = time()

            # transfer the tensor into gpu device
            images_tensor = images_tensor.cuda()

            # transfer the tensor into ndarray format
            image_np = images_tensor.cpu().numpy().squeeze()
            pixel_level_label_np = pixel_level_labels_tensor.cpu().numpy().squeeze()

            # generated raw radiograph-level residue
            _, raw_residue_radiograph_np = generate_radiograph_level_reconstructed_and_residue_result(images_tensor,
                                                                                                      reconstruction_net,
                                                                                                      pixel_level_label_np,
                                                                                                      args.reconstruction_patch_size,
                                                                                                      args.reconstruction_patch_stride)

            # post-process the raw radiograph-level residue
            processed_residue_radiograph_np = post_process_residue_radiograph(raw_residue_radiograph_np,
                                                                              pixel_level_label_np,
                                                                              args.prob_threshold,
                                                                              args.area_threshold)

            # generate coordinates list for the post-processed radiograph-level residue
            pred_coord_list = generate_coordinate_list(processed_residue_radiograph_np)

            # generate coordinates list for the mask
            label_coord_list = generate_coordinate_list(pixel_level_label_np, mode='annotated')

            # merge pred_coord_list and label_coord_list
            coord_list = merge_coord_list(pred_coord_list, label_coord_list)

            positive_patch_num_radiograph_level, negative_patch_num_radiograph_level = \
                save_images_labels_uncertainty_maps(coord_list, images_tensor, image_np, pixel_level_label_np, net_list,
                                                    filename, positive_dataset_type_dir, negative_dataset_type_dir,
                                                    args.reconstruction_patch_size, args.patch_size)

            positive_patch_num_dataset_type_level += positive_patch_num_radiograph_level
            negative_patch_num_dataset_type_level += negative_patch_num_radiograph_level

            # logging
            # print logging information of this radiograph
            logger.write_and_print('    Finish evaluating radiograph: {}, consuming time: {:.4f}s'.
                                   format(radiograph_idx + 1, time() - start_time_for_radiograph))
            logger.write_and_print('    This radiograph contains {} positive patches and {} negative patches.'.
                                   format(positive_patch_num_radiograph_level, negative_patch_num_radiograph_level))
            logger.write_and_print('    ------------------------------------------------------------------------------')
            logger.flush()

        # print logging information of this dataset
        logger.write_and_print('  Finished evaluating {} set, consuming time: {:.4f}s'.
                               format(dataset_type, time() - start_time_for_dataset))
        logger.write_and_print('  This {} set contains {} positive patches and {} negative patches.'.
                               format(dataset_type, positive_patch_num_dataset_type_level,
                                      negative_patch_num_dataset_type_level))
        logger.write_and_print('  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestMicroCalcificationRadiographLevelDetection(args)
