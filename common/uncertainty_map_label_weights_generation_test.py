import argparse
import os
import sys
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.getcwd()))

from common.uncertainty_map_label_weights_generation import save_uncertainty_maps
from common.utils import get_net_list, generate_uncertainty_maps
from config.config_micro_calcification_patch_level_reconstruction import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from logger.logger import Logger
from net.vnet2d_v2 import VNet2d
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-sub-datasets/sub-dataset-1/',
                        help='Source data dir.')
    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/',
                        help='Destination data dir.')
    parser.add_argument('--model_saving_dir',
                        type=str,
                        default='/data/lars/models/20191108_5764_uCs_patch_level_reconstruction_ttestlossv3_default_dilation_radius_7/',
                        help='Model saved dir.')
    parser.add_argument('--mc_epoch_indexes',
                        type=int,
                        default=[410, 420, 430, 440, 450, 460, 470, 480, 490, 500],
                        help='The epoch ckpt indexes for generating uncertainty maps set null list [] to switch off.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='training',
                        help='The type of dataset to be evaluated (training, validation, test).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=48,
                        help='Batch size for evaluation.')

    args = parser.parse_args()

    return args





def TestUncertaintyMapLabelWeightsGeneration(args):
    positive_patch_results_saving_dir = os.path.join(args.dst_data_root_dir, 'positive_patches', args.dataset_type,
                                                     'uncertainty-maps')
    negative_patch_results_saving_dir = os.path.join(args.dst_data_root_dir, 'negative_patches', args.dataset_type,
                                                     'uncertainty-maps')

    # create dir when it does not exist
    if not os.path.exists(positive_patch_results_saving_dir):
        os.mkdir(positive_patch_results_saving_dir)
    if not os.path.exists(negative_patch_results_saving_dir):
        os.mkdir(negative_patch_results_saving_dir)

    # initialize logger
    logger = Logger(args.src_data_root_dir, 'uncertainty.txt')
    logger.write_and_print('Dataset: {}'.format(args.src_data_root_dir))
    logger.write_and_print('Dataset type: {}'.format(args.dataset_type))

    # define the network
    network = VNet2d(num_in_channels=cfg.net.in_channels, num_out_channels=cfg.net.out_channels)

    ckpt_dir = os.path.join(args.model_saving_dir, 'ckpt')

    # get net list for imitating MC dropout process
    net_list = get_net_list(network, ckpt_dir, args.mc_epoch_indexes, logger)

    # create dataset
    dataset = MicroCalcificationDataset(data_root_dir=args.src_data_root_dir,
                                        mode=args.dataset_type,
                                        enable_random_sampling=False,
                                        pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                        image_channels=cfg.dataset.image_channels,
                                        cropping_size=cfg.dataset.cropping_size,
                                        dilation_radius=0,
                                        load_uncertainty_map=False,
                                        calculate_micro_calcification_number=False,
                                        enable_data_augmentation=False)

    # create data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=cfg.train.num_threads)

    for batch_idx, (images_tensor, _, _, _, _, _, filenames) in enumerate(data_loader):
        logger.write_and_print('Evaluating batch: {}'.format(batch_idx))

        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # imitating MC dropout
        uncertainty_maps_np = generate_uncertainty_maps(net_list, images_tensor)
        save_uncertainty_maps(uncertainty_maps_np, filenames, positive_patch_results_saving_dir,
                              negative_patch_results_saving_dir, logger)

        logger.write_and_print('Finished evaluating, consuming time = {:.4f}s'.format(time() - start_time_for_batch))
        logger.write_and_print('--------------------------------------------------------------------------------------')

        logger.flush()

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestUncertaintyMapLabelWeightsGeneration(args)
