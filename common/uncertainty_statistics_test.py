import argparse
import copy
import numpy as np
import os
import shutil
import sys
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.dirname(os.getcwd()))

from common.utils import get_net_list, generate_uncertainty_maps
from config.config_micro_calcification_patch_level_reconstruction import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from logger.logger import Logger
from metrics.metrics_patch_level_reconstruction import MetricsReconstruction
from net.vnet2d_v2 import VNet2d
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-microcalcification-datasets-5764-uCs-20191107/Inbreast-patch-level-split-pos2neg-ratio-1-dataset/',
                        help='Source data dir.')
    parser.add_argument('--sta_save_dir',
                        type=str,
                        default='/data/lars/results/uncertainty_sta',
                        help='statistics save dir')
    parser.add_argument('--bins',
                        type=int,
                        default=500,
                        help='the number of intervals to divide uncertainty value')
    parser.add_argument('--model_saving_dir',
                        type=str,
                        default='/data/lars/models/models_copied_from_10.26.2.22/20191129_5764_uCs_patch_level_reconstruction_ttestlossv3_default_dilation_radius_7/',
                        help='Model saved dir.')
    parser.add_argument('--epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
    parser.add_argument('--mc_epoch_indexes',
                        type=int,
                        default=[580, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680],
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
                        default=30,
                        help='Batch size for evaluation.')
    parser.add_argument('--bin_range',
                        type=int,
                        default=0.15,
                        help='Batch size for evaluation.')
    parser.add_argument('--save_nii',
                        type=bool,
                        default=False,
                        help='A bool variable indicating whether nii format data is gonna be saved.')

    args = parser.parse_args()

    return args


def pltsave(hist, dir, name):
    objects = np.linspace(0, 1, len(hist))
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, hist)
    plt.xticks(np.linspace(0, args.bins, num=6), np.linspace(0, args.bin_range, num=6))
    plt.ylabel('uncertainty value')
    plt.title("{} distribution".format(name))
    plt.savefig(os.path.join(dir, '{}.png'.format(name)))
    plt.close()


def find_mean_and_var(seq, start, end, bins, bin_range):
    mean = 0
    mean_pvalue = 0
    var = 0
    var_weight = 0
    for idx in range(int(start), int(seq.shape[0])):
        if idx > end:
            continue
        if seq[idx] > mean_pvalue:
            mean = idx
            mean_pvalue = seq[idx]
    for idx in range(mean, int(seq.shape[0])):
        var += ((idx - mean) / bins * bin_range) ** 2 * seq[idx]
        var_weight += seq[idx]
    var = var / var_weight

    return mean / bins * bin_range, var ** 0.5


def UncertaintySTA(args):
    prediction_saving_dir = os.path.join(args.model_saving_dir,
                                         'reconstruction_results_dataset_{}_epoch_{}'.format(args.dataset_type,
                                                                                             args.epoch_idx))
    # initialize logger
    if os.path.exists(args.sta_save_dir):
        shutil.rmtree(args.sta_save_dir)
    os.mkdir(args.sta_save_dir)
    logger = Logger(args.sta_save_dir, 'uncertainty_distribution_sta.txt')
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

    all_positive_uncertainty_in_dataset = np.zeros(args.bins)
    tp_uncertainty_in_dataset = np.zeros(args.bins)
    fn_uncertainty_in_dataset = np.zeros(args.bins)
    fp_uncertainty_in_dataset = np.zeros(args.bins)
    uncertainty_max=0

    for batch_idx, (images_tensor, pixel_level_labels_tensor, pixel_level_labels_dilated_tensor, _,
                    image_level_labels_tensor, _, filenames) in enumerate(data_loader):
        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # network forward
        reconstructed_images_tensor, prediction_residues_tensor = net(images_tensor)

        # MC dropout
        uncertainty_maps_np = generate_uncertainty_maps(net_list, images_tensor) if calculate_uncertainty else None

        # in tp, fn label area  uncertainty value distribution
        post_process_preds_np, calcification_num_batch_level, recall_num_batch_level, FP_num_batch_level, \
        result_flag_list = metrics.metric_batch_level(prediction_residues_tensor, pixel_level_labels_tensor)

        pixel_level_labels_dilated = pixel_level_labels_dilated_tensor.view(-1).numpy()
        preds_positive = post_process_preds_np.reshape(-1)
        uncertainty_maps = uncertainty_maps_np.reshape(-1)
        if uncertainty_max< np.amax(uncertainty_maps):
            uncertainty_max = np.amax(uncertainty_maps)

        all_positive_uncertainty_batch = uncertainty_maps[pixel_level_labels_dilated == 1]
        all_positive_uncertainty_distr_batch, _ = np.histogram(all_positive_uncertainty_batch,
                                                               bins=args.bins, range=(0, args.bin_range))
        all_positive_uncertainty_in_dataset += all_positive_uncertainty_distr_batch

        pixel_level_unlabels_dilated = np.subtract(np.ones_like(pixel_level_labels_dilated), pixel_level_labels_dilated)
        fp_location = np.multiply(pixel_level_unlabels_dilated, preds_positive)
        tp_location = np.multiply(pixel_level_labels_dilated, preds_positive)
        fn_location = np.zeros_like(preds_positive)
        fn_location[pixel_level_labels_dilated == 1] = 1
        fn_location[preds_positive == 1] = 0

        tp_uncertainty_batch = uncertainty_maps[tp_location == 1]
        tp_uncertainty_distr_batch, _ = np.histogram(tp_uncertainty_batch, bins=args.bins, range=(0, args.bin_range))
        tp_uncertainty_in_dataset += tp_uncertainty_distr_batch

        fn_uncertainty_batch = uncertainty_maps[fn_location == 1]
        fn_uncertainty_distr_batch, _ = np.histogram(fn_uncertainty_batch, bins=args.bins, range=(0, args.bin_range))
        fn_uncertainty_in_dataset += fn_uncertainty_distr_batch

        fp_uncertainty_batch = uncertainty_maps[fp_location == 1]
        fp_uncertainty_distr_batch, _ = np.histogram(fp_uncertainty_batch, bins=args.bins, range=(0, args.bin_range))
        fp_uncertainty_in_dataset += fp_uncertainty_distr_batch

    # debug only
    # print(all_positive_uncertainty_in_dataset[0:5])
    # print(tp_uncertainty_in_dataset[0:5])
    # print(fn_uncertainty_in_dataset[0:5])

    all_positive_uncertainty_in_dataset[all_positive_uncertainty_in_dataset > 1000] = 1000
    tp_uncertainty_in_dataset[tp_uncertainty_in_dataset > 1000] = 1000
    fn_uncertainty_in_dataset[fn_uncertainty_in_dataset > 1000] = 1000
    fp_uncertainty_in_dataset[fp_uncertainty_in_dataset > 1000] = 1000

    pltsave(all_positive_uncertainty_in_dataset, dir=args.sta_save_dir, name='all positive uncertainty')
    pltsave(tp_uncertainty_in_dataset, dir=args.sta_save_dir, name='True Positive uncertainty')
    pltsave(fn_uncertainty_in_dataset, dir=args.sta_save_dir, name='False Negative uncertainty')
    pltsave(fp_uncertainty_in_dataset, dir=args.sta_save_dir, name='False Positive uncertainty')

    fp_uncertainty_in_dataset_filtered = gaussian_filter1d(fp_uncertainty_in_dataset, sigma=3)
    tp_uncertainty_in_dataset_filtered = gaussian_filter1d(tp_uncertainty_in_dataset, sigma=3)
    fn_uncertainty_in_dataset_filtered = gaussian_filter1d(fn_uncertainty_in_dataset, sigma=3)
    fp_and_fn = fp_uncertainty_in_dataset_filtered + fn_uncertainty_in_dataset_filtered

    pltsave(fp_and_fn, dir=args.sta_save_dir, name='FP & FN uncertainty filtered')
    pltsave(tp_uncertainty_in_dataset_filtered, dir=args.sta_save_dir, name='True Positive uncertainty filtered')
    pltsave(fn_uncertainty_in_dataset_filtered, dir=args.sta_save_dir, name='False Negative uncertainty filtered')
    pltsave(fp_uncertainty_in_dataset_filtered, dir=args.sta_save_dir, name='False Positive uncertainty filtered')

    fp_mean, fp_var = find_mean_and_var(fp_uncertainty_in_dataset_filtered, start=args.bins / 10, end=args.bins,
                                        bins=args.bins, bin_range=args.bin_range)
    tp_mean, tp_var = find_mean_and_var(tp_uncertainty_in_dataset_filtered, start=args.bins / 10, end=args.bins,
                                        bins=args.bins,
                                        bin_range=args.bin_range)
    fn_mean, fn_var = find_mean_and_var(fn_uncertainty_in_dataset_filtered, start=args.bins / 10, end=args.bins,
                                        bins=args.bins,
                                        bin_range=args.bin_range)
    logger.write_and_print('max uncertainty is {0}  '.format(uncertainty_max))
    logger.write_and_print('fp uncertainty mean is {0}  variance is {1}'.format(fp_mean, fp_var))
    logger.write_and_print('tp uncertainty mean is {0}  variance is {1}'.format(tp_mean, tp_var))
    logger.write_and_print('fn uncertainty mean is {0}  variance is {1}'.format(fn_mean, fn_var))

    return


if __name__ == '__main__':
    args = ParseArguments()

    UncertaintySTA(args)
