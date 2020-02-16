import argparse
import numpy as np
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.getcwd()))

from config.config_micro_calcification_patch_level_reconstruction import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from metrics.metrics_patch_level_reconstruction import MetricsReconstruction
from net.vnet2d_v2 import VNet2d
from torch.utils.data import DataLoader
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cudnn.benchmark = True


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/home/groupprofzli/data1/dwz/data/Inbreast-patch-level-split-pos2neg-ratio-1-dataset-5764-1107/',
                        help='Source data dir.')
    parser.add_argument('--data_save_dir',
                        type=str,
                        default='/home/groupprofzli/data1/dwz/results/residues_distribution_statistics/',
                        help='statistics save dir')
    parser.add_argument('--model_saving_dir',
                        type=str,
                        default='/home/groupprofzli/data1/dwz/models/20191117_5764_uCs_reconstruction_ttestlossv3_default_dilation_radius_7/',
                        help='Model saved dir.')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='test',
                        help='The type of dataset to be evaluated (training, validation, test).')
    parser.add_argument('--epoch_idx',
                        type=int,
                        default=-1,
                        help='The epoch index of ckpt, set -1 to choose the best ckpt on validation set.')
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
    parser.add_argument('--batch_size',
                        type=int,
                        default=30,
                        help='Batch size for evaluation.')
    parser.add_argument('--slack_for_recall',
                        type=bool,
                        default=False,
                        help='The bool variable for slacking recall metric standard.')
    parser.add_argument('--histogram_bins',
                        type=int,
                        default=1000,
                        help='the number of bins in histogram')

    args = parser.parse_args()

    return args


def pltsave(hist, dir, name):
    plt.bar(np.arange(len(hist)), hist)
    plt.title("{} distribution".format(name))
    plt.savefig(os.path.join(dir, '{}.png'.format(name)))
    plt.close()


def RdsidueDistributionSTA(args):
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
    dataset = MicroCalcificationDataset(data_root_dir=args.data_root_dir,
                                        mode=args.dataset_type,
                                        enable_random_sampling=False,
                                        pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                        image_channels=cfg.dataset.image_channels,
                                        cropping_size=cfg.dataset.cropping_size,
                                        dilation_radius=args.dilation_radius,
                                        calculate_micro_calcification_number=cfg.dataset.calculate_micro_calcification_number,
                                        enable_data_augmentation=False)

    # create data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=cfg.train.num_threads)

    metrics = MetricsReconstruction(args.prob_threshold, args.area_threshold, args.distance_threshold,
                                    args.slack_for_recall)

    residue_in_dataset = np.zeros(args.histogram_bins)
    mask_positive_residue_in_dataset = np.zeros(args.histogram_bins)
    mask_negative_residue_in_dataset = np.zeros(args.histogram_bins)
    recon_positive_residue_in_dataset = np.zeros(args.histogram_bins)
    recon_negative_residue_in_dataset = np.zeros(args.histogram_bins)

    for batch_idx, (images_tensor, pixel_level_labels_tensor, pixel_level_labels_dilated_tensor,
                    image_level_labels_tensor, _, filenames) in enumerate(data_loader):
        # start time of this batch
        start_time_for_batch = time()

        # transfer the tensor into gpu device
        images_tensor = images_tensor.cuda()

        # network forward
        reconstructed_images_tensor, prediction_residues_tensor = net(images_tensor)

        # evaluation

        post_process_preds_np, calcification_num_batch_level, recall_num_batch_level, FP_num_batch_level = metrics.metric_batch_level(
            prediction_residues_tensor, pixel_level_labels_tensor)

        # dilated label  , predict label
        pixel_level_labels_dilated = pixel_level_labels_dilated_tensor.cpu().view(-1).numpy()
        process_preds = post_process_preds_np.reshape(-1)

        residues = prediction_residues_tensor.cpu().view(-1).detach().numpy()
        residues_hist, _ = np.histogram(residues, bins=args.histogram_bins, range=(0, 1))
        residue_in_dataset += residues_hist

        assert residues.shape == pixel_level_labels_dilated.shape
        assert residues.shape == process_preds.shape

        mask_positive_residue = residues[pixel_level_labels_dilated == 1]
        mask_positive_residue_hist, _ = np.histogram(mask_positive_residue, bins=args.histogram_bins, range=(0, 1))
        mask_positive_residue_in_dataset += mask_positive_residue_hist

        mask_negative_residue = residues[pixel_level_labels_dilated == 0]
        mask_negative_residue_hist, _ = np.histogram(mask_negative_residue, bins=args.histogram_bins, range=(0, 1))
        mask_negative_residue_in_dataset += mask_negative_residue_hist

        process_positive_residue = residues[process_preds == 1]
        process_positive_residue_hist, _ = np.histogram(process_positive_residue, bins=args.histogram_bins,
                                                        range=(0, 1))
        recon_positive_residue_in_dataset += process_positive_residue_hist

        process_negative_residue = residues[process_preds == 0]
        process_negative_residue_hist, _ = np.histogram(process_negative_residue, bins=args.histogram_bins,
                                                        range=(0, 1))
        recon_negative_residue_in_dataset += process_negative_residue_hist

    residue_in_dataset[residue_in_dataset > 15000] = 15000
    mask_negative_residue_in_dataset[mask_negative_residue_in_dataset > 15000] = 15000
    recon_negative_residue_in_dataset[recon_negative_residue_in_dataset > 15000] = 15000
    pltsave(residue_in_dataset, args.data_save_dir, 'total residues')
    pltsave(mask_positive_residue_in_dataset, args.data_save_dir, 'mask positive residues')
    pltsave(mask_negative_residue_in_dataset, args.data_save_dir, 'mask negative residues')
    pltsave(recon_positive_residue_in_dataset, args.data_save_dir, 'predict positive residues')
    pltsave(recon_negative_residue_in_dataset, args.data_save_dir, 'predict negative residues')

    print('on dataset {0} with {1} dilation {2} histogram bins'.format(args.dataset_type, args.dilation_radius,
                                                                       args.histogram_bins))
    print('the whole residues distribution is {}'.format(np.around(residue_in_dataset, 3)))
    print('in dilated mask label, the positive residues distribution is {0}\n'
          'the negative residues distribution is {1}.'.format(np.around(mask_positive_residue_hist, 3),
                                                              np.around(mask_negative_residue_hist, 3)))
    print('in predicted label, the positive residues distribution is {0}\n'
          'the negative residues distribution is {1}'.format(np.around(recon_positive_residue_in_dataset, 3),
                                                             np.around(recon_negative_residue_in_dataset, 3)))

    return


if __name__ == '__main__':
    args = ParseArguments()

    RdsidueDistributionSTA(args)
