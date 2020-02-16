import argparse
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from config.config_micro_calcification_patch_level_reconstruction import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from torch.utils.data import DataLoader


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/home/groupprofzli/data1/dwz/data/Inbreast-patch-level-split-pos2neg-ratio-1-dataset-5764-1107/',
                        help='Destination data dir.')

    parser.add_argument('--mode',
                        type=str,
                        default='training',
                        help='Only training, validation or test is supported.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=480,
                        help='The patch number in each batch.')

    parser.add_argument('--num_workers',
                        type=int,
                        default=24,
                        help='The number of thread.')

    parser.add_argument('--dilation_radius',
                        type=float,
                        default=7,
                        help='dilated radius length')

    parser.add_argument('--pos_to_neg_ratio',
                        type=float,
                        default=1,
                        help='the ratio of positive and negative patches number')

    args = parser.parse_args()

    return args


def SoftPositiveSTA(args):
    dataset = MicroCalcificationDataset(data_root_dir=cfg.general.data_root_dir,
                                        mode=args.mode,
                                        enable_random_sampling=False,
                                        pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                        image_channels=cfg.dataset.image_channels,
                                        cropping_size=cfg.dataset.cropping_size,
                                        dilation_radius=cfg.dataset.dilation_radius,
                                        calculate_micro_calcification_number=cfg.dataset.calculate_micro_calcification_number,
                                        enable_data_augmentation=cfg.dataset.augmentation.enable_data_augmentation,
                                        enable_vertical_flip=cfg.dataset.augmentation.enable_vertical_flip,
                                        enable_horizontal_flip=cfg.dataset.augmentation.enable_horizontal_flip)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)
    # enumerating

    print('-------------------------------------------------------------------------------------------------------')
    print('Loading on {0} dataset...'.format(args.mode))

    # the following two variables are used for counting positive and negative patch number in an epoch
    positive_patch_num_total = 0
    negative_patch_num_total = 0
    pixels_total = 0
    label_pixels_total = 0
    dilated_label_pixel_total = 0

    for batch_idx, (
            images_tensor, pixel_level_labels_tensor, pixel_level_labels_dilated_tensor, image_level_labels_tensor,
            _,
            filenames) in enumerate(data_loader):
        images_np = images_tensor.cpu().view(-1).numpy()
        pixel_level_labels_np = pixel_level_labels_tensor.cpu().view(-1).numpy()
        pixel_level_labels_dilated_np = pixel_level_labels_dilated_tensor.cpu().view(-1).numpy()
        image_level_labels_np = image_level_labels_tensor.cpu().view(-1).numpy()

        pixels_total += images_np.shape[0]
        positve_patches_num = sum(image_level_labels_np[image_level_labels_np == 1])
        positive_patch_num_total += positve_patches_num
        negative_patch_num = (image_level_labels_np.shape[0] - positve_patches_num)
        negative_patch_num_total += negative_patch_num
        labels = pixel_level_labels_np
        labed_pixels = sum(labels[labels == 1])
        label_pixels_total += labed_pixels
        dilated_labels = pixel_level_labels_dilated_np
        dilated_label_pixels = sum(dilated_labels[dilated_labels == 1])
        dilated_label_pixel_total += dilated_label_pixels

        print('----batch {0} loading finished; '
              'positive patches: {1}, negative patches: {2} '
              'labeled pixels number is {3}, dilated labeled pixels number is {4}, total pixels number is {5}'.format(
            batch_idx, positve_patches_num, negative_patch_num, labed_pixels, dilated_label_pixels, images_np.shape[0]))

        print()

    soft_negative_ratio = (dilated_label_pixel_total - label_pixels_total) / (pixels_total - label_pixels_total)
    print('on dataset {0} with {1} dilation loading finished; \n'
          'total patches: {2}, positive patches: {3}, negative patches: {4}\n'
          'total pixels number is {5} \n'
          'masked label pixels number is {6}\n'
          'dilated pixels number is {7}\n'
          'soft label pixels number is {8}\n'
          'the ratio of soft positive and negative pixels is {9}'.format(args.mode, args.dilation_radius,
                                                                         positive_patch_num_total + negative_patch_num_total,
                                                                         positive_patch_num_total,
                                                                         negative_patch_num_total, pixels_total,
                                                                         label_pixels_total, dilated_label_pixel_total,
                                                                         dilated_label_pixel_total - label_pixels_total,
                                                                         soft_negative_ratio))

    return


if __name__ == '__main__':
    args = ParseArguments()
    SoftPositiveSTA(args)
