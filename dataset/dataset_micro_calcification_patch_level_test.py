import argparse
import cv2
import numpy as np
import os
import shutil

from config.config_micro_calcification_patch_level_reconstruction import cfg
from dataset.dataset_micro_calcification_patch_level import MicroCalcificationDataset
from torch.utils.data import DataLoader


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/results/Inbreast-patch-level-split-pos2neg-ratio-1-dataset-test/',
                        help='Destination data dir.')

    parser.add_argument('--mode',
                        type=str,
                        default='training',
                        help='Only training, validation or test is supported.')

    parser.add_argument('--load_uncertainty_map',
                        type=bool,
                        default=False,
                        help='Whether load uncertainty maps.')

    parser.add_argument('--num_epoch',
                        type=int,
                        default=5,
                        help='The epoch number for test.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=480,
                        help='The patch number in each batch.')

    parser.add_argument('--num_workers',
                        type=int,
                        default=24,
                        help='The number of thread.')

    args = parser.parse_args()

    # remove the existing folder with the same name
    if os.path.isdir(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)

    # create a new folder
    os.mkdir(args.dst_data_root_dir)

    return args


def MicroCalcificationPatchLevelDatasetTest(args):
    # create dataset
    dataset = MicroCalcificationDataset(data_root_dir=cfg.general.data_root_dir,
                                        mode=args.mode,
                                        enable_random_sampling=False,
                                        pos_to_neg_ratio=cfg.dataset.pos_to_neg_ratio,
                                        image_channels=cfg.dataset.image_channels,
                                        cropping_size=cfg.dataset.cropping_size,
                                        dilation_radius=cfg.dataset.dilation_radius,
                                        load_uncertainty_map=args.load_uncertainty_map,
                                        calculate_micro_calcification_number=cfg.dataset.calculate_micro_calcification_number,
                                        enable_data_augmentation=cfg.dataset.augmentation.enable_data_augmentation,
                                        enable_vertical_flip=cfg.dataset.augmentation.enable_vertical_flip,
                                        enable_horizontal_flip=cfg.dataset.augmentation.enable_horizontal_flip)

    # create data loader for training
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)

    # enumerating
    for epoch_idx in range(args.num_epoch):
        # create folder for this epoch
        output_dir_epoch = os.path.join(args.dst_data_root_dir, 'epoch_{0}'.format(epoch_idx))
        os.mkdir(output_dir_epoch)

        print('-------------------------------------------------------------------------------------------------------')
        print('Loading epoch {0}...'.format(epoch_idx))

        # the following two variables are used for counting positive and negative patch number in an epoch
        positive_patch_num_for_this_epoch = 0
        negative_patch_num_for_this_epoch = 0

        for batch_idx, (images_tensor, pixel_level_labels_tensor, pixel_level_labels_dilated_tensor,
                        uncertainty_maps_tensor, image_level_labels_tensor, _, filenames) in enumerate(data_loader):
            # create folder for this batch
            output_dir_batch = os.path.join(output_dir_epoch, 'batch_{0}'.format(batch_idx))
            os.mkdir(output_dir_batch)

            # create folder for saving positive patches
            output_dir_positive = os.path.join(output_dir_batch, 'positive')
            os.mkdir(output_dir_positive)

            # create folder for saving negative patches
            output_dir_negative = os.path.join(output_dir_batch, 'negative')
            os.mkdir(output_dir_negative)

            # the following two variables are used for counting positive and negative patch number in a batch
            positive_patch_num_for_this_batch = 0
            negative_patch_num_for_this_batch = 0

            images_np = images_tensor.cpu().numpy()
            pixel_level_labels_np = pixel_level_labels_tensor.cpu().numpy()
            pixel_level_labels_dilated_np = pixel_level_labels_dilated_tensor.cpu().numpy()
            uncertainty_maps_np = uncertainty_maps_tensor.cpu().numpy()
            image_level_labels_np = image_level_labels_tensor.cpu().numpy()

            for image_idx in range(images_np.shape[0]):
                image_np = images_np[image_idx, 0, :, :]
                pixel_level_label_np = pixel_level_labels_np[image_idx, :, :]
                pixel_level_label_dilated_np = pixel_level_labels_dilated_np[image_idx, :, :]
                uncertainty_map_np = uncertainty_maps_np[image_idx, :, :]
                image_level_label = image_level_labels_np[image_idx, 0]
                filename = filenames[image_idx]

                image_np *= 255
                image_np = image_np.astype(np.uint8)

                pixel_level_label_np *= 255
                pixel_level_label_np = pixel_level_label_np.astype(np.uint8)

                pixel_level_label_dilated_np *= 255
                pixel_level_label_dilated_np = pixel_level_label_dilated_np.astype(np.uint8)

                uncertainty_map_np *= 255
                uncertainty_map_np = uncertainty_map_np.astype(np.uint8)
                uncertainty_map_np = cv2.applyColorMap(uncertainty_map_np, cv2.COLORMAP_JET)

                # image_level_label is either 0 or 1
                assert image_level_label in [0, 1]

                if image_level_label == 1:
                    cv2.imwrite(os.path.join(output_dir_positive, filename), image_np)
                    cv2.imwrite(os.path.join(output_dir_positive, filename.replace('.png', '_mask.png')),
                                pixel_level_label_np)
                    cv2.imwrite(os.path.join(output_dir_positive, filename.replace('.png', '_dilated_mask.png')),
                                pixel_level_label_dilated_np)
                    cv2.imwrite(os.path.join(output_dir_positive, filename.replace('.png', '_uncertainty_map.png')),
                                uncertainty_map_np)
                    positive_patch_num_for_this_epoch += 1
                    positive_patch_num_for_this_batch += 1
                elif image_level_label == 0:
                    cv2.imwrite(os.path.join(output_dir_negative, filename), image_np)
                    cv2.imwrite(os.path.join(output_dir_negative, filename.replace('.png', '_mask.png')),
                                pixel_level_label_np)
                    cv2.imwrite(os.path.join(output_dir_negative, filename.replace('.png', '_dilated_mask.png')),
                                pixel_level_label_dilated_np)
                    cv2.imwrite(os.path.join(output_dir_negative, filename.replace('.png', '_uncertainty_map.png')),
                                uncertainty_map_np)
                    negative_patch_num_for_this_epoch += 1
                    negative_patch_num_for_this_batch += 1

            print('----batch {0} loading finished; '
                  'positive patches: {1}, negative patches: {2}'.format(batch_idx,
                                                                        positive_patch_num_for_this_batch,
                                                                        negative_patch_num_for_this_batch))

        print('epoch {0} loading finished; '
              'positive patches: {1}, negative patches: {2}'.format(epoch_idx,
                                                                    positive_patch_num_for_this_epoch,
                                                                    negative_patch_num_for_this_epoch))

    return


if __name__ == '__main__':
    args = ParseArguments()

    MicroCalcificationPatchLevelDatasetTest(args)
