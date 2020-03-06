import argparse
import cv2
import numpy as np
import os
import shutil

from config.config_confident_learning_pixel_level_classification import cfg
from dataset.dataset_confident_learning_2d import ConfidentLearningDataset2d
from torch.utils.data import DataLoader


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                        type=str,
                        default='training',
                        help='Only training or validation is supported.')

    parser.add_argument('--class_name',
                        type=str,
                        default='clavicle',
                        help='Only clavicle, heart or lung is supported.')

    parser.add_argument('--num_epoch',
                        type=int,
                        default=1,
                        help='The epoch number for test.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=480,
                        help='The patch number in each batch.')

    parser.add_argument('--num_workers',
                        type=int,
                        default=24,
                        help='The number of thread.')

    parser.add_argument('--saving_root_dir',
                        type=str,
                        default='/data1/minqing/results/JRST-dataset-test/',
                        help='Output saving root dir.')

    args = parser.parse_args()

    # remove the existing folder with the same name
    if os.path.isdir(args.saving_root_dir):
        shutil.rmtree(args.saving_root_dir)

    # create a new folder
    os.mkdir(args.saving_root_dir)

    return args


def ConfidentLearningDataset2dTest(args):
    # create dataset
    dataset = ConfidentLearningDataset2d(data_root_dir=cfg.general.data_root_dir,
                                         mode=args.mode,
                                         class_name=args.class_name,
                                         enable_random_sampling=False,
                                         image_channels=cfg.dataset.image_channels,
                                         cropping_size=cfg.dataset.cropping_size,
                                         load_confident_map=cfg.dataset.load_confident_map,
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
        output_dir_epoch = os.path.join(args.saving_root_dir, 'epoch_{0}'.format(epoch_idx))
        os.mkdir(output_dir_epoch)

        print('-------------------------------------------------------------------------------------------------------')
        print('Loading epoch {0}...'.format(epoch_idx))

        # the following two variables are used for counting positive and negative patch number in an epoch
        image_num_for_this_epoch = 0

        for batch_idx, (images_tensor, pixel_level_labels_tensor, confident_maps_tensor, filenames) in enumerate(
                data_loader):
            # create folder for this batch
            output_dir_batch = os.path.join(output_dir_epoch, 'batch_{0}'.format(batch_idx))
            os.mkdir(output_dir_batch)

            # the following two variables are used for counting positive and negative patch number in a batch
            image_num_for_this_batch = 0

            images_np = images_tensor.cpu().numpy()
            pixel_level_labels_np = pixel_level_labels_tensor.cpu().numpy()
            confident_maps_np = confident_maps_tensor.cpu().numpy()

            for image_idx in range(images_np.shape[0]):
                image_np = images_np[image_idx, 0, :, :]
                pixel_level_label_np = pixel_level_labels_np[image_idx, :, :]
                confident_map_np = confident_maps_np[image_idx, :, :]
                filename = filenames[image_idx]

                image_np *= 255
                image_np = image_np.astype(np.uint8)

                pixel_level_label_np = pixel_level_label_np.astype(np.float)
                pixel_level_label_np *= 255
                pixel_level_label_np = pixel_level_label_np.astype(np.uint8)

                confident_map_np *= 255
                confident_map_np = confident_map_np.astype(np.uint8)

                cv2.imwrite(os.path.join(output_dir_batch, filename), image_np)
                cv2.imwrite(os.path.join(output_dir_batch, filename.replace('.png', '_mask.png')),
                            pixel_level_label_np)
                cv2.imwrite(os.path.join(output_dir_batch, filename.replace('.png', '_confident_map.png')),
                            confident_map_np)

                image_num_for_this_epoch += 1
                image_num_for_this_batch += 1

            print('----batch {0} loading finished; patch count: {1}'.format(batch_idx,
                                                                            image_num_for_this_batch))

        print('epoch {0} loading finished; patch count: {1}'.format(epoch_idx,
                                                                    image_num_for_this_epoch))

    return


if __name__ == '__main__':
    args = ParseArguments()

    ConfidentLearningDataset2dTest(args)
