import argparse
import cv2
import numpy as np
import os
import shutil

from dataset.dataset_micro_calcification_radiograph_level import MicroCalcificationRadiographLevelDataset
from torch.utils.data import DataLoader


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_data_root_dir',
                        type=str,
                        default='/data/lars/data/Inbreast-radiograph-level-roi-extracted-data-split-dataset/',
                        help='Source data root dir.')

    parser.add_argument('--dst_data_root_dir',
                        type=str,
                        default='/data/lars/results/Inbreast-radiograph-level-roi-extracted-data-split-dataset-test/',
                        help='Destination data root dir.')

    parser.add_argument('--mode',
                        type=str,
                        default='test',
                        help='Only training, validation or test is supported.')

    parser.add_argument('--num_epoch',
                        type=int,
                        default=1,
                        help='The epoch numbner for test.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Only batch_size=1 is supported.')

    parser.add_argument('--num_workers',
                        type=int,
                        default=24,
                        help='The number of thread.')

    args = parser.parse_args()

    assert os.path.exists(args.src_data_root_dir)

    # remove the existing folder with the same name
    if os.path.isdir(args.dst_data_root_dir):
        shutil.rmtree(args.dst_data_root_dir)

    # create a new folder
    os.mkdir(args.dst_data_root_dir)

    return args


def MicroCalcificationRadiographLevelDatasetTest(args):
    # create dataset
    dataset = MicroCalcificationRadiographLevelDataset(data_root_dir=args.src_data_root_dir,
                                                       mode=args.mode)

    # create data loader
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)

    # enumerating
    for epoch_idx in range(args.num_epoch):
        # create folder for this epoch
        output_dir_epoch = os.path.join(args.dst_data_root_dir, 'epoch_{0}'.format(epoch_idx))
        os.mkdir(output_dir_epoch)

        # create folder for saving positive radiographs
        output_dir_positive = os.path.join(output_dir_epoch, 'positive')
        os.mkdir(output_dir_positive)

        # create folder for saving negative radiographs
        output_dir_negative = os.path.join(output_dir_epoch, 'negative')
        os.mkdir(output_dir_negative)

        print('-------------------------------------------------------------------------------------------------------')
        print('Loading epoch {0}...'.format(epoch_idx))

        # the following two variables are used for counting positive and negative radiograph number in an epoch
        positive_radiograph_num_for_this_epoch = 0
        negative_radiograph_num_for_this_epoch = 0

        for batch_idx, (
        images_tensor, pixel_level_labels_tensor, radiograph_level_labels_tensor, filenames) in enumerate(data_loader):
            # the following two variables are used for counting positive and negative radiograph number in a batch
            positive_radiograph_num_for_this_batch = 0
            negative_radiograph_num_for_this_batch = 0

            images_np = images_tensor.cpu().numpy()
            pixel_level_labels_np = pixel_level_labels_tensor.cpu().numpy()
            radiograph_level_labels_np = radiograph_level_labels_tensor.cpu().numpy()

            for image_idx in range(images_np.shape[0]):
                image_np = images_np[image_idx, 0, :, :]
                pixel_level_label_np = pixel_level_labels_np[image_idx, :, :]
                radiograph_level_label = radiograph_level_labels_np[image_idx, 0]
                filename = filenames[image_idx]

                image_np *= 255
                image_np = image_np.astype(np.uint8)

                # process pixel-level label                            # normal tissue: 0 (.png) <- 0 (tensor)
                pixel_level_label_np[pixel_level_label_np == 1] = 255  # micro calcification: 255 (.png) <- 1 (tensor)
                pixel_level_label_np[pixel_level_label_np == 2] = 165  # other lesion: 165 (.png) <- 2 (tensor)
                pixel_level_label_np[pixel_level_label_np == 3] = 85   # background: 85 (.png) <- 3 (tensor)
                pixel_level_label_np = pixel_level_label_np.astype(np.uint8)

                # radiograph_level_label is either 0 or 1
                assert radiograph_level_label in [0, 1]

                if radiograph_level_label == 1:
                    cv2.imwrite(os.path.join(output_dir_positive, filename), image_np)
                    cv2.imwrite(os.path.join(output_dir_positive, filename.replace('.png', '_mask.png')),
                                pixel_level_label_np)
                    positive_radiograph_num_for_this_epoch += 1
                    positive_radiograph_num_for_this_batch += 1
                elif radiograph_level_label == 0:
                    cv2.imwrite(os.path.join(output_dir_negative, filename), image_np)
                    cv2.imwrite(os.path.join(output_dir_negative, filename.replace('.png', '_mask.png')),
                                pixel_level_label_np)
                    negative_radiograph_num_for_this_epoch += 1
                    negative_radiograph_num_for_this_batch += 1

            print('----batch {0} loading finished; '
                  'positive patches: {1}, negative patches: {2}'.format(batch_idx,
                                                                        positive_radiograph_num_for_this_batch,
                                                                        negative_radiograph_num_for_this_batch))

        print('epoch {0} loading finished; '
              'positive patches: {1}, negative patches: {2}'.format(epoch_idx,
                                                                    positive_radiograph_num_for_this_epoch,
                                                                    negative_radiograph_num_for_this_epoch))

    return


if __name__ == '__main__':
    args = ParseArguments()

    MicroCalcificationRadiographLevelDatasetTest(args)
