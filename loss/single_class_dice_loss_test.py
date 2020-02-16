import argparse
import os
import time
import torch

from loss.single_class_dice_loss import SingleClassDiceLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_test',
                        type=int,
                        default=100,
                        help='number of test')

    parser.add_argument('--batch_size',
                        type=int,
                        default=48,
                        help='number of patches in each batch')

    parser.add_argument('--num_channels',
                        type=int,
                        default=1,
                        help='1 for grayscale, 3 for RGB images')

    parser.add_argument('--height',
                        type=int,
                        default=112,
                        help='the pixels of patch height')

    parser.add_argument('--width',
                        type=int,
                        default=112,
                        help='the pixels of patch width')

    args = parser.parse_args()

    return args


def TestSingleClassDiceLoss(args):
    for i in range(args.num_test):
        start_time = time.time()

        residues = torch.rand(args.batch_size, args.num_channels, args.height, args.width).cuda()

        pixel_level_labels = torch.rand(args.batch_size, args.height, args.width)
        pixel_level_labels[pixel_level_labels <= 0.5] = 0
        pixel_level_labels[pixel_level_labels > 0.5] = 1

        loss = SingleClassDiceLoss()(residues, pixel_level_labels)
        print('consuming time: {:.4f}s; loss: {:.4f}'.format(time.time() - start_time, loss.item()))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestSingleClassDiceLoss(args)
