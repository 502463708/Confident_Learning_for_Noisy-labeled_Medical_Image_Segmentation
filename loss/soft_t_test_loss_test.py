import argparse
import os
import time
import torch

from loss.soft_t_test_loss import SoftTTestLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_test',
                        type=int,
                        default=100,
                        help='number of test')

    parser.add_argument('--batch_size',
                        type=int,
                        default=3,
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

    parser.add_argument('--beta',
                        type=float,
                        default=0.8,
                        help='beta value between 0 to 1')

    parser.add_argument('--lambda_p',
                        type=float,
                        default=1,
                        help='positive lambda value between 0 to 1')

    parser.add_argument('--lambda_n',
                        type=float,
                        default=0.1,
                        help='negative beta value between 0 to 1')

    args = parser.parse_args()

    return args


def TestSoftTTestLoss(args):
    for i in range(args.num_test):
        loss_func = SoftTTestLoss(args.beta, args.lambda_p, args.lambda_n)

        start_time = time.time()
        residues = torch.rand(args.batch_size, args.num_channels, args.height, args.width).cuda()

        pixel_level_labels = torch.rand(args.batch_size, args.height, args.width).cuda()
        pixel_level_labels[pixel_level_labels <= 0.5] = 0
        pixel_level_labels[pixel_level_labels > 0.5] = 1
        pixel_level_labels = pixel_level_labels.long()

        loss = loss_func(residues, pixel_level_labels)
        print('time:', time.time() - start_time, 'loss: ', loss.item())
        print('\n')

    return


if __name__ == '__main__':
    args = ParseArguments()
    TestSoftTTestLoss(args)
