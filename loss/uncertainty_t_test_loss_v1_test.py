import argparse
import os
import time
import torch

from loss.uncertainty_t_test_loss_v1 import UncertaintyTTestLossV1

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

    parser.add_argument('--u_low',
                        type=float,
                        default=0.02,
                        help='The lower bound of uncertainty.')

    parser.add_argument('--u_up',
                        type=float,
                        default=0.1,
                        help='The upper bound of uncertainty.')

    parser.add_argument('--w_low',
                        type=float,
                        default=0.2,
                        help='The lower bound of weights.')

    parser.add_argument('--w_up',
                        type=float,
                        default=0.8,
                        help='The upper bound of weights.')

    args = parser.parse_args()

    return args


def TestUncertaintyTTestLossV1(args):
    for i in range(args.num_test):
        loss_func = UncertaintyTTestLossV1(args.beta, args.lambda_p, args.lambda_n, args.u_low, args.u_up, args.w_low,
                                           args.w_up)

        start_time = time.time()
        residues = torch.rand(args.batch_size, args.num_channels, args.height, args.width).cuda()

        pixel_level_labels = torch.rand(args.batch_size, args.height, args.width).cuda()
        pixel_level_labels[pixel_level_labels <= 0.5] = 0
        pixel_level_labels[pixel_level_labels > 0.5] = 1
        pixel_level_labels = pixel_level_labels.long()

        uncertainty_maps = torch.rand(args.batch_size, args.height, args.width).cuda()

        loss = loss_func(residues, pixel_level_labels, uncertainty_maps)
        print('time:', time.time() - start_time, 'loss: ', loss.item())
        print('\n')

    return


if __name__ == '__main__':
    args = ParseArguments()
    TestUncertaintyTTestLossV1(args)
