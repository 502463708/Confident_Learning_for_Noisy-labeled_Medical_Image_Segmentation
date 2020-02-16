import argparse
import os
import time
import torch

from loss.uncertainty_cross_entropy_loss_v1 import UncertaintyCrossEntropyLossV1

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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

    parser.add_argument('--upn',
                        type=int,
                        default=56 * 56,
                        help='uncertainty pixel number threshold')

    parser.add_argument('--num_classes',
                        type=int,
                        default=1,
                        help='1 for grayscale, 3 for RGB images')

    parser.add_argument('--height',
                        type=int,
                        default=112,
                        help='height of images')

    parser.add_argument('--width',
                        type=int,
                        default=112,
                        help='width of images')

    args = parser.parse_args()

    return args


def TestUncertaintyTTestLossV1(args):
    loss_func = UncertaintyCrossEntropyLossV1(args.upn)

    for i in range(args.num_test):
        start_time = time.time()
        preds = torch.rand(args.batch_size, 2).cuda()

        labels = torch.rand(args.batch_size).cuda()
        labels[labels <= 0.5] = 0
        labels[labels > 0.5] = 1
        labels = labels.cuda()

        uncertainty_maps = torch.rand(args.batch_size, args.height, args.width).cuda()

        loss = loss_func(preds, labels, uncertainty_maps)

        print('time:', time.time() - start_time, 'loss: ', loss.item())
        print('\n')

    return


if __name__ == '__main__':
    args = ParseArguments()
    TestUncertaintyTTestLossV1(args)
